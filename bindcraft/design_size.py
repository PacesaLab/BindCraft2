import sys
from dataclasses import replace

import jax

from bindcraft.af2 import GRADIENT_MEMORY_SHARE, MONOMER_POOL, MULTIMER_POOL, campaign_length_bucket, is_out_of_memory, worker_memory_budget_bytes
from bindcraft.campaign import campaign_design_model, campaign_design_plan
from bindcraft.model_weights import model_weights
from bindcraft.protein_preparation import design_residue_count, initialize_design_trajectory
from bindcraft.settings import build_design_settings, parse_setting_overrides, read_settings, select_design_and_validation_models

CALIBRATION_TOTALS = (416, 544)
SMALLEST_SEARCHABLE_TOTAL = 416
LARGEST_SEARCHABLE_TOTAL = 2560
REFIT_ROUNDS = 3
DID_NOT_FIT = float('inf')

def binder_length_for_total(settings: dict, padded_total: int) -> int:
    bucket = campaign_length_bucket(settings)
    length = padded_total - design_residue_count({**settings, 'binder_lengths': [bucket]}) + bucket
    return max(bucket, (length // bucket) * bucket)

def gradient_fits(design_model, settings: dict, binder_length: int, key, compile_only: bool) -> bool:
    probe_settings = build_design_settings({**settings, 'binder_lengths': [binder_length]})
    protein_states, _multi_chain_binders, losses = initialize_design_trajectory(probe_settings, key)
    try:
        design_model.sequence_gradients(protein_states, losses, model=design_model.models[0], compile_only=compile_only)
    except Exception as gradient_failure:
        if not is_out_of_memory(gradient_failure):
            raise
        return False
    return True

def gradient_bytes(design_model, settings: dict, binder_length: int, key) -> float:
    if not gradient_fits(design_model, settings, binder_length, key, True):
        return DID_NOT_FIT
    return design_model.last_gradient_bytes

def quadratic_through(first: tuple[int, float], second: tuple[int, float]) -> tuple[float, float]:
    (first_total, first_bytes), (second_total, second_bytes) = first, second
    slope = (second_bytes - first_bytes) / (second_total ** 2 - first_total ** 2)
    return slope, first_bytes - slope * first_total ** 2

def solved_total(slope: float, intercept: float, allowance: float) -> int:
    return int(max(0.0, (allowance - intercept) / slope) ** 0.5) if slope > 0 else 0

def report_lines(settings: dict, design_model, allowance: float, budget: float, key) -> list[str]:
    bucket = campaign_length_bucket(settings)
    total_at = lambda length: design_residue_count({**settings, 'binder_lengths': [length]})
    measured: dict[int, float] = {}
    routes: dict[int, str] = {}

    def measure(binder_length: int) -> float:
        if binder_length not in measured:
            measured[binder_length] = gradient_bytes(design_model, settings, binder_length, key)
            routes[binder_length] = design_model.last_attention_backend
        return measured[binder_length]

    for total in CALIBRATION_TOTALS:
        measure(binder_length_for_total(settings, total))

    slope = 0.0
    candidate = binder_length_for_total(settings, SMALLEST_SEARCHABLE_TOTAL)
    for _ in range(REFIT_ROUNDS):
        latest_route = routes[max(measured, key=total_at)]
        fitted = sorted((total_at(length), claimed) for length, claimed in measured.items()
                        if claimed != DID_NOT_FIT and routes[length] == latest_route)
        if len(fitted) < 2 or fitted[-1][0] == fitted[-2][0]:
            break
        slope, intercept = quadratic_through(fitted[-2], fitted[-1])
        solved = min(LARGEST_SEARCHABLE_TOTAL, max(SMALLEST_SEARCHABLE_TOTAL, solved_total(slope, intercept, allowance)))
        candidate = binder_length_for_total(settings, solved)
        if candidate in measured:
            break
        measure(candidate)

    while candidate > bucket and measure(candidate) > allowance:
        candidate -= bucket
    while total_at(candidate + bucket) <= LARGEST_SEARCHABLE_TOTAL and measure(candidate + bucket) <= allowance:
        candidate += bucket

    largest, first_refused = candidate, candidate + bucket
    while largest > bucket and not gradient_fits(design_model, settings, largest, key, False):
        first_refused, largest = largest, largest - bucket

    refused_reason = ('needs more than this card will compile)' if measured.get(first_refused) == DID_NOT_FIT
                      else f'claimed {measured[first_refused] / 1e9:.2f} GB)' if first_refused in measured
                      else 'its gradient step ran out of memory)')
    searched_out = total_at(first_refused) > LARGEST_SEARCHABLE_TOTAL
    return [f'card budget for one worker: {budget / 1e9:.1f} GB, of which the gradient may claim {allowance / 1e9:.1f} GB',
            f'calibrated here: {slope:,.0f} bytes per residue-pair over {len(measured)} compiles, attention route {routes[largest]}',
            f'{"this card holds at least" if searched_out else "largest binder that fits:"} {largest} aa '
            f'({total_at(largest)} padded residues, claimed {measured[largest] / 1e9:.2f} GB, gradient step ran)',
            f'the search stops there, so the real ceiling is higher and was not compiled' if searched_out
            else f'first length that does not: {first_refused} aa ({total_at(first_refused)} padded residues, {refused_reason}',
            f'set it with: --set binder_lengths=[{largest}]']

def main(arguments: list[str] | None=None) -> None:
    from bindcraft.cli import split_campaign_presets, split_metadata_file, split_setting_overrides
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    settings_paths, assignments = split_setting_overrides(arguments)
    settings_paths, preset_assignments = split_campaign_presets(settings_paths)
    settings_paths, _metadata_path = split_metadata_file(settings_paths)
    if not settings_paths:
        print('usage: bindcraft design_size <settings.json> [--core NAME] [--modality NAME] [--set KEY=VALUE]...', file=sys.stderr)
        raise SystemExit(2)
    settings = read_settings(settings_paths[0], parse_setting_overrides(preset_assignments + assignments))
    budget = worker_memory_budget_bytes()
    af2_weights, _mpnn_weights = model_weights()
    selected_models = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)
    design_plan = replace(campaign_design_plan(settings), subbatch_size=settings.get('subbatch_size', 'auto'))
    design_model = campaign_design_model(design_plan, selected_models.design_models, af2_weights, settings)
    key = jax.random.PRNGKey(settings.get('campaign_seed') or 0)
    print('\n'.join(report_lines(settings, design_model, GRADIENT_MEMORY_SHARE * budget, budget, key)), flush=True)

if __name__ == '__main__':
    main()
