import csv
import itertools
import json
import math
import os
import random
import statistics
from bindcraft.campaign_log import campaign_label
from bindcraft.campaign_output import CampaignProgress, TRAJECTORY_STAGE, accepted_table, numeric_value, per_target_readings, read_metric_rows, recorded_trajectory_rows, stage_table
from bindcraft.epitope_targeting import EPITOPE_CUTOFF
from bindcraft.loss import build_losses
from bindcraft.protein import BINDER_ALONE
from bindcraft.settings import BinderDesignSettings, DEFAULT_SETTINGS, design_stage_rounds, detarget_state_names, known_campaign_settings

SWEEP_FILENAME = 'sweep.csv'
BEST_SETTINGS_FILENAME = 'best_settings.json'
BASELINE_ARM = 'baseline'
SWEEP_OBJECTIVE = 'paired change in trajectory interface pTM, then accepted designs per trajectory, then mean accepted i_pTM'
SWEEP_COLUMNS = ('arm', 'rank', 'trajectories', 'accepted_designs', 'accepted_per_trajectory', 'accepted_interface_ptm', 'trajectory_interface_ptm', 'paired_trajectories', 'paired_change', 'paired_error', 'resolved')
DEFAULT_PARAMETER_SWEEP = {'axes': (), 'levels': (0.5, 2.0), 'max_arms': 5, 'block_trajectories': 10}
SWEEP_RESOLVING_ERRORS = 2.0
SWEEP_MIN_PAIRED_TRAJECTORIES = 10
AUTOTUNE_REVIEW_TRAJECTORIES = 10
AUTOTUNE_STAGE_STEP_CHANGE = 0.25
AUTOTUNE_WEIGHT_MULTIPLIERS = (0.5, 2.0)
AUTOTUNE_MIN_MEASURED_TRAJECTORIES = 5
AUTOTUNE_FINAL_THIRD_SHARE = (0.1, 0.5)
AUTOTUNE_STAGE_LENGTH_BAND = (0.5, 2.0)
AUTOTUNE_BLOCK_STAGE = 'screen'
FILTER_STAGE_AXES = ('screen_steps', 'refine_steps')
#forced targeting case
MODALITY_SWITCH_AXES = {'forced_targeting': 'forced_targeting_shell'}
AXIS_DEFAULTS = {'forced_targeting_shell': EPITOPE_CUTOFF}
DEFAULT_SWEEP_AXES = ('weights_binder_helicity', 'weights_interface_contacts', 'weights_compactness', 'weights_binder_pae', 'weights_interface_pae')
AXIS_SWEEP_VALUES = {'weights_binder_helicity': (-0.6, 0.3)}
SHARED_MODEL_SETTINGS = ('design_models', 'validation_models', 'validation_model', 'design_recycles', 'validation_recycles', 'subbatch_size', 'attention_backend', 'use_cueq', 'length_bucket_size', 'cyclic_offset_mode', 'cyclize_peptide', 'mpnn_model', 'mpnn_variant', 'aa_bias', 'copies', 'oligomer_tie')

def parameter_sweep_options(settings: dict) -> dict | None:
    configured = settings.get('parameter_sweep')
    if not isinstance(configured, dict) and not configured:
        return None
    return {**DEFAULT_PARAMETER_SWEEP, **(configured if isinstance(configured, dict) else {})}

def modality_loss_weight_axes(settings: dict) -> tuple[str, ...]:
    weights = {name: float(value) for name, value in settings.items() if name.startswith('weights_') and isinstance(value, (int, float)) and value and DEFAULT_SETTINGS.get(name) != value}
    return tuple(sorted(weights, key=lambda name: (-abs(weights[name]), name)))

def default_sweep_axes(settings: dict) -> tuple[str, ...]:
    return tuple(name for name in DEFAULT_SWEEP_AXES if isinstance(settings.get(name), (int, float)) and settings.get(name))

def parameter_sweep_levels(options: dict) -> tuple[float, ...]:
    return (float(options['multiplier']),) if options.get('multiplier') else tuple(float(level) for level in options['levels'])

def axis_sweep_values(name: str, options: dict) -> tuple[float, ...]:
    return () if tuple(options['axes']) else AXIS_SWEEP_VALUES.get(name, ())

def axis_arm_count(name: str, options: dict) -> int:
    return len(axis_sweep_values(name, options) or parameter_sweep_levels(options))

def parameter_sweep_axes(settings: dict, options: dict) -> tuple[str, ...]:
    derived = tuple(axis for switch, axis in MODALITY_SWITCH_AXES.items() if settings.get(switch)) + modality_loss_weight_axes(settings) + default_sweep_axes(settings) + tuple(name for name in FILTER_STAGE_AXES if settings.get(name))
    axes = tuple(dict.fromkeys(tuple(options['axes']) or derived))
    unread = [name for name in axes if name not in known_campaign_settings()]
    if unread:
        raise ValueError(f'parameter_sweep axes {unread} name no setting this codebase reads')
    shared = [name for name in axes if name in SHARED_MODEL_SETTINGS]
    if shared:
        raise ValueError(f'parameter_sweep axes {shared} are read by a model constructor that every arm shares')
    drawn = [name for name in axes if isinstance(settings.get(name), (list, tuple))]
    if drawn:
        raise ValueError(f'parameter_sweep axes {drawn} are drawn per trajectory from the array the campaign wrote, so a sweep has no single value of them to move')
    swept_arms = itertools.accumulate(axis_arm_count(name, options) for name in axes)
    return tuple(name for name, arms in zip(axes, swept_arms) if 1 + arms <= int(options['max_arms']))

def swept_value(settings: dict, name: str, level: float):
    stage_rounds = design_stage_rounds(settings)
    design_stage = name.removesuffix('_steps')
    current = stage_rounds[design_stage] if name.endswith('_steps') and design_stage in stage_rounds else settings.get(name, AXIS_DEFAULTS.get(name))
    return max(1, round(current * level)) if name.endswith('_steps') else current * level

def axis_sweep_arms(settings: dict, name: str, options: dict) -> tuple[tuple[str, dict], ...]:
    return tuple((f'{name}_{value:g}', {name: value}) for value in axis_sweep_values(name, options)) or tuple((f'{name}_x{level:g}', {name: swept_value(settings, name, level)}) for level in parameter_sweep_levels(options))

def parameter_sweep_arms(settings: dict) -> tuple[tuple[str, dict], ...]:
    options = parameter_sweep_options(settings)
    if options is None:
        return ()
    if not settings.get('max_trajectories'):
        raise ValueError('a parameter sweep splits max_trajectories over its arms, so a campaign asking for one has to set max_trajectories')
    arms = [(BASELINE_ARM, {})] + [arm for name in parameter_sweep_axes(settings, options) for arm in axis_sweep_arms(settings, name, options)]
    return tuple(arms) if len(arms) > 1 else ()

def sweep_block_budgets(arm_trajectories: int, block_trajectories: int) -> tuple[int, ...]:
    return tuple(range(block_trajectories, arm_trajectories, max(1, block_trajectories))) + (arm_trajectories,)

def arm_trajectory_budget(max_trajectories: int | None, arm_count: int) -> int | None:
    return max(1, max_trajectories // arm_count) if max_trajectories else None

def excluded_interface_states(design_settings: BinderDesignSettings) -> tuple[str, ...]:
    #detargets are meant to have no interface
    return (BINDER_ALONE, *detarget_state_names(design_settings))

def mean_metric_values(metric_rows: list[dict], selects_column, column_scale=lambda column: 1.0) -> float | None:
    selected = [(numeric_value(value), column_scale(column)) for row in metric_rows for column, value in row.items() if selects_column(column)]
    values = [value * scale for value, scale in selected if value is not None]
    return statistics.fmean(values) if values else None

def mean_accepted_interface_ptm(accepted_rows: list[dict], excluded_states: tuple[str, ...]) -> float | None:
    readings = [reading for row in accepted_rows for target_name, reading in per_target_readings(row, 'i_pTM').items() if target_name not in excluded_states]
    return statistics.fmean(readings) if readings else None

def mean_trajectory_interface_ptm(trajectory_rows: list[dict], excluded_states: tuple[str, ...]) -> float | None:
    readings = [reading for row in trajectory_rows for target_name, reading in per_target_readings(row, 'iptm').items() if target_name not in excluded_states]
    return statistics.fmean(readings) if readings else None

def trajectory_number(row: dict) -> int | None:
    number = str(row.get('trajectory') or row.get('design', '').rpartition('_')[2])
    return int(number) if number.isdigit() else None

def trajectory_interface_ptm_by_number(trajectory_rows: list[dict], excluded_states: tuple[str, ...]) -> dict[int, float]:
    scored = {number: mean_trajectory_interface_ptm([row], excluded_states) for row in trajectory_rows if (number := trajectory_number(row)) is not None}
    return {number: value for number, value in scored.items() if value is not None}

def paired_change(arm_scores: dict[int, float], baseline_scores: dict[int, float]) -> tuple[int, float | None, float | None]:
    changes = [arm_scores[number] - baseline_scores[number] for number in arm_scores.keys() & baseline_scores.keys()]
    if not changes:
        return (0, None, None)
    return (len(changes), statistics.fmean(changes), statistics.stdev(changes) / math.sqrt(len(changes)) if len(changes) > 1 else None)

def resolved_change(paired_trajectories: int, change: float | None, error: float | None) -> bool:
    return paired_trajectories >= SWEEP_MIN_PAIRED_TRAJECTORIES and change is not None and error is not None and abs(change) > SWEEP_RESOLVING_ERRORS * error

def swept_setting_names(arms: tuple[tuple[str, dict], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for _, overrides in arms for name in overrides))

def arm_result(project_folder: str, design_settings: BinderDesignSettings, arm: str, overrides: dict, setting_names: tuple[str, ...]) -> dict:
    settings = design_settings.settings
    arm_folder = os.path.join(project_folder, arm)
    trajectory_rows = read_metric_rows(stage_table(arm_folder, TRAJECTORY_STAGE))
    accepted_rows = read_metric_rows(accepted_table(arm_folder))
    excluded_states = excluded_interface_states(design_settings)
    return {'arm': arm, 'trajectories': len(trajectory_rows), 'accepted_designs': len(accepted_rows), 'accepted_per_trajectory': len(accepted_rows) / len(trajectory_rows) if trajectory_rows else None, 'accepted_interface_ptm': mean_accepted_interface_ptm(accepted_rows, excluded_states), 'trajectory_interface_ptm': mean_trajectory_interface_ptm(trajectory_rows, excluded_states), 'scores': trajectory_interface_ptm_by_number(trajectory_rows, excluded_states), **{name: overrides.get(name, settings.get(name, AXIS_DEFAULTS.get(name))) for name in setting_names}}

def arm_objective(result: dict) -> tuple[float, float, float]:
    return (result['paired_change'] or 0.0, result['accepted_per_trajectory'] or 0.0, result['accepted_interface_ptm'] or 0.0)

def write_sweep_record(project_folder: str, design_settings: BinderDesignSettings, arms: tuple[tuple[str, dict], ...], campaign: str | None=None) -> str | None:
    settings = design_settings.settings
    setting_names = swept_setting_names(arms)
    results = [arm_result(project_folder, design_settings, arm, overrides, setting_names) for arm, overrides in arms]
    if not any(result['trajectories'] for result in results):
        return None
    baseline_scores = dict(next((result['scores'] for result in results if result['arm'] == BASELINE_ARM), results[0]['scores']))
    for result in results:
        result['paired_trajectories'], result['paired_change'], result['paired_error'] = paired_change(result.pop('scores'), baseline_scores)
        result['resolved'] = resolved_change(result['paired_trajectories'], result['paired_change'], result['paired_error'])
    ranked = sorted(results, key=arm_objective, reverse=True)
    for rank, result in enumerate(ranked, start=1):
        result['rank'] = rank
    sweep_path = os.path.join(project_folder, SWEEP_FILENAME)
    partial_path = f'{sweep_path}.partial'
    with open(partial_path, 'w', newline='') as sweep_file:
        writer = csv.DictWriter(sweep_file, fieldnames=[*SWEEP_COLUMNS, *setting_names])
        writer.writeheader()
        writer.writerows(results)
    os.replace(partial_path, sweep_path)
    best = ranked[0]
    best_settings = {'campaign': campaign or campaign_label(settings) or os.path.basename(os.path.normpath(project_folder)), 'arm': best['arm'], 'objective': SWEEP_OBJECTIVE, 'resolved': best['resolved'], 'arms': len(results), 'trajectories_per_arm': best['trajectories'], 'paired_trajectories': best['paired_trajectories'], 'paired_change': best['paired_change'], 'paired_error': best['paired_error'], 'accepted_designs_in_arm': best['accepted_designs'], 'accepted_per_trajectory': best['accepted_per_trajectory'], 'accepted_interface_ptm': best['accepted_interface_ptm'], 'trajectory_interface_ptm': best['trajectory_interface_ptm'], 'settings': {name: best[name] for name in setting_names}}
    with open(os.path.join(project_folder, BEST_SETTINGS_FILENAME), 'w') as best_file:
        json.dump(best_settings, best_file, indent=2, sort_keys=False)
    return sweep_path

def design_loss_name(column: str) -> str:
    return column.partition('.')[2].partition('.')[0]

def configured_weight_scales(settings: dict, overrides: dict) -> dict[str, float]:
    return {name.removeprefix('weights_'): settings[name] / weight for name, weight in overrides.items() if name.startswith('weights_') and weight and settings.get(name)}

def mean_design_loss(metric_rows: list[dict], loss_names: frozenset[str], weight_scales: dict[str, float]) -> float | None:
    return mean_metric_values(metric_rows, lambda column: design_loss_name(column) in loss_names, lambda column: weight_scales.get(design_loss_name(column), 1.0))

def stage_loss_thirds(metric_rows: list[dict], design_stage: str, loss_names: frozenset[str], weight_scales: dict[str, float]) -> tuple[float, float, float] | None:
    stage_rows = [row for row in metric_rows if row.get('phase') == design_stage]
    final_third = len(stage_rows) // 3
    if not final_third:
        return None
    thirds = tuple(mean_design_loss(rows, loss_names, weight_scales) for rows in (stage_rows[:final_third], stage_rows[-2 * final_third:-final_third], stage_rows[-final_third:]))
    return None if None in thirds else thirds

def autotune_review(settings: dict, project_folder: str, record: dict, trajectory_rows: list[dict], accepted_designs: int) -> tuple[dict, list[str]]:
    loss_names = frozenset(build_losses(settings))
    weight_axes = modality_loss_weight_axes(settings) if settings.get('autotune_loss_weights') else ()
    weight_scales = configured_weight_scales(settings, record['overrides'])
    block = trajectory_rows[record['reviewed']:]
    block_losses = [recorded_trajectory_rows(project_folder, trajectory['design']) for trajectory in block]
    stage_outcomes = {design_stage: [(trajectory.get('terminated') == design_stage, thirds) for trajectory, metric_rows in zip(block, block_losses) if (thirds := stage_loss_thirds(metric_rows, design_stage, loss_names, weight_scales)) is not None] for design_stage in (axis.removesuffix('_steps') for axis in FILTER_STAGE_AXES)}
    block_closing = [closing for _, (_, _, closing) in stage_outcomes[AUTOTUNE_BLOCK_STAGE]]
    block_loss = statistics.fmean(block_closing) if len(block_closing) >= AUTOTUNE_MIN_MEASURED_TRAJECTORIES else None
    overrides, report, changes, drawn = ({name: value for name, value in record['overrides'].items() if name in FILTER_STAGE_AXES or name in weight_axes}, [], [], [])
    accepted_since = accepted_designs > record['accepted']
    reverting = record.get('alternating') and not accepted_since and block_loss is not None and record['loss'] is not None and (block_loss >= record['loss'])
    if reverting:
        reverted = [f"{name} {value:g} -> {record['restores'].get(name, settings[name]):g}" for name, value in overrides.items() if name in weight_axes and record['restores'].get(name) != value]
        overrides = {name: value for name, value in overrides.items() if name not in weight_axes} | record['restores']
        if reverted:
            report.append(f"autotune: undid {', '.join(reverted)}, mean end-of-{AUTOTUNE_BLOCK_STAGE} design loss {block_loss:.4f} against {record['loss']:.4f} at the campaign's own weights over the {len(block)} trajectories since")
    restores = {name: value for name, value in overrides.items() if name in weight_axes}
    if accepted_since:
        alternated = {name: overrides.pop(name) for name in weight_axes if name in overrides}
        changes += [f'{name} {weight:g} -> {settings[name]:g}, a design was accepted' for name, weight in alternated.items()]
    elif weight_axes and not reverting:
        alternation = random.Random(f"{settings.get('campaign_seed') or 0}:{len(trajectory_rows)}")
        for name in weight_axes:
            overrides[name] = settings[name] * alternation.choice(AUTOTUNE_WEIGHT_MULTIPLIERS)
            drawn.append(f'{name} {settings[name]:g} -> {overrides[name]:g}, {len(block)} trajectories accepted nothing')
        changes += drawn
    for axis in FILTER_STAGE_AXES:
        design_stage = axis.removesuffix('_steps')
        configured, measured = (settings.get(axis), stage_outcomes[design_stage])
        if not configured or len(measured) < AUTOTUNE_MIN_MEASURED_TRAJECTORIES:
            continue
        opening, middle, closing = (statistics.fmean(values) for values in zip(*(thirds for _, thirds in measured)))
        if closing >= opening:
            continue
        final_third_share = (closing - middle) / (closing - opening)
        converged, still_descending = AUTOTUNE_FINAL_THIRD_SHARE
        rejections = sum(rejected for rejected, _ in measured)
        rejecting = rejections > len(measured) / 2
        shortest, longest = (max(1, round(bound * configured)) for bound in AUTOTUNE_STAGE_LENGTH_BAND)
        current = overrides.get(axis, configured)
        step_change = AUTOTUNE_STAGE_STEP_CHANGE if rejecting or final_third_share > still_descending else -AUTOTUNE_STAGE_STEP_CHANGE if final_third_share < converged else 0.0
        adjusted = min(longest, max(shortest, round(current * (1 + step_change))))
        if adjusted != current:
            overrides[axis] = adjusted
            reason = autotune_stage_reason(design_stage, rejecting, rejections, final_third_share, step_change, len(measured))
            changes.append(f'{axis} {current} -> {adjusted} (configured {configured}, held within {shortest}-{longest}), {reason}')
    return ({'reviewed': len(trajectory_rows), 'accepted': accepted_designs, 'loss': block_loss, 'overrides': overrides, 'restores': restores, 'alternating': bool(drawn)}, report + [f'autotune: {change}' for change in changes])

def autotune_stage_reason(design_stage: str, rejecting: bool, rejections: int, final_third_share: float, step_change: float, trajectories: int) -> str:
    if rejecting:
        return f'{design_stage} rejected {rejections} of the {trajectories} trajectories that reached it'
    if final_third_share < 0:
        return f'{design_stage} stage deteriorated in its final third by {-final_third_share:.0%}, over {trajectories} trajectories'
    if step_change > 0:
        return f'{design_stage} stage was still descending at its end, with {final_third_share:.0%} of its improvement in the final third, over {trajectories} trajectories'
    return f'{design_stage} stage had flattened by its end, with only {final_third_share:.0%} of its improvement in the final third, over {trajectories} trajectories'

def autotuned_settings(campaign_progress: CampaignProgress, settings: dict, project_folder: str) -> dict:
    trajectory_rows = read_metric_rows(stage_table(project_folder, TRAJECTORY_STAGE))
    with campaign_progress.locked_progress() as state:
        record = state.setdefault('autotune', {'reviewed': 0, 'accepted': 0, 'loss': None, 'overrides': {}, 'restores': {}, 'alternating': False})
        accepted_designs, reviewing = (state['accepted'], len(trajectory_rows) - record['reviewed'] >= AUTOTUNE_REVIEW_TRAJECTORIES)
        if reviewing:
            state['autotune'] = {**record, 'reviewed': len(trajectory_rows)}
    if reviewing:
        record, report = autotune_review(settings, project_folder, record, trajectory_rows, accepted_designs)
        with campaign_progress.locked_progress() as state:
            state['autotune'] = record
        for line in report:
            print(line, flush=True)
    return {**settings, **record['overrides']}

def autotuned_value(value) -> str:
    return f'{value:g}' if isinstance(value, (int, float)) and (not isinstance(value, bool)) else str(value)

def autotuned_stamp(settings: dict, tuned_settings: dict) -> str:
    return ', '.join(f'{name} {autotuned_value(settings.get(name))} -> {autotuned_value(value)}' for name, value in tuned_settings.items() if settings.get(name) != value)
