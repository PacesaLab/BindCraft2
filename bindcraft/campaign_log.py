import math
import os
from importlib.metadata import PackageNotFoundError, version
from bindcraft.protein import recorded_number

def released_version() -> str:
    """The version every output is stamped with, read from the installed package so it cannot drift from pyproject."""
    try:
        return f'BindCraft 2 v{version("bindcraft")}'
    except PackageNotFoundError:
        return 'BindCraft 2 (unreleased)'

VERSION = released_version()
ALWAYS_SUPPRESSED_METRICS = ('Unbound_Binder_pLDDT', 'Protomer_Identity_Fraction', 'Oligomer_Symmetry_RMSD', 'Backbone_Clashes', 'All_Atom_Clashes', 'Epitope_Residues_Contacted', 'Binder_Length', 'Binder_Mass_kDa', 'Binder_pI', 'Binder_Net_Charge', 'Binder_Extinction', 'Binder_Cysteines', 'Binder_Free_Cysteines', 'Binder_Disulfides', 'Interface_BuriedArea', 'Interface_BuriedArea_Fraction', 'Surface_Hydrophobicity', 'Interface_Hydrophobicity', 'SS_pLDDT', 'Target_RMSD', 'Binder_Helix_Fraction', 'Binder_BetaSheet_Fraction', 'Binder_Loop_Fraction', 'Interface_Helix_Fraction', 'Interface_BetaSheet_Fraction', 'Interface_Loop_Fraction')
ACCEPTED_ONLY_METRICS = ('Induced_Fit_RMSD',)
SYMMETRY_RMSD_LIMIT = 1.0
COUNT_METRICS = ('Interface_Residues', 'Binder_Disulfides', 'Binder_Length', 'Binder_Cysteines', 'Binder_Free_Cysteines', 'Binder_Extinction', 'Backbone_Clashes', 'All_Atom_Clashes', 'Epitope_Residues_Contacted', 'Binder_Chain_Breaks', 'Receptor_Chains_Contacted', 'Target_Crop_Length')
SCAFFOLD_SUPPRESSED_METRICS = ('Backbone_Clashes', 'Off_Epitope_Contact_Fraction', 'Epitope_Residues_Contacted', 'Interface_Residues', 'Scaffold_Framework_RMSD', 'Scaffold_Sequence_Retained_Fraction', 'Target_pLDDT')

def trajectory_design_name(trajectory_directory: str | None) -> str:
    return os.path.basename(trajectory_directory) if trajectory_directory else 'design'

def trajectory_header(trajectory_number: int, design: str, accepted_design_count: int, requested_designs: int, worker_label: str='', autotuned: str='') -> str:
    tuned = f' | autotune:[{autotuned}]' if autotuned else ''
    return f'\n=== trajectory {trajectory_number} | {design} | accepted {accepted_design_count}/{requested_designs}{worker_label}{tuned} ==='

def desperation_warning(rungs: int, ladder_rungs: int, fruitless_trajectories: int, desperate_settings: str) -> str:
    return f'desperation: rung {rungs} of {ladder_rungs} after {fruitless_trajectories} trajectories with nothing accepted. Designing AND validating at {desperate_settings}. Designs accepted from here have a lower wet-lab success probability than designs accepted at the settings the campaign asked for.'

def stage_label(stage: str) -> str:
    return f'{stage} design stage'

STAGE_REPORTED_CONFIDENCES = ('pLDDT', 'i_pTM')

def stage_confidences(metrics: dict[str, float], target_state_names: tuple[str, ...]=()) -> str:
    reported = {single_state_metric_name(name, target_state_names): value for name, value in metrics.items() if name.split('.')[0] in STAGE_REPORTED_CONFIDENCES}
    return '  '.join(metric_field(name, value) for name, value in sorted(reported.items()))

def stage_outcome(stage: str, passed: bool, failed_filters: tuple[str, ...]=(), target_state_names: tuple[str, ...]=(), metrics: dict[str, float] | None=None) -> str:
    measured = f'  {stage_confidences(metrics, target_state_names)}' if metrics else ''
    if passed:
        return f'  passed {stage_label(stage)}{measured}'
    named = unique_in_order(single_state_metric_name(name, target_state_names) for name in failed_filters)
    reason = f' due to [{", ".join(named)}]' if named else ''
    if stage == 'final':
        return f'  trajectory rejected{measured}{reason}'
    return f'  rejected at {stage_label(stage)}{measured}{reason}'

def hallucination_successful() -> str:
    return '  binder hallucination successful'

def binder_optimization(candidate_count: int=0) -> str:
    return '  binder optimization' + (f', {candidate_count} MPNN redesigns' if candidate_count else '')

def single_state_metric_name(name: str, target_state_names: tuple[str, ...]) -> str:
    suffix = f'.{target_state_names[0]}' if len(target_state_names) == 1 else None
    return name[:-len(suffix)] if suffix and name.endswith(suffix) else name

def unique_in_order(names) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for name in names:
        seen.setdefault(name, None)
    return tuple(seen)

def symmetry_verdict(metrics: dict, limit: float=SYMMETRY_RMSD_LIMIT) -> str | None:
    rmsd = metrics.get('Oligomer_Symmetry_RMSD')
    return None if rmsd is None else ('yes' if float(rmsd) <= limit else 'no')

def reported_metrics(metrics: dict, binder_scaffold: bool=False, target_state_names: tuple[str, ...]=(), failed_filters: tuple[str, ...]=()) -> dict:
    suppressed = ALWAYS_SUPPRESSED_METRICS + (SCAFFOLD_SUPPRESSED_METRICS if binder_scaffold else ())
    reported = {name: value for name, value in metrics.items() if name in failed_filters or (name.split('.')[0] not in ACCEPTED_ONLY_METRICS and name.split('.')[0] not in suppressed)}
    if len(target_state_names) == 1:
        collapsed: dict[str, float] = {}
        for name, value in reported.items():
            collapsed.setdefault(single_state_metric_name(name, target_state_names), value)
        reported = collapsed
    return reported

def metric_field(name: str, value: float) -> str:
    whole = name.split('.')[0].endswith('_Count') or name.split('.')[0] in COUNT_METRICS
    value_text = f'{round(float(value))}' if whole else str(recorded_number(value))
    return f'{name}={value_text}'.ljust(len(name) + (4 if whole else 7))

def kept_on_acceptance(metrics: dict, failed_filters: tuple[str, ...]) -> dict:
    return {} if failed_filters else {name: value for name, value in metrics.items() if name.split('.')[0] in ACCEPTED_ONLY_METRICS}

def candidate_outcome(candidate_number: int, candidate_count: int, decode_source: str, failed_filters: list, metrics: dict, binder_scaffold: bool=False, target_state_names: tuple[str, ...]=()) -> str:
    named = unique_in_order(single_state_metric_name(name, target_state_names) for name in failed_filters)
    always = reported_metrics(metrics, binder_scaffold, target_state_names)
    reported = reported_metrics(metrics, binder_scaffold, target_state_names, tuple(failed_filters))
    symmetric = symmetry_verdict(metrics)
    fields = [metric_field(name, value) for name, value in sorted(always.items())]
    if symmetric is not None:
        fields.append(f'symmetric={symmetric:<3}')
    fields += [metric_field(name, value) for name, value in sorted(kept_on_acceptance(metrics, named).items())]
    fields += [metric_field(name, value) for name, value in sorted(reported.items()) if name not in always]
    numbering = f'{candidate_number:>{len(str(candidate_count))}}/{candidate_count}'
    return f"  {numbering}  {'rejected' if named else 'ACCEPTED'}  " + '  '.join(fields) + (f"  failed [{', '.join(named)}]" if named else '')

def exhausted_redesign_window(drawn: int, requested: int) -> str:
    return f'  {drawn} of {requested} sequences: the redesign window has no more distinct sequences to give'

def redesigns_kept(kept_candidates: tuple[tuple[int, float], ...], accepted_count: int, candidate_count: int, metric: str) -> str:
    if not kept_candidates:
        return f'  0 of {candidate_count} redesigns passed'
    ranked = ', '.join(f'candidate {number} ({recorded_number(value)})' for number, value in kept_candidates)
    return f'  {accepted_count} of {candidate_count} redesigns passed, keeping the best {len(kept_candidates)} by {metric}: {ranked}'

def campaign_metadata_written(version: str, settings_path: str, metadata_path: str) -> str:
    return f'campaign metadata: {version} | settings {settings_path} | written {metadata_path}'

def scaffold_name(settings: dict) -> str:
    return os.path.splitext(os.path.basename(str(settings.get('binder_scaffold') or '')))[0]

def campaign_modality_shorthand(settings: dict, target_objectives: tuple[str, ...]=()) -> str:
    from bindcraft.settings import feature_shorthand, requested_campaign_features
    named = next((word for word in (feature_shorthand(feature, settings) for feature in requested_campaign_features(settings)) if word), '')
    if named:
        return named
    if 'detarget' in target_objectives:
        return 'detarget'
    return 'multitarget' if len(target_objectives) > 1 else 'denovo'

def campaign_label(settings: dict) -> str:
    return str(settings.get('campaign_name') or settings.get('binder_name') or '').strip()

def trajectory_design_label(settings: dict, target_objectives: tuple[str, ...]=()) -> str:
    names = dict.fromkeys(name for name in (campaign_label(settings), str(settings.get('binder_name') or '').strip()) if name)
    return '_'.join((*names, campaign_modality_shorthand(settings, target_objectives))) if names else 'design'

def trajectory_already_designed(design: str) -> str:
    return f'  skipped: {design} is already designed in this folder'

def campaign_header(campaign: str, settings: dict, chain_residues: dict[str, dict[str, int]], losses: dict, targets: tuple=()) -> tuple[str, ...]:
    lines = [VERSION, f'campaign {campaign}']
    scaffold, copies = scaffold_name(settings), int(settings.get('copies', 1) or 1)
    lengths = list(sampled_binder_lengths(settings))
    measured = unique_in_order(str(length) for chains in chain_residues.values() for chain, length in chains.items() if chain.startswith('binder'))
    if scaffold:
        described = f'{scaffold} scaffold, framework held and the named spans redesigned'
    elif lengths:
        described = 'length ' + length_choices(lengths) + (', drawn per trajectory' if len(lengths) > 1 else '')
    else:
        described = 'length ' + (length_choices([int(value) for value in measured]) if measured else 'unset')
    lines.append(f'binder {described}' + (f" | multi-chain binder of {copies} chains, {settings.get('oligomer_tie', 'symmetric')} tie" if copies > 1 else ''))
    crop_count = int(settings.get('idr_crop_count') or 0)
    window = list(settings.get('crop_fasta_sequence') or [])
    for target in targets:
        steering = ''.join(f' | {name} {getattr(target, name)}' for name in ('hotspots', 'coldspots') if getattr(target, name, None))
        if crop_count:
            span = '-'.join(str(value) for value in window) if window else 'unset'
            lines.append(f"target {target.name} | {crop_count} windows of {span} residues, redrawn every trajectory{steering}")
        else:
            residues = [length for state, chains in chain_residues.items() for chain, length in chains.items() if not chain.startswith('binder') and state.startswith(target.name)]
            measured = ', '.join(f'{length} residues' for length in residues) or 'unmeasured'
            lines.append(f"target {target.name} ({target.objective}) | {measured}{steering}")
    lines.append('losses ' + ' '.join(f'{name}={weight}' for name, weight in sorted(collapsed_loss_weights(losses).items())))
    features = campaign_feature_notes(settings)
    if features:
        lines.append('features ' + ', '.join(features))
    filters = acceptance_filters(settings)
    if filters:
        lines.append('filters ' + ', '.join(filters))
    project_folder = settings.get('project_folder')
    if project_folder:
        from bindcraft.campaign_output import RANK_STAGE, STAGE_TABLE_NAMES, SUMMARY_FILENAME
        lines.append(f'results {project_folder} | {RANK_STAGE}/{STAGE_TABLE_NAMES[RANK_STAGE]} rewritten as each design is accepted | {SUMMARY_FILENAME} written when the campaign ends')
    return tuple(lines)

def collapsed_loss_weights(losses: dict) -> dict:
    grouped: dict[str, dict[str, float]] = {}
    for name, weight in losses.items():
        base, _, state = name.partition('.')
        grouped.setdefault(base, {})[state] = weight
    collapsed = {}
    for base, weights in grouped.items():
        if len(weights) > 1 and len(set(weights.values())) == 1:
            collapsed[f'{base} (every target)'] = next(iter(weights.values()))
        else:
            collapsed.update({f'{base}.{state}' if state else base: weight for state, weight in weights.items()})
    return collapsed

def campaign_budget_exhausted(max_trajectories: int, trajectory_count: int, accepted_design_count: int, requested_designs: int) -> str:
    if not accepted_design_count:
        return f'campaign stopped: {trajectory_count} trajectories ran and none were accepted, so the settings rather than the budget are what to change'
    projected = -(-requested_designs * trajectory_count // accepted_design_count)
    return f'campaign stopped at max_trajectories={max_trajectories}: {accepted_design_count}/{requested_designs} accepted in {trajectory_count} trajectories, so {requested_designs} needs roughly {projected} at this rate'

def acceptance_filters(settings: dict) -> tuple[str, ...]:
    filters = []
    for name, entry in sorted((settings.get('filters') or {}).items()):
        threshold, higher = entry.get('threshold'), entry.get('higher', False)
        if threshold is None or not math.isfinite(float(threshold)):
            continue
        if (higher and float(threshold) <= 0) or (not higher and name.endswith('_Fraction') and float(threshold) >= 1):
            continue
        filters.append(f"{name} {'>=' if higher else '<='} {threshold:g}")
    return tuple(filters)

def design_worker_index() -> int | None:
    worker = os.environ.get('BINDCRAFT_WORKER_ID')
    return int(worker) if worker is not None and worker.isdigit() else None

def speaks_for_the_campaign() -> bool:
    return design_worker_index() in (None, 0)

def campaign_closed(accepted_design_count: int, trajectory_count: int, metric: str) -> str:
    plural = 'y' if trajectory_count == 1 else 'ies'
    return f'\ncampaign done: {accepted_design_count} accepted design(s) after {trajectory_count} trajector{plural}, ranked by {metric}'

def campaign_feature_notes(settings: dict) -> tuple[str, ...]:
    from bindcraft.settings import requested_campaign_features
    return tuple(feature.note(settings) for feature in requested_campaign_features(settings) if feature.note is not None)

def sampled_binder_lengths(settings: dict) -> tuple[int, ...]:
    from bindcraft.settings import campaign_binder_lengths
    return campaign_binder_lengths(settings.get('binder_lengths')) or ()

def length_choices(lengths) -> str:
    ordered = sorted(dict.fromkeys(int(length) for length in lengths))
    spacings = {later - earlier for earlier, later in zip(ordered, ordered[1:])}
    if len(ordered) < 3 or len(spacings) > 1:
        return ' or '.join(str(length) for length in ordered)
    spacing = spacings.pop()
    return f'{ordered[0]} to {ordered[-1]}' + ('' if spacing == 1 else f' every {spacing}')
