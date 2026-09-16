import math
import os
import jax
from typing import Callable, NamedTuple
from jax import Array
from bindcraft.campaign_log import hallucination_successful, stage_outcome
from bindcraft.campaign_output import trajectory_output_path
from bindcraft.trajectory_output import TrajectoryRecorder
from bindcraft.epitope_targeting import force_epitope_targeting, restore_wild_type_targets
from bindcraft.prediction import ProteinPredictor, DifferentiableProteinPredictor, collect_shared_chains, update_shared_sequences
from bindcraft.filters import binder_beta_sheet_fraction_metric, design_stage_filters, evaluate_design_filters, induced_fit_interface_rmsd_metric, induced_fit_tm_metric, passes_design_filters
from bindcraft.protein_preparation import receptor_chain_layouts, initialize_design_trajectory
from bindcraft.loss import DesignLoss, binder_alone_design_losses, freeze_induced_fit_interface, induced_fit_hinge_names, induced_fit_reference_losses, interface_mask_residues, weighted_design_loss
from bindcraft.sequence_optimization import SequenceAnnealingOptimizer, OneHotSequenceOptimizer, SequenceOptimizer, SequenceMutationSampler, SemigreedySequenceSampler, LogitSequenceOptimizer
from bindcraft.protein import recorded_number, BINDER_ALONE, Protein, StructurePrediction, StructurePredictions, ProteinStates, is_target_chain, target_chain_name, write_structure
from bindcraft.target_schedule import DesignSchedule, FixedTargetSchedule, build_design_schedule, build_target_schedule, losses_for_active_states, rotating_round_predictions, stage_filter_predictions
from bindcraft.settings import BinderDesignSettings, DEFAULT_SETTINGS, design_stage_rounds, merged_gradient_sequence_updates

STAGE_NAMES = ('screen', 'refine', 'anneal', 'harden')

class TrajectoryState(NamedTuple):
    protein_states: ProteinStates
    predictions: StructurePredictions
    design_target_states: ProteinStates
    losses: dict[str, DesignLoss]
    binder_alone_reference: StructurePrediction | None = None
    failed: tuple[str, ...] = ()

class StageOperation(NamedTuple):
    name: str
    apply: Callable[[TrajectoryState], TrajectoryState]

class DesignStage(NamedTuple):
    name: str
    sequence_optimizer: SequenceOptimizer
    dropout: bool
    prepare: tuple[StageOperation, ...] = ()
    review: tuple[StageOperation, ...] = ()
    advance: tuple[StageOperation, ...] = ()

def primary_target_state(protein_states: ProteinStates) -> str:
    names = [name for name in protein_states if name != BINDER_ALONE]
    return names[0] if names else next(iter(protein_states))

def binder_alone_state(protein_states: ProteinStates, bound_state: str, target_chain: str) -> ProteinStates:
    return {BINDER_ALONE: {name: protein for name, protein in protein_states[bound_state].items() if not is_target_chain(name, target_chain)}}

def binder_alone_switch_metrics(protein_states: ProteinStates, predictions: StructurePredictions, losses: dict[str, DesignLoss]) -> dict[str, float | None]:
    interface_name = next((name for name in induced_fit_hinge_names(losses) if name.split('.')[0] == 'induced_fit_interface'), None)
    interface_parameters = losses[interface_name].function.keywords if interface_name else {}
    interface_rmsd = induced_fit_interface_rmsd_metric(protein_states, predictions, cutoff=float(interface_parameters.get('cutoff', 8.0)), interface_residues=interface_mask_residues(losses[interface_name].interface_mask)) if interface_name else None
    return {'binder_alone_plddt': float(predictions[BINDER_ALONE].metrics['plddt'].mean()), 'interface_rmsd': interface_rmsd, 'global_tm': induced_fit_tm_metric(protein_states, predictions)}

def binder_alone_switch_reached(metrics: dict[str, float | None], losses: dict[str, DesignLoss], minimum_plddt: float) -> bool:
    if metrics['binder_alone_plddt'] < minimum_plddt:
        return False
    for name in induced_fit_hinge_names(losses):
        parameters = losses[name].function.keywords
        if name.split('.')[0] == 'induced_fit_interface' and (metrics['interface_rmsd'] or 0.0) < float(parameters.get('interface_rmsd_target', 3.0)):
            return False
        if name.split('.')[0] == 'induced_fit_global' and (metrics['global_tm'] is None or metrics['global_tm'] > float(parameters.get('tm_target', 0.6))):
            return False
    return True

def binder_alone_reference_refresh(design_model: ProteinPredictor, design_settings: BinderDesignSettings) -> Callable[..., StructurePredictions | None]:
    interval = max(1, int(design_settings.settings.get('induced_fit_monomer_chunk', 5)))
    def refresh(protein_states: ProteinStates, sequence_updates: int, sequence_optimizer: SequenceOptimizer | None=None) -> StructurePredictions | None:
        if sequence_updates % interval:
            return None
        sequence_parameters = dict(zip(('softmax_weight', 'one_hot_weight', 'temperature', 'logit_scale'), sequence_optimizer.sequence_parameters())) if sequence_optimizer is not None else {}
        return design_model.predict(binder_alone_state(protein_states, primary_target_state(protein_states), design_settings.target_chain_prefix), **sequence_parameters)
    return refresh

def run_binder_alone_optimization_block(protein_states: ProteinStates, bound_predictions: StructurePredictions, design_model: DifferentiableProteinPredictor, losses: dict[str, DesignLoss], design_settings: BinderDesignSettings, design_stage: str, multi_chain_binders: tuple[tuple[str, ...], ...]=(), recorder: TrajectoryRecorder | None=None) -> tuple[ProteinStates, StructurePredictions]:
    settings = design_settings.settings
    maximum_steps = int(settings.get('induced_fit_monomer_steps', 30))
    if maximum_steps < 1 or not induced_fit_hinge_names(losses):
        return protein_states, bound_predictions
    target_chain = settings.get('target_chain', 'target')
    bound_state = primary_target_state(protein_states)
    monomer_states = binder_alone_state(protein_states, bound_state, target_chain)
    monomer_losses = binder_alone_design_losses(losses, bound_state)
    reference_predictions = {bound_state: bound_predictions[bound_state]}
    sequence_optimizer = SequenceAnnealingOptimizer(iterations=maximum_steps, multi_chain_binders=multi_chain_binders)
    chunk_size, completed_steps, metrics = max(1, int(settings.get('induced_fit_monomer_chunk', 5))), 0, {}
    combined_predictions = None
    while completed_steps < maximum_steps:
        block_steps = min(chunk_size, maximum_steps - completed_steps)
        monomer_states, monomer_predictions = run_gradient_design_stage(monomer_states, design_model, monomer_losses, sequence_optimizer, FixedTargetSchedule(block_steps), record_sequence_update=recorder, reference_predictions=reference_predictions, first_sequence_update=completed_steps)
        if monomer_predictions is None:
            break
        completed_steps += block_steps
        combined_states = {**transfer_binder_sequences(protein_states, monomer_states), **monomer_states}
        combined_predictions = {**bound_predictions, **monomer_predictions}
        metrics = binder_alone_switch_metrics(combined_states, combined_predictions, losses)
        if settings.get('induced_fit_monomer_adaptive', True) and binder_alone_switch_reached(metrics, losses, float(settings.get('induced_fit_monomer_plddt', 0.7))):
            break
    measured = ' '.join(f'{name}={recorded_number(value)}' for name, value in metrics.items() if value is not None)
    print(f'induced fit: binder-alone block after {design_stage}, {completed_steps}/{maximum_steps} steps; {measured}', flush=True)
    if combined_predictions is None:
        return protein_states, bound_predictions
    return transfer_binder_sequences(protein_states, monomer_states), combined_predictions

def transfer_binder_sequences(target_states: ProteinStates, designed_states: ProteinStates) -> ProteinStates:
    return update_shared_sequences(target_states, collect_shared_chains(designed_states)[1])

def pooled_stage_predictions(design_model: ProteinPredictor, protein_states: ProteinStates, filter_models: int, sequence_parameters: dict[str, Array] | None=None) -> StructurePredictions:
    models, sequence_parameters = (tuple(getattr(design_model, 'models', ()))[:filter_models], sequence_parameters or {})
    if len(models) < 2:
        return design_model.predict(protein_states, **sequence_parameters)
    pooled = [design_model.predict(protein_states, model=model, **sequence_parameters) for model in models]
    return {name: StructurePrediction(protein_complex=pooled[0][name].protein_complex, metrics={metric: sum(prediction[name].metrics[metric] for prediction in pooled) / len(pooled) for metric in pooled[0][name].metrics}) for name in pooled[0]}

def rotating_stage_filter_round(design_settings: BinderDesignSettings, target_schedule: DesignSchedule, protein_states: ProteinStates, design_stage: str) -> Callable[[ProteinStates, StructurePredictions], bool] | None:
    if not design_settings.settings.get('multitarget_best_round') or len(design_settings.prepared_states) < 2:
        return None
    stage_filters = design_stage_filters(design_settings, protein_states, design_stage)
    def round_passes_stage_filter(round_states: ProteinStates, predictions: StructurePredictions) -> bool:
        return passes_design_filters(stage_filters, round_states, rotating_round_predictions(target_schedule, predictions)) is True
    return round_passes_stage_filter if stage_filters else None

def run_gradient_design_stage(protein_states: ProteinStates, design_model: DifferentiableProteinPredictor, losses: dict[str, DesignLoss], sequence_optimizer: SequenceOptimizer, design_schedule: DesignSchedule, record_sequence_update: Callable[[int, StructurePredictions], None] | None=None, select_best_round: bool=True, select_filtered_round: Callable[[ProteinStates, StructurePredictions], bool] | None=None, reference_predictions: StructurePredictions | None=None, first_sequence_update: int=0, refresh_reference_predictions: Callable[[ProteinStates, int, SequenceOptimizer], StructurePredictions | None] | None=None) -> tuple[ProteinStates, StructurePredictions | None]:
    accumulated_gradients: dict[str, list[Array]] = {}
    predictions = None
    sequence_updates = first_sequence_update
    best_design_loss, best_protein_states, best_predictions, measured_predictions = (None, protein_states, None, None)
    while not design_schedule.is_complete(predictions):
        active_protein_states = design_schedule.select_protein_states(protein_states, predictions)
        softmax_weight, one_hot_weight, temperature, logit_scale = sequence_optimizer.sequence_parameters()
        active_losses = losses_for_active_states(losses, active_protein_states)
        predictions, chain_gradients, graph_design_loss = design_model.sequence_gradients(active_protein_states, active_losses, softmax_weight=softmax_weight, one_hot_weight=one_hot_weight, temperature=temperature, logit_scale=logit_scale, reference_predictions=reference_predictions)
        design_loss = float(graph_design_loss)
        if not math.isfinite(design_loss):
            design_loss = float(weighted_design_loss(active_losses, active_protein_states, {**predictions, **(reference_predictions or {})}))
        if not math.isfinite(design_loss):
            break
        measured_predictions = predictions
        if select_best_round and (best_design_loss is None or design_loss < best_design_loss):
            best_design_loss, best_protein_states, best_predictions = (design_loss, protein_states, predictions)
        if select_filtered_round is not None and select_filtered_round(protein_states, predictions):
            best_protein_states, best_predictions = (protein_states, predictions)
        for name, chain_gradient in chain_gradients.items():
            accumulated_gradients.setdefault(name, []).append(chain_gradient)
        if design_schedule.should_update_sequence(active_protein_states, predictions, accumulated_gradients):
            protein_states = transfer_binder_sequences(protein_states, sequence_optimizer.update_sequence(active_protein_states, accumulated_gradients))
            accumulated_gradients = {}
            design_schedule.record_sequence_update()
            sequence_updates += 1
            if record_sequence_update is not None:
                record_sequence_update(sequence_updates, predictions)
            if refresh_reference_predictions is not None:
                reference_predictions = refresh_reference_predictions(protein_states, sequence_updates, sequence_optimizer) or reference_predictions
    return (best_protein_states, best_predictions) if best_predictions is not None else (protein_states, measured_predictions)

def run_sequence_mutation_stage(protein_states: ProteinStates, design_model: ProteinPredictor, losses: dict[str, DesignLoss], mutation_sampler: SequenceMutationSampler, design_schedule: DesignSchedule, record_sequence_update: Callable[[int, StructurePredictions], None] | None=None, reference_predictions: StructurePredictions | None=None, refresh_reference_predictions: Callable[..., StructurePredictions | None] | None=None) -> tuple[ProteinStates, StructurePredictions]:
    predictions, initial_sequence_scored = None, False
    sequence_updates = 0
    while not design_schedule.is_complete(predictions):
        active_protein_states = design_schedule.select_protein_states(protein_states, predictions)
        candidate_states = mutation_sampler.propose_sequence_mutation(active_protein_states) if initial_sequence_scored else active_protein_states
        initial_sequence_scored = True
        predictions = design_model.predict(candidate_states)
        design_loss = weighted_design_loss(losses, candidate_states, {**predictions, **(reference_predictions or {})})
        protein_states = transfer_binder_sequences(protein_states, mutation_sampler.select_best_sequence(design_loss, predictions))
        design_schedule.record_sequence_update()
        sequence_updates += 1
        if record_sequence_update is not None:
            record_sequence_update(sequence_updates, predictions)
        if refresh_reference_predictions is not None:
            reference_predictions = refresh_reference_predictions(protein_states, sequence_updates) or reference_predictions
    return protein_states, predictions

def restore_wild_type_target_operation(design_settings: BinderDesignSettings, wild_type_states: ProteinStates) -> StageOperation:
    def restore_wild_type_target(trajectory: TrajectoryState) -> TrajectoryState:
        return trajectory._replace(protein_states=restore_wild_type_targets(trajectory.protein_states, wild_type_states, design_settings.settings), design_target_states=wild_type_states)
    return StageOperation('restore the wild-type target', restore_wild_type_target)

def pooled_prediction_operation(design_settings: BinderDesignSettings, design_model: ProteinPredictor, sequence_optimizer: SequenceOptimizer) -> StageOperation:
    def pool_multitarget_predictions(trajectory: TrajectoryState) -> TrajectoryState:
        protein_states = transfer_binder_sequences(trajectory.design_target_states, trajectory.protein_states)
        sequence_parameters = dict(zip(('softmax_weight', 'one_hot_weight', 'temperature', 'logit_scale'), sequence_optimizer.sequence_parameters()))
        return trajectory._replace(protein_states=protein_states, predictions=pooled_stage_predictions(design_model, protein_states, int(design_settings.settings.get('multitarget_filter_models', 1)), sequence_parameters))
    return StageOperation('pool the multitarget predictions', pool_multitarget_predictions)

def beta_sheet_budget_operation(design_settings: BinderDesignSettings, design_model: DifferentiableProteinPredictor, sequence_optimizers: dict[str, SequenceOptimizer]) -> StageOperation:
    settings = design_settings.settings
    def widen_budget_for_beta_sheet(trajectory: TrajectoryState) -> TrajectoryState:
        beta_sheet_fraction = binder_beta_sheet_fraction_metric(trajectory.protein_states, trajectory.predictions, binder=design_settings.designed_binder_chain)
        if beta_sheet_fraction is not None and beta_sheet_fraction > float(settings.get('betasheet_reopt_trigger', DEFAULT_SETTINGS['betasheet_reopt_trigger'])):
            sequence_optimizers['refine'].iterations += int(settings.get('betasheet_reopt_extra_refine_steps', DEFAULT_SETTINGS['betasheet_reopt_extra_refine_steps']))
            sequence_optimizers['anneal'].iterations += int(settings.get('betasheet_reopt_extra_anneal_steps', DEFAULT_SETTINGS['betasheet_reopt_extra_anneal_steps']))
            design_model.num_recycle = int(settings.get('betasheet_reopt_recycles', design_model.num_recycle))
        return trajectory
    return StageOperation('widen the budget for a beta-sheet binder', widen_budget_for_beta_sheet)

def freeze_induced_fit_operation() -> StageOperation:
    def freeze_induced_fit_hinge(trajectory: TrajectoryState) -> TrajectoryState:
        losses = freeze_induced_fit_interface(trajectory.losses, trajectory.protein_states, trajectory.predictions)
        unfrozen = any((name.split('.')[0] == 'induced_fit_interface' and losses[name].interface_mask is None for name in induced_fit_hinge_names(losses)))
        return trajectory._replace(losses=losses, failed=('induced_fit_interface has no frozen interface',) if unfrozen else ())
    return StageOperation('freeze the induced-fit interface', freeze_induced_fit_hinge)

def binder_alone_operation(design_settings: BinderDesignSettings, design_model: DifferentiableProteinPredictor, multi_chain_binders: tuple[tuple[str, ...], ...], recorder: TrajectoryRecorder | None, stage_name: str) -> StageOperation:
    def optimize_binder_alone(trajectory: TrajectoryState) -> TrajectoryState:
        if recorder is not None:
            recorder.design_stage = f'{stage_name}_binder_alone'
        protein_states, predictions = run_binder_alone_optimization_block(trajectory.protein_states, trajectory.predictions, design_model, trajectory.losses, design_settings, stage_name, multi_chain_binders, recorder)
        return trajectory._replace(protein_states=protein_states, predictions=predictions, binder_alone_reference=predictions.get(BINDER_ALONE, trajectory.binder_alone_reference))
    return StageOperation('fold the binder alone', optimize_binder_alone)

def build_stage_plan(design_settings: BinderDesignSettings, losses: dict[str, DesignLoss], multi_chain_binders: tuple[tuple[str, ...], ...], design_model: DifferentiableProteinPredictor, wild_type_states: ProteinStates, recorder: TrajectoryRecorder | None=None) -> tuple[DesignStage, ...]:
    settings = design_settings.settings
    stage_rounds = merged_gradient_sequence_updates(design_settings)
    fold_switching = bool(induced_fit_hinge_names(losses))
    multitarget = len(design_settings.prepared_states) > 1
    sequence_optimizers = {'screen': LogitSequenceOptimizer(iterations=stage_rounds['screen'], start_softmax_weight=0.0, end_softmax_weight=0.9, multi_chain_binders=multi_chain_binders),
                           'refine': LogitSequenceOptimizer(iterations=stage_rounds['refine'], start_softmax_weight=0.9, end_softmax_weight=1.0, multi_chain_binders=multi_chain_binders),
                           'anneal': SequenceAnnealingOptimizer(iterations=stage_rounds['anneal'], multi_chain_binders=multi_chain_binders),
                           'harden': OneHotSequenceOptimizer(iterations=stage_rounds['harden'], multi_chain_binders=multi_chain_binders)}
    stage_plan = []
    for name in STAGE_NAMES:
        prepare, review, advance = [], [], []
        if name == 'harden' and settings.get('forced_targeting'):
            prepare.append(restore_wild_type_target_operation(design_settings, wild_type_states))
        if multitarget:
            review.append(pooled_prediction_operation(design_settings, design_model, sequence_optimizers[name]))
        if name == 'screen':
            review.append(beta_sheet_budget_operation(design_settings, design_model, sequence_optimizers))
        if name == 'screen' and fold_switching:
            advance.append(freeze_induced_fit_operation())
        if fold_switching and name != 'harden':
            advance.append(binder_alone_operation(design_settings, design_model, multi_chain_binders, recorder, name))
        stage_plan.append(DesignStage(name, sequence_optimizers[name], name != 'harden', tuple(prepare), tuple(review), tuple(advance)))
    return tuple(stage_plan)

def required_final_states(design_settings: BinderDesignSettings, losses: dict[str, DesignLoss]) -> tuple[str, ...]:
    switches_conformation = any(BINDER_ALONE in group for group in design_settings.binder_shapes)
    return (BINDER_ALONE,) if induced_fit_hinge_names(losses) or switches_conformation else ()

def run_stage_operations(operations: tuple[StageOperation, ...], trajectory: TrajectoryState) -> TrajectoryState:
    for operation in operations:
        trajectory = operation.apply(trajectory)
        if trajectory.failed:
            break
    return trajectory

def induced_fit_reference_arguments(losses: dict[str, DesignLoss], binder_alone_reference: StructurePrediction | None, design_model: ProteinPredictor, design_settings: BinderDesignSettings) -> dict:
    if binder_alone_reference is None:
        return {'losses': losses}
    return {'losses': induced_fit_reference_losses(losses, BINDER_ALONE), 'reference_predictions': {BINDER_ALONE: binder_alone_reference}, 'refresh_reference_predictions': binder_alone_reference_refresh(design_model, design_settings)}

def judge_stage(trajectory: TrajectoryState, design_settings: BinderDesignSettings, target_schedule: DesignSchedule, filter_stage: str) -> tuple[TrajectoryState, dict[str, float]]:
    stage_filters = design_stage_filters(design_settings, trajectory.protein_states, filter_stage)
    filter_predictions = stage_filter_predictions(target_schedule, trajectory.predictions, design_settings.settings.get('multitarget_cumulative_filter', len(design_settings.prepared_states) > 1))
    filter_result, measured = evaluate_design_filters(stage_filters, trajectory.protein_states, filter_predictions) if stage_filters else (True, {})
    return (trajectory if filter_result is True else trajectory._replace(predictions=filter_predictions, failed=tuple(filter_result))), measured

def run_mutation_polish(design_settings: BinderDesignSettings, design_model: DifferentiableProteinPredictor, protein_states: ProteinStates, wild_type_states: ProteinStates, losses: dict[str, DesignLoss], binder_alone_reference: StructurePrediction | None, multi_chain_binders: tuple[tuple[str, ...], ...], mutation_random_key: Array, conformation_random_key: Array, target_names: tuple[str, ...], recorder: TrajectoryRecorder | None) -> tuple[ProteinStates, StructurePredictions, str | None]:
    settings = design_settings.settings
    mutate_steps = design_stage_rounds(settings)['mutate']
    if recorder is not None:
        recorder.design_stage = 'mutate'
    design_model.dropout = False
    #pLDDT weighting, for multitargeting
    mutation_sampler = SemigreedySequenceSampler(key=mutation_random_key, multi_chain_binders=multi_chain_binders, mutation_weighting='plddt' if len(design_settings.prepared_states) > 1 else 'interface_iptm')
    design_schedule = build_design_schedule(design_settings, wild_type_states, losses, mutate_steps, conformation_random_key, False)
    if len(design_settings.prepared_states) > 1:
        protein_states = transfer_binder_sequences(wild_type_states, protein_states)
        design_schedule = FixedTargetSchedule(mutate_steps)
    protein_states, predictions = run_sequence_mutation_stage(protein_states, design_model, mutation_sampler=mutation_sampler, design_schedule=design_schedule, record_sequence_update=recorder, **induced_fit_reference_arguments(losses, binder_alone_reference, design_model, design_settings))
    protein_states = transfer_binder_sequences(wild_type_states, protein_states)
    predictions = design_model.predict(protein_states)
    stage_filters = design_stage_filters(design_settings, protein_states, 'mutate')
    filter_result, measured = evaluate_design_filters(stage_filters, protein_states, predictions) if stage_filters else (True, {})
    print(stage_outcome('mutate', filter_result is True, () if filter_result is True else tuple(filter_result), target_names, measured), flush=True)
    return protein_states, predictions, None if filter_result is True else 'mutate'

def run_trajectory(design_settings: BinderDesignSettings, design_model: DifferentiableProteinPredictor, key: Array, trajectory_directory: str | None=None, targets: dict[str, Protein] | None=None) -> tuple[ProteinStates, StructurePredictions, str | None]:
    settings = design_settings.settings
    binder_initialization_key, mutation_random_key = jax.random.split(key)
    conformation_random_key = jax.random.split(binder_initialization_key)[1]
    protein_states, multi_chain_binders, losses = initialize_design_trajectory(design_settings, binder_initialization_key, targets)
    keep_frames = bool(settings.get('save_design_frames') or settings.get('save_design_animations') or settings.get('save_loss_plots'))
    recorder = TrajectoryRecorder(trajectory_directory, keep_frames, receptor_chain_layouts(design_settings), keep_sequences=bool(settings.get('save_design_sequences'))) if trajectory_directory is not None else None
    target_names = tuple(state.name for state in design_settings.prepared_states)
    wild_type_states = protein_states
    #focused epitope case
    if settings.get('forced_targeting'):
        protein_states = force_epitope_targeting(protein_states, settings)
        if recorder is not None:
            for name, protein_complex in protein_states.items():
                write_structure({chain: protein for chain, protein in protein_complex.items() if chain == target_chain_name(design_settings.target_chain_prefix, name)}, trajectory_output_path(trajectory_directory, f'forced_targeting_{name}.pdb'), receptor_chains=receptor_chain_layouts(design_settings))
    stage_plan = build_stage_plan(design_settings, losses, multi_chain_binders, design_model, wild_type_states, recorder)
    target_schedule = build_target_schedule(design_settings, protein_states, stage_plan[0].sequence_optimizer.iterations, stage_plan[0].name)
    trajectory, failed_stage = TrajectoryState(protein_states, {}, protein_states, losses), None
    for stage in stage_plan:
        if recorder is not None:
            recorder.design_stage = stage.name
            recorder.sequence_parameters = stage.sequence_optimizer.sequence_parameters
        trajectory = run_stage_operations(stage.prepare, trajectory)
        design_model.dropout = stage.dropout and settings.get('design_dropout', DEFAULT_SETTINGS['design_dropout'])
        design_schedule = build_design_schedule(design_settings, trajectory.design_target_states, trajectory.losses, stage.sequence_optimizer.iterations, conformation_random_key, False, target_schedule, stage.name)
        protein_states, predictions = run_gradient_design_stage(trajectory.protein_states, design_model, sequence_optimizer=stage.sequence_optimizer, design_schedule=design_schedule, record_sequence_update=recorder, select_best_round=len(design_settings.prepared_states) < 2, select_filtered_round=rotating_stage_filter_round(design_settings, target_schedule, trajectory.protein_states, stage.name), **induced_fit_reference_arguments(trajectory.losses, trajectory.binder_alone_reference, design_model, design_settings))
        if predictions is None:
            trajectory = trajectory._replace(predictions={}, failed=('no finite round',))
        else:
            trajectory = run_stage_operations(stage.review, trajectory._replace(protein_states=protein_states, predictions=predictions))
        stage_metrics: dict[str, float] = {}
        if not trajectory.failed:
            trajectory, stage_metrics = judge_stage(trajectory, design_settings, target_schedule, stage.name)
        if not trajectory.failed:
            trajectory = run_stage_operations(stage.advance, trajectory)
        print(stage_outcome(stage.name, not trajectory.failed, trajectory.failed, target_names, stage_metrics), flush=True)
        if trajectory.failed:
            failed_stage = stage.name
            break
    protein_states, predictions, losses = trajectory.protein_states, trajectory.predictions, trajectory.losses
    if failed_stage is None and design_stage_rounds(settings)['mutate']:
        protein_states, predictions, failed_stage = run_mutation_polish(design_settings, design_model, protein_states, wild_type_states, losses, trajectory.binder_alone_reference, multi_chain_binders, mutation_random_key, conformation_random_key, target_names, recorder)
    if failed_stage is None:
        stage_filters = design_stage_filters(design_settings, protein_states, 'final')
        filter_result, measured = evaluate_design_filters(stage_filters, protein_states, predictions) if stage_filters else (True, {})
        failed_stage = None if filter_result is True else 'final'
        print(hallucination_successful() if filter_result is True else stage_outcome('final', False, tuple(filter_result), target_names, measured), flush=True)
    if BINDER_ALONE in required_final_states(design_settings, losses) and BINDER_ALONE not in predictions:
        predictions = {**predictions, **design_model.predict(binder_alone_state(protein_states, primary_target_state(protein_states), design_settings.target_chain_prefix))}
    if recorder is not None:
        recorder.write_trajectory_outputs(settings)
    return protein_states, predictions, failed_stage
