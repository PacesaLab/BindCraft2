from typing import Callable, NamedTuple, Protocol, runtime_checkable
import jax
from jax import Array
from bindcraft.prediction import collect_shared_chains
from bindcraft.filters import interface_residues_metric
from bindcraft.loss import DesignLoss
from bindcraft.protein import StructurePredictions, StructurePrediction, Protein, ProteinStates, BINDER_ALONE, target_chain_name
from bindcraft.settings import DEFAULT_DETARGET_CHECK_INTERVAL, DEFAULT_DETARGET_INTERFACE_RESIDUES, MERGED_GRADIENT_STAGE, BinderDesignSettings, design_model_count, merged_gradient_targets

def build_target_schedule(design_settings: BinderDesignSettings, target_states: ProteinStates, iterations: int, design_stage: str='screen') -> 'IterationLimitedDesignSchedule':
    settings = design_settings.settings
    target_chain = design_settings.target_chain_prefix
    targets = {state.name: target_states[state.name][state.target_chain] for state in design_settings.prepared_states}
    if not targets:
        return FixedTargetSchedule(iterations)
    target_objectives = {state.name: state.objective for state in design_settings.prepared_states}
    epitope_rotation_interval = design_model_count(settings) if int(settings.get('idr_crop_count', 1)) > 1 else None
    return MultitargetSchedule(targets, target_objectives, iterations, iptm_threshold=settings.get('multitarget_swap_threshold', 0.5), swap_patience=settings.get('multitarget_swap_patience', 20), warmup_swap_patience=settings.get('multitarget_warmup_patience'), target_chain=target_chain, epitope_rotation_interval=epitope_rotation_interval, merged_target_gradients=merged_gradient_targets(design_settings, design_stage) > 1, max_detarget_iptm=settings.get('max_detarget_iptm', 0.4), detarget_check_interval=settings.get('detarget_check_interval', DEFAULT_DETARGET_CHECK_INTERVAL), max_detarget_rounds=settings.get('max_detarget_rounds', 10), max_detarget_interface_residues=settings.get('max_detarget_interface_residues_final', DEFAULT_DETARGET_INTERFACE_RESIDUES))

def build_design_schedule(design_settings: BinderDesignSettings, target_states: ProteinStates, losses: dict[str, DesignLoss], iterations: int, conformation_random_key: Array, induced_fit_active: bool=True, target_schedule: 'IterationLimitedDesignSchedule | None'=None, design_stage: str=MERGED_GRADIENT_STAGE) -> 'DesignSchedule':
    settings = design_settings.settings
    target_chain = design_settings.target_chain_prefix
    targets = {state.name: target_states[state.name][state.target_chain] for state in design_settings.prepared_states}
    target_schedule = target_schedule.restart_for_stage(iterations, targets, merged_gradient_targets(design_settings, design_stage) > 1) if target_schedule is not None else build_target_schedule(design_settings, target_states, iterations, design_stage)
    if not targets:
        return target_schedule
    induced_fit = induced_fit_active and any(name.startswith(('induced_fit_interface', 'induced_fit_global')) for name in losses)
    conformation_groups = (tuple(targets), (BINDER_ALONE,)) if induced_fit else design_settings.binder_shapes
    if not conformation_groups:
        return target_schedule
    remove_target_chains = {target_chain_name(target_chain, name): None for name in targets}
    target_chain_overrides = {name: {**remove_target_chains, target_chain_name(target_chain, name): targets[name]} if name in targets else remove_target_chains for group in conformation_groups for name in group}
    induced_fit_schedule = InducedFitSchedule(target_chain_overrides, conformation_groups, iterations, conformation_random_key)
    if len(targets) > 1:
        return MixedDesignSchedule([target_schedule, induced_fit_schedule], iterations_per_schedule=[settings.get('multitarget_steps', 1), settings.get('induced_fit_steps', 1)])
    return induced_fit_schedule

@runtime_checkable
class DesignSchedule(Protocol):
    def select_protein_states(self, protein_states: ProteinStates, predictions: StructurePredictions | None) -> ProteinStates:
        ...

    def should_update_sequence(self, protein_states: ProteinStates, predictions: StructurePredictions, accumulated_gradients: dict[str, list[Array]]) -> bool:
        ...

    def record_sequence_update(self) -> None:
        ...

    def is_complete(self, predictions: StructurePredictions | None) -> bool:
        ...

class MixedDesignSchedule(DesignSchedule):
    def __init__(self, design_schedules: list[DesignSchedule], iterations_per_schedule: list[int]):
        self.design_schedules = design_schedules
        self.iterations_per_schedule = iterations_per_schedule
        self.active_schedule_index = 0
        self.schedule_sequence_updates = 0

    def select_protein_states(self, protein_states: ProteinStates, predictions: StructurePredictions | None) -> ProteinStates:
        return self.design_schedules[self.active_schedule_index].select_protein_states(protein_states, predictions)

    def should_update_sequence(self, protein_states: ProteinStates, predictions: StructurePredictions, accumulated_gradients: dict[str, list[Array]]) -> bool:
        return self.design_schedules[self.active_schedule_index].should_update_sequence(protein_states, predictions, accumulated_gradients)

    def record_sequence_update(self) -> None:
        self.design_schedules[self.active_schedule_index].record_sequence_update()
        self.schedule_sequence_updates += 1
        if self.schedule_sequence_updates >= self.iterations_per_schedule[self.active_schedule_index]:
            self.schedule_sequence_updates = 0
            self.active_schedule_index = (self.active_schedule_index + 1) % len(self.design_schedules)

    def is_complete(self, predictions: StructurePredictions | None) -> bool:
        return all(schedule.is_complete(predictions) for schedule in self.design_schedules)

class IterationLimitedDesignSchedule(DesignSchedule):
    def __init__(self, iterations: int):
        self.iterations = iterations
        self.completed_sequence_updates = 0

    def select_protein_states(self, protein_states: ProteinStates, predictions: StructurePredictions | None) -> ProteinStates:
        raise NotImplementedError

    def should_update_sequence(self, protein_states: ProteinStates, predictions: StructurePredictions, accumulated_gradients: dict[str, list[Array]]) -> bool:
        return True

    def record_sequence_update(self) -> None:
        self.completed_sequence_updates += 1

    def is_complete(self, predictions: StructurePredictions | None) -> bool:
        return self.completed_sequence_updates >= self.iterations

    def restart_for_stage(self, iterations: int, targets: dict[str, Protein], merged_target_gradients: bool=False) -> 'IterationLimitedDesignSchedule':
        self.iterations, self.completed_sequence_updates = iterations, 0
        return self

class FixedTargetSchedule(IterationLimitedDesignSchedule):
    def select_protein_states(self, protein_states: ProteinStates, predictions: StructurePredictions | None) -> ProteinStates:
        return protein_states

def losses_for_active_states(losses: dict[str, DesignLoss], protein_states: ProteinStates) -> dict[str, DesignLoss]:
    return {loss_name: design_loss for loss_name, design_loss in losses.items() if design_loss.required_states <= set(protein_states)}

class TargetTransition(NamedTuple):
    name: str
    reached: Callable[['MultitargetSchedule', str, float, float], bool]

def fixed_cadence_reached(schedule: 'MultitargetSchedule', objective: str, interface_confidence: float, interface_residues: float) -> bool:
    return schedule.iterations_on_target + 1 >= schedule.rounds_before_target_swap()

def detarget_avoided(schedule: 'MultitargetSchedule', interface_confidence: float, interface_residues: float) -> bool:
    """An off-target the binder has actually come off: quiet at the interface and no longer touching it.

    Interface confidence alone says a binder got away when it did not, because a short one holds a
    whole epitope at a confidence no threshold rejects, so the repulsion keeps the round while any
    contact remains and the binder is pushed off rather than left where it sits."""
    return (interface_confidence <= schedule.max_detarget_iptm and interface_residues <= schedule.max_detarget_interface_residues) or schedule.iterations_on_target + 1 >= schedule.max_detarget_rounds

def binding_confidence_reached(schedule: 'MultitargetSchedule', interface_confidence: float, interface_residues: float) -> bool:
    return interface_confidence > schedule.iptm_threshold or schedule.iterations_on_target + 1 >= schedule.rounds_before_target_swap()

def objective_gated_reached(schedule: 'MultitargetSchedule', objective: str, interface_confidence: float, interface_residues: float) -> bool:
    return detarget_avoided(schedule, interface_confidence, interface_residues) if objective == 'detarget' else binding_confidence_reached(schedule, interface_confidence, interface_residues)

FIXED_CADENCE = TargetTransition('fixed cadence', fixed_cadence_reached)
OBJECTIVE_GATED = TargetTransition('objective gated', objective_gated_reached)

class MultitargetSchedule(IterationLimitedDesignSchedule):
    def __init__(self, targets: dict[str, Protein], target_objectives: dict[str, str], iterations: int, iptm_threshold: float=0.5, swap_patience: int=20, warmup_swap_patience: int | None=None, target_chain: str='target', epitope_rotation_interval: int | None=None, merged_target_gradients: bool=False, max_detarget_iptm: float=0.4, detarget_check_interval: int=1, max_detarget_rounds: int=10, max_detarget_interface_residues: float | None=3):
        super().__init__(iterations)
        self.epitope_rotation_interval = None if epitope_rotation_interval is None else max(1, int(epitope_rotation_interval))
        self.target_entries = [(target_name, target, target_objectives[target_name]) for target_name, target in targets.items()]
        self.target_chain = target_chain
        self.iptm_threshold = iptm_threshold
        self.swap_patience = swap_patience
        self.warmup_swap_patience = swap_patience if warmup_swap_patience is None else warmup_swap_patience
        self.active_target_index = 0
        self.iterations_on_target = 0
        self.target_warmup_complete = False
        self.stage_peak_predictions: dict[str, StructurePrediction] = {}
        self.latest_stage_predictions: dict[str, StructurePrediction] = {}
        self.merged_target_gradients = merged_target_gradients
        self.targets_in_accumulated_gradient: set[str] = set()
        self.max_detarget_iptm = max_detarget_iptm
        self.detarget_check_interval = max(1, int(detarget_check_interval))
        self.max_detarget_rounds = max(1, int(max_detarget_rounds))
        #a campaign that writes the ceiling off keeps the confidence criterion alone, where a missing number would compare against nothing
        self.max_detarget_interface_residues = float('inf') if max_detarget_interface_residues is None else float(max_detarget_interface_residues)
        self.rounds_since_detarget_check = 0
        self.detarget_position = 0
        self.rotated_target_index = self.active_target_index = next(iter(self._rotated_and_detargeted_indices()[0]), 0)
        self.target_transition = self._select_target_transition()

    def _select_target_transition(self) -> TargetTransition:
        return FIXED_CADENCE if self.merged_target_gradients or self.epitope_rotation_interval is not None else OBJECTIVE_GATED

    def restart_for_stage(self, iterations: int, targets: dict[str, Protein], merged_target_gradients: bool=False) -> 'MultitargetSchedule':
        self.target_entries = [(name, targets[name], objective) for name, _, objective in self.target_entries]
        self.stage_peak_predictions, self.latest_stage_predictions, self.targets_in_accumulated_gradient = ({}, {}, set())
        self.merged_target_gradients = merged_target_gradients
        self.target_transition = self._select_target_transition()
        return super().restart_for_stage(iterations, targets)

    def rounds_before_target_swap(self) -> int:
        if self.merged_target_gradients:
            return 1
        configured_rounds_on_target = self.epitope_rotation_interval if self.epitope_rotation_interval is not None else self.swap_patience if self.target_warmup_complete else self.warmup_swap_patience
        return max(1, min(configured_rounds_on_target, self.iterations // len(self._rotated_and_detargeted_indices()[0])))

    def _target_chain_name(self, target_name: str) -> str:
        return target_chain_name(self.target_chain, target_name)

    def record_stage_peak(self, predictions: StructurePredictions | None) -> None:
        target_name, _, objective = self.target_entries[self.active_target_index]
        if predictions is None or target_name not in predictions:
            return
        self.latest_stage_predictions[target_name] = predictions[target_name]
        peak, interface_confidence = (self.stage_peak_predictions.get(target_name), predictions[target_name].metrics['iptm'])
        if peak is None or (interface_confidence < peak.metrics['iptm'] if objective == 'detarget' else interface_confidence > peak.metrics['iptm']):
            self.stage_peak_predictions[target_name] = predictions[target_name]

    def should_update_sequence(self, protein_states: ProteinStates, predictions: StructurePredictions, accumulated_gradients: dict[str, list[Array]]) -> bool:
        self.record_stage_peak(predictions)
        if not self.merged_target_gradients:
            return True
        self.targets_in_accumulated_gradient.add(self.target_entries[self.active_target_index][0])
        return len(self.targets_in_accumulated_gradient) >= len(self.target_entries)

    def record_sequence_update(self) -> None:
        self.targets_in_accumulated_gradient = set()
        super().record_sequence_update()

    def _rotated_and_detargeted_indices(self) -> tuple[list[int], list[int]]:
        objective_indices = {objective: [index for index, entry in enumerate(self.target_entries) if entry[2] == objective] for objective in ('target', 'detarget')}
        if not objective_indices['target'] or self.merged_target_gradients or self.epitope_rotation_interval is not None:
            return list(range(len(self.target_entries))), []
        return objective_indices['target'], objective_indices['detarget']

    def _advance_active_target(self, objective: str) -> None:
        rotated_indices, detargeted_indices = self._rotated_and_detargeted_indices()
        if objective == 'detarget' and detargeted_indices:
            self.rounds_since_detarget_check = 0
        else:
            rotation_position = (rotated_indices.index(self.rotated_target_index) + 1) % len(rotated_indices) if self.rotated_target_index in rotated_indices else 0
            self.rotated_target_index = rotated_indices[rotation_position]
            self.target_warmup_complete = self.target_warmup_complete or rotation_position == 0
        detarget_due = bool(detargeted_indices) and objective != 'detarget' and self.rounds_since_detarget_check >= self.detarget_check_interval
        self.active_target_index = detargeted_indices[self.detarget_position] if detarget_due else self.rotated_target_index
        if detarget_due:
            self.detarget_position = (self.detarget_position + 1) % len(detargeted_indices)

    def _count_round_towards_detarget_visit(self, objective: str) -> None:
        if self.target_transition is OBJECTIVE_GATED and objective != 'detarget':
            self.rounds_since_detarget_check += 1

    def select_protein_states(self, protein_states: ProteinStates, predictions: StructurePredictions | None) -> ProteinStates:
        target_name, _, objective = self.target_entries[self.active_target_index]
        if predictions is not None and target_name in predictions:
            self.record_stage_peak(predictions)
            interface_confidence = float(predictions[target_name].metrics['iptm'])
            #only an off-target is decided on contact, and only its own rounds pay for measuring it
            interface_residues = float(interface_residues_metric(protein_states, predictions, target_name, target=self._target_chain_name(target_name))) if objective == 'detarget' else 0.0
            self._count_round_towards_detarget_visit(objective)
            if self.target_transition.reached(self, objective, interface_confidence, interface_residues):
                self.iterations_on_target = 0
                self._advance_active_target(objective)
            else:
                self.iterations_on_target += 1
        target_name, target, _ = self.target_entries[self.active_target_index]
        return {target_name: replace_target_chains(protein_states[target_name], {self._target_chain_name(target_name): target})}

def stage_filter_predictions(design_schedule: DesignSchedule, predictions: StructurePredictions, cumulative: bool) -> StructurePredictions:
    peak_predictions = getattr(design_schedule, 'stage_peak_predictions', {}) if cumulative else {}
    return {**predictions, **{name: prediction for name, prediction in peak_predictions.items() if name in predictions}}

def rotating_round_predictions(design_schedule: DesignSchedule, predictions: StructurePredictions) -> StructurePredictions:
    return {**getattr(design_schedule, 'latest_stage_predictions', {}), **predictions}

def replace_target_chains(protein_complex: dict[str, Protein], target_chain_overrides: dict[str, Protein | None]) -> dict[str, Protein]:
    return {chain_name: protein for chain_name, protein in {**protein_complex, **target_chain_overrides}.items() if protein is not None}

def _sample_conformation_index(random_key: Array, conformation_count: int, excluded_conformation: int | None=None) -> int:
    available_conformations = [index for index in range(conformation_count) if index != excluded_conformation] or [0]
    return available_conformations[int(jax.random.randint(random_key, (), 0, len(available_conformations)))]

class InducedFitSchedule(IterationLimitedDesignSchedule):
    def __init__(self, target_chain_overrides: dict[str, dict[str, Protein | None]], binder_shapes: tuple[tuple[str, ...], ...], iterations: int, random_key: Array):
        super().__init__(iterations)
        self.target_chain_overrides = target_chain_overrides
        self.binder_shapes = binder_shapes
        self.random_key = random_key
        self.current_conformation = 0, 0

    def select_protein_states(self, protein_states: ProteinStates, predictions: StructurePredictions | None) -> ProteinStates:
        self.random_key, conformation_random_key, state_random_key = jax.random.split(self.random_key, 3)
        current_conformation_index, current_state_index = self.current_conformation
        next_conformation_index = _sample_conformation_index(conformation_random_key, len(self.binder_shapes), excluded_conformation=current_conformation_index)
        next_state_index = _sample_conformation_index(state_random_key, len(self.binder_shapes[next_conformation_index]))
        compared_states = {self.binder_shapes[current_conformation_index][current_state_index], self.binder_shapes[next_conformation_index][next_state_index]}
        induced_fit_states = {state_name: replace_target_chains(protein_states[state_name] if state_name in protein_states else collect_shared_chains(protein_states)[1], self.target_chain_overrides[state_name]) for state_name in compared_states}
        self.current_conformation = next_conformation_index, next_state_index
        return induced_fit_states
