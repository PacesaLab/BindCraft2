import os
import time
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple
from bindcraft.prediction import ProteinPredictor
from bindcraft.filters import binder_assembly_sequence, binder_chain_sequences, binder_target_contact_masks, confidence_stage_filters, design_sequence_report, design_stage_filters, evaluate_design_filters, protomer_identity_fraction, validates_unbound_binder, INTERFACE_PDAE_METRICS
from bindcraft.loss import align_binder_coordinates, binder_copy_chains, chain_atom_coordinates, chain_residue_slices, multidomain_linker_residues
from bindcraft.protein import StructurePrediction, StructurePredictions, Protein, ResidueFlags, ProteinStates, has_residue_flag, has_resolved_atom, superposed_on_binder, write_structure, BINDER_ALONE
from bindcraft.campaign_log import candidate_outcome, redesigns_kept, exhausted_redesign_window
from bindcraft.campaign_output import CampaignProgress, RANKING_METRIC, REFOLD_STAGE, accepted_state_suffixes, append_campaign_metrics, ranking_value, redesigned_sequence, stage_folder, target_ordered_row, timing_stamp, weighted_target_order
from bindcraft.protein_preparation import frame_holding_target, receptor_chain_layouts, states_holding_the_frame
from bindcraft.settings import BinderDesignSettings, DEFAULT_SETTINGS

_NOT_DESIGN = int(~ResidueFlags.DESIGN)

class ValidatedBinder(NamedTuple):
    predictions: StructurePredictions
    metrics: dict[str, float | None]
    candidate_number: int = 0
    model_metrics: dict[str, dict[str, float]] = {}

class ValidationEnsemble(NamedTuple):
    predictions: StructurePredictions
    model_predictions: dict[str, StructurePredictions]
    failed_confidence: list[str] = []

INDUCED_FIT_DESIGNED_SHARE = 0.25

def induced_fit_mobile_residues(binder: Protein, binder_alone: Protein, rmsd_threshold: float=2.0, shell: int=1, designed_share: float=INDUCED_FIT_DESIGNED_SHARE) -> Array:
    coordinates, mask = chain_atom_coordinates(binder)
    reference_coordinates, reference_mask = chain_atom_coordinates(binder_alone)
    valid_mask = mask * reference_mask
    deviation = jnp.sqrt(jnp.square(align_binder_coordinates(reference_coordinates, coordinates, valid_mask) - coordinates).sum(-1) + 1e-08)
    resolved_residues = valid_mask > 0
    mobile_residues = (deviation > rmsd_threshold) & resolved_residues
    for _ in range(int(shell)):
        mobile_residues = mobile_residues | jnp.roll(mobile_residues, 1).at[0].set(False) | jnp.roll(mobile_residues, -1).at[-1].set(False)
    mobile_residues = mobile_residues & resolved_residues
    held_ceiling = int(resolved_residues.sum()) - int(designed_share * int(resolved_residues.sum()))
    if int(mobile_residues.sum()) > held_ceiling:
        released = jnp.argsort(jnp.where(mobile_residues, deviation, jnp.inf))[:int(mobile_residues.sum()) - held_ceiling]
        mobile_residues = mobile_residues.at[released].set(False)
    return mobile_residues

def mark_redesign_residues(protein_complex: dict[str, Protein], binder: str, target: str, keep_interface: bool, cutoff: float=4.0, mobile_binder_residues: Array | None=None, hold_framework: bool=False, multi_chain_binder: tuple[str, ...]=(), linker_binder_residues: Array | None=None) -> dict[str, Protein]:
    redesign_complex = dict(protein_complex)
    redesign_complex[target] = protein_complex[target].replace(flags=(protein_complex[target].flags & _NOT_DESIGN).astype(jnp.uint8))
    held_residue_masks = {}
    for name in binder_copy_chains(protein_complex, binder):
        held_residues = jnp.zeros(len(protein_complex[name]), dtype=bool) if mobile_binder_residues is None else mobile_binder_residues
        if keep_interface or mobile_binder_residues is not None:
            held_residues = held_residues | binder_target_contact_masks(protein_complex[name], protein_complex[target], cutoff)[0]
        if hold_framework:
            held_residues = held_residues | has_residue_flag(protein_complex[name].flags, ResidueFlags.TEMPLATE) #mask out fold conditioning scaffold
        if linker_binder_residues is not None:
            held_residues = held_residues | linker_binder_residues #mask out the inter-domain linker, multi-domain case
        held_residue_masks[name] = held_residues
    if multi_chain_binder:
        tied_residues = jnp.stack([held_residue_masks[name] for name in multi_chain_binder]).any(0)
        held_residue_masks.update({name: tied_residues for name in multi_chain_binder})
    for name, held_residues in held_residue_masks.items():
        redesign_residue_flags = jnp.where(held_residues, protein_complex[name].flags & _NOT_DESIGN, protein_complex[name].flags | int(ResidueFlags.DESIGN))
        redesign_complex[name] = protein_complex[name].replace(flags=redesign_residue_flags.astype(jnp.uint8))
    return redesign_complex

#for oligomers
def share_binder_chain_sequences(protein_complex: dict[str, Protein], binder: str, multi_chain_binder: tuple[str, ...]=()) -> dict[str, Protein]:
    return {name: protein.replace(sequence=protein_complex[binder].sequence) if name in multi_chain_binder and name != binder else protein for name, protein in protein_complex.items()}

#pick which target MPNN sees, for multitargeting
def redesign_target_chain(protein_complex: dict[str, Protein], binder: str) -> str:
    return next(name for name in sorted(protein_complex) if name not in binder_copy_chains(protein_complex, binder))

def prepare_multitarget_redesign(target_states: ProteinStates, binder: str, keep_interface: bool, cutoff: float=4.0, multi_chain_binder: tuple[str, ...]=(), mobile_binder_residues: Array | None=None, hold_framework: bool=False, linker_binder_residues: Array | None=None) -> ProteinStates:
    if any((not has_resolved_atom(protein_complex[binder].atom_mask, 'CA').any() for protein_complex in target_states.values())):
        raise ValueError(f'Multitarget redesign needs the predicted complex of every positive target; the {binder!r} chain it was given has no backbone')
    redesign_states = {name: mark_redesign_residues(protein_complex, binder, redesign_target_chain(protein_complex, binder), keep_interface, cutoff, mobile_binder_residues, hold_framework, multi_chain_binder, linker_binder_residues) for name, protein_complex in target_states.items()}
    union_of_interfaces = jnp.zeros(len(next(iter(redesign_states.values()))[binder]), dtype=bool)
    for protein_complex in redesign_states.values():
        union_of_interfaces |= jnp.logical_not(has_residue_flag(protein_complex[binder].flags, ResidueFlags.DESIGN))
    for protein_complex in redesign_states.values():
        binder_flags = jnp.where(union_of_interfaces, protein_complex[binder].flags & _NOT_DESIGN, protein_complex[binder].flags | int(ResidueFlags.DESIGN))
        protein_complex[binder] = protein_complex[binder].replace(flags=binder_flags.astype(jnp.uint8))
    return redesign_states

def ensemble_mean_predictions(protein_states: ProteinStates, model_predictions: dict[str, StructurePredictions]) -> StructurePredictions:
    predictions: StructurePredictions = {}
    for name in protein_states:
        metric_names = {metric_name for model_outputs in model_predictions.values() for metric_name in model_outputs[name].metrics}
        metrics = {}
        for metric_name in metric_names:
            values = [model_outputs[name].metrics[metric_name] for model_outputs in model_predictions.values() if metric_name in model_outputs[name].metrics]
            metrics[metric_name] = sum(values) / len(values)
        predictions[name] = StructurePrediction(protein_complex=next(iter(model_predictions.values()))[name].protein_complex, metrics=metrics)
    return predictions

REACHABLE_CONFIDENCE_BOUNDS = {'plddt': (0.0, 1.0), 'ptm': (0.0, 1.0), 'iptm': (0.0, 1.0), 'pae': (0.0, 0.0)}

def best_reachable_ensemble(model_predictions: dict[str, StructurePredictions], model_count: int, higher: bool) -> StructurePredictions:
    folded = list(model_predictions.values())
    predictions: StructurePredictions = {}
    for name in folded[0]:
        metrics = {}
        for metric_name in {metric_name for model_outputs in folded for metric_name in model_outputs[name].metrics}:
            values = [model_outputs[name].metrics[metric_name] for model_outputs in folded if metric_name in model_outputs[name].metrics]
            bounds = REACHABLE_CONFIDENCE_BOUNDS.get(metric_name)
            best = (bounds[1] if higher else bounds[0]) if bounds else sum(values) / len(values)
            metrics[metric_name] = (sum(values) + (model_count - len(values)) * best) / model_count
        predictions[name] = StructurePrediction(protein_complex=folded[0][name].protein_complex, metrics=metrics)
    return predictions

def checks_out_of_reach(stage_filters: dict, protein_states: ProteinStates, model_predictions: dict[str, StructurePredictions], model_count: int) -> list[str]:
    out_of_reach: list[str] = []
    for higher in (True, False):
        bounded = {name: design_filter for name, design_filter in stage_filters.items() if design_filter.threshold is not None and design_filter.higher == higher}
        failed = evaluate_design_filters(bounded, protein_states, best_reachable_ensemble(model_predictions, model_count, higher))[0] if bounded else True
        out_of_reach += [] if failed is True else list(failed)
    return out_of_reach

def predict_validation_ensemble(structure_predictor: ProteinPredictor, protein_states: ProteinStates, models: tuple[str, ...], stage_filters: dict | None=None) -> ValidationEnsemble:
    confidence_filters = confidence_stage_filters(stage_filters) if stage_filters else {}
    model_predictions: dict[str, StructurePredictions] = {}
    for model_number, model_name in enumerate(models, start=1):
        model_predictions[model_name] = structure_predictor.predict(protein_states, model=model_name)
        failed = evaluate_design_filters(confidence_filters, protein_states, model_predictions[model_name])[0] if confidence_filters else True
        if failed is not True:
            return ValidationEnsemble(ensemble_mean_predictions(protein_states, model_predictions), model_predictions, list(failed))
        if stage_filters is not None and model_number < len(models) and checks_out_of_reach(stage_filters, protein_states, model_predictions, len(models)):
            break
    return ValidationEnsemble(ensemble_mean_predictions(protein_states, model_predictions), model_predictions, [])

def per_model_design_scores(stage_filters: dict, protein_states: ProteinStates, model_predictions: dict[str, StructurePredictions], prediction_state: str, binder: str, target: str) -> dict[str, dict[str, float]]:
    return {model_name: {**evaluate_design_filters(stage_filters, protein_states, predictions)[1],
                         **{name: interface_pdae(protein_states, predictions, prediction_state=prediction_state, binder=binder, target=target) for name, interface_pdae in INTERFACE_PDAE_METRICS.items()}}
            for model_name, predictions in model_predictions.items()}

def decode_sequence_candidates(mpnn_model: ProteinPredictor, redesign_rotation: ProteinStates, tied_redesign_states: ProteinStates, candidate_count: int) -> list[tuple[str, dict[str, Protein]]]:
    rotation_states = tuple(redesign_rotation)
    states_for_candidates = [rotation_states[index % len(rotation_states)] for index in range(candidate_count)]
    if tied_redesign_states:
        decoded = mpnn_model.predict_tied_candidates(tied_redesign_states, candidate_count)
        return [(state, decoded[index][state].protein_complex) for index, state in enumerate(states_for_candidates)]
    decoded = {state: iter(mpnn_model.predict_candidates({state: redesign_rotation[state]}, candidate_count=states_for_candidates.count(state))) for state in rotation_states}
    return [(state, next(decoded[state])[state].protein_complex) for state in states_for_candidates]

def redesign_linker_residues(settings: dict, protein_complex: dict[str, Protein], binder: str, design_pae: Array | None, trajectory_seed: int) -> Array | None:
    if design_pae is None or not settings.get('weights_multidomain') or not settings.get('mpnn_fix_linker', True):
        return None
    binder_slice = chain_residue_slices(protein_complex)[binder]
    return multidomain_linker_residues(settings, protein_complex[binder], design_pae[binder_slice, binder_slice], trajectory_seed)

class RedesignContext(NamedTuple):
    decode_states: ProteinStates
    joint_decode_states: ProteinStates
    validation_states: ProteinStates
    shared_binder_chains: tuple[str, ...]
    decode_source: str

def prepare_binder_redesign(protein_complex: dict[str, Protein], design_settings: BinderDesignSettings, binder: str, target: str, prediction_state: str, target_states: ProteinStates | None=None, predicted_states: ProteinStates | None=None, multi_chain_binder: tuple[str, ...]=(), binder_alone_complex: dict[str, Protein] | None=None, design_pae: Array | None=None, trajectory_seed: int=0) -> RedesignContext:
    settings = design_settings.settings
    keep_interface = not settings.get('redesign_interface', DEFAULT_SETTINGS['redesign_interface'])
    #fold switching case
    if any((BINDER_ALONE in group for group in design_settings.binder_shapes)):
        binder_alone_complex = None
    mobile_residues = induced_fit_mobile_residues(protein_complex[binder], binder_alone_complex[binder], float(settings.get('induced_fit_mpnn_threshold', 2.0)), int(settings.get('induced_fit_mpnn_shell', 1)), float(settings.get('induced_fit_mpnn_designed_share', INDUCED_FIT_DESIGNED_SHARE))) if binder_alone_complex else None #induced fit case
    if mobile_residues is not None:
        print(f'induced fit: holding {int(mobile_residues.sum())} of {len(protein_complex[binder])} residue(s) that move between the two states through ProteinMPNN', flush=True)
    hold_framework = bool(settings.get('binder_scaffold')) #fold conditioning case
    if hold_framework:
        print(f"scaffold: holding {sum((int(has_residue_flag(protein_complex[name].flags, ResidueFlags.TEMPLATE).sum()) for name in binder_copy_chains(protein_complex, binder)))} framework residue(s) through ProteinMPNN", flush=True)
    linker_residues = redesign_linker_residues(settings, protein_complex, binder, design_pae, trajectory_seed) #multi-domain case
    redesign_complex = mark_redesign_residues(protein_complex, binder, target, keep_interface, mobile_binder_residues=mobile_residues, hold_framework=hold_framework, multi_chain_binder=multi_chain_binder, linker_binder_residues=linker_residues)
    #a detarget is validated against but never decoded on
    detarget_states = {state.name for state in design_settings.prepared_states if state.objective == 'detarget'}
    predicted_target_states = {name: state_complex for name, state_complex in (predicted_states or {}).items() if name not in detarget_states}
    decode_states = prepare_multitarget_redesign(predicted_target_states, binder, keep_interface, multi_chain_binder=multi_chain_binder, mobile_binder_residues=mobile_residues, hold_framework=hold_framework, linker_binder_residues=linker_residues) if len(predicted_target_states) > 1 else {prediction_state: redesign_complex}
    #multitargeting: tied redesign reads every target at once
    joint_decode_states = decode_states if len(decode_states) > 1 and settings.get('multitarget_tied_redesign', True) else {}
    return RedesignContext(decode_states, joint_decode_states, target_states or {prediction_state: redesign_complex}, multi_chain_binder, '+'.join(joint_decode_states))

def distinct_sequence_candidates(mpnn_model: ProteinPredictor, redesign: RedesignContext, protein_complex: dict[str, Protein], binder: str, candidate_count: int, project_folder: str | None=None, redraws: int=3):
    drawn: set[str] = set()
    empty_redraws = 0
    while len(drawn) < candidate_count and empty_redraws <= redraws:
        fresh = 0
        for rotation_state, decoded_complex in decode_sequence_candidates(mpnn_model, redesign.decode_states, redesign.joint_decode_states, candidate_count - len(drawn)):
            designed_binder = {name: decoded_complex[name].replace(flags=protein_complex[name].flags) for name in binder_copy_chains(protein_complex, binder)}
            sequence = binder_assembly_sequence(designed_binder, binder)
            if sequence in drawn or (project_folder and redesigned_sequence(project_folder, sequence)):
                continue
            drawn.add(sequence)
            fresh += 1
            yield rotation_state, decoded_complex, designed_binder
        empty_redraws = 0 if fresh else empty_redraws + 1


def candidate_validation_states(redesign: RedesignContext, designed_binder: dict[str, Protein], binder: str, unbound_binder: bool=True) -> ProteinStates:
    protein_states = {name: share_binder_chain_sequences({**target_complex, **designed_binder}, binder, redesign.shared_binder_chains) for name, target_complex in redesign.validation_states.items()}
    if unbound_binder:
        protein_states[BINDER_ALONE] = share_binder_chain_sequences(designed_binder, binder, redesign.shared_binder_chains)
    return protein_states

def decoded_sequence_metrics(decoded_complex: dict[str, Protein], binder: str) -> dict[str, float]:
    protomer_identity = protomer_identity_fraction(decoded_complex, binder)
    return {'Protomer_Identity_Fraction': protomer_identity} if protomer_identity is not None else {}

def write_refolded_candidate(refold_folder: str, candidate: str, predictions: StructurePredictions, prediction_state: str, failed_filters: list[str], settings: dict, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None, design_settings: BinderDesignSettings | None=None, protein_states: ProteinStates | None=None) -> str | None:
    """Keep the complexes one redesigned sequence refolded to, one per target it was scored against.

    A multitarget candidate is scored on every target, so a single complex left the stage unable to say
    which target it was, and the other targets unreadable. The suffixes come from the ranking stage's own
    accepted_state_suffixes, so a design reads the same in both stages; below two targets that helper
    returns no suffix and the name is unchanged."""
    if failed_filters and not settings.get('save_failed_refolds', DEFAULT_SETTINGS['save_failed_refolds']):
        return None
    prediction = predictions[prediction_state]
    metadata = {'design': candidate, 'outcome': 'rejected' if failed_filters else 'passed', **({'failed_filters': ','.join(failed_filters)} if failed_filters else {})}
    state_suffixes = accepted_state_suffixes(design_settings.prepared_states, predictions) if design_settings else {}
    #a target that does not hold its own frame is superposed on the binder, so the targets can be read against each other
    framed_states = states_holding_the_frame({name: state for name, state in protein_states.items() if name != BINDER_ALONE}, frame_holding_target(design_settings), design_settings.target_chain_prefix) if design_settings and protein_states else set(predictions)
    written = os.path.join(refold_folder, 'Complexes', f'{candidate}{state_suffixes.get(prediction_state, "")}.cif')
    write_structure(prediction.protein_complex, written, plddt=prediction.metrics.get('plddt'), metadata=metadata, receptor_chains=receptor_chains)
    for target_state in predictions:
        if target_state in (prediction_state, BINDER_ALONE):
            continue
        state = predictions[target_state]
        state_complex = state.protein_complex if target_state in framed_states else superposed_on_binder(state.protein_complex, prediction.protein_complex)
        write_structure(state_complex, os.path.join(refold_folder, 'Complexes', f'{candidate}{state_suffixes.get(target_state, f"_{target_state}")}.cif'), plddt=state.metrics.get('plddt'), metadata=metadata, receptor_chains=receptor_chains)
    if BINDER_ALONE in predictions and settings.get('save_binder_monomers', DEFAULT_SETTINGS['save_binder_monomers']):
        monomer = predictions[BINDER_ALONE]
        write_structure(superposed_on_binder(monomer.protein_complex, prediction.protein_complex), os.path.join(refold_folder, 'BinderMonomer', f'{candidate}_monomer.cif'), plddt=monomer.metrics.get('plddt'), metadata=metadata, receptor_chains=receptor_chains)
    return written

def redesign_and_validate_binders(protein_complex: dict[str, Protein],
                                  structure_predictor: ProteinPredictor,
                                  mpnn_model: ProteinPredictor, design_settings: BinderDesignSettings,
                                  validation_models: tuple[str, ...],
                                  binder: str='binder',
                                  target: str='target',
                                  prediction_state: str='complex',
                                  target_states: ProteinStates | None=None,
                                  predicted_states: ProteinStates | None=None,
                                  multi_chain_binder: tuple[str, ...]=(),
                                  binder_alone_complex: dict[str, Protein] | None=None, #for induced-fit only
                                  design_pae: Array | None=None, #for multi-domain only
                                  trajectory_seed: int=0,
                                  design: str='design',
                                  design_hash: str='',
                                  candidates_csv: str | None=None,
                                  project_folder: str | None=None) -> list[ValidatedBinder]:
    settings = design_settings.settings
    redesign = prepare_binder_redesign(protein_complex, design_settings, binder, target, prediction_state, target_states, predicted_states, multi_chain_binder, binder_alone_complex, design_pae, trajectory_seed)
    candidate_count = settings.get('sequence_candidates', DEFAULT_SETTINGS['sequence_candidates'])
    kept_sequence_count = settings.get('kept_sequences', DEFAULT_SETTINGS['kept_sequences'])
    accepted_sequence_limit = settings.get('enough_passing_sequences', DEFAULT_SETTINGS['enough_passing_sequences']) or candidate_count
    accepted_binders: list[ValidatedBinder] = []
    candidate_number = 0
    unbound_binder = binder_alone_complex is not None or validates_unbound_binder(design_settings.filters)
    campaign_progress = CampaignProgress(project_folder, 0) if project_folder else None
    refold_folder = stage_folder(project_folder, REFOLD_STAGE) if project_folder else None
    for candidate_number, (rotation_state, decoded_complex, designed_binder) in enumerate(distinct_sequence_candidates(mpnn_model, redesign, protein_complex, binder, candidate_count, project_folder), start=1):
        protein_states = candidate_validation_states(redesign, designed_binder, binder, unbound_binder)
        stage_filters = design_stage_filters(design_settings, protein_states, 'final', campaign_filters=design_settings.filters)
        reprediction_started = time.time()
        ensemble = predict_validation_ensemble(structure_predictor, protein_states, validation_models, stage_filters)
        predictions = ensemble.predictions
        filter_result, metrics = evaluate_design_filters(stage_filters, protein_states, predictions)
        failed_filters = ensemble.failed_confidence or ([] if filter_result is True else list(filter_result))
        metrics.update(decoded_sequence_metrics(decoded_complex, binder))
        interface_pdae_scores = {name: interface_pdae(protein_states, predictions, prediction_state=prediction_state, binder=binder, target=target) for name, interface_pdae in INTERFACE_PDAE_METRICS.items()}
        print(candidate_outcome(candidate_number, candidate_count, redesign.decode_source or rotation_state, failed_filters, metrics, bool(settings.get('binder_scaffold')), tuple(redesign.validation_states)), flush=True)
        if candidates_csv:
            append_campaign_metrics(candidates_csv, target_ordered_row({'design': f'{design}_candidate{candidate_number}', 'length': len(binder_chain_sequences(designed_binder, binder).split('/')[0]), 'hash': design_hash, 'outcome': 'rejected' if failed_filters else 'passed', 'failed_filters': ','.join(failed_filters), **design_sequence_report(predictions, prediction_state=prediction_state, binder=binder, target=target, receptor_chains=receptor_chain_layouts(design_settings)), **metrics, **interface_pdae_scores, 'Timing': timing_stamp(worker=os.environ.get('BINDCRAFT_WORKER_ID', '0'), start=reprediction_started, reprediction=time.time() - reprediction_started)}, weighted_target_order(design_settings.prepared_states)))
        if refold_folder:
            write_refolded_candidate(refold_folder, f'{design}_candidate{candidate_number}', predictions, prediction_state, failed_filters, settings, receptor_chain_layouts(design_settings), design_settings, protein_states)
        if campaign_progress is not None:
            campaign_progress.record_candidate_outcome(failed_filters)
        if not failed_filters:
            model_metrics = per_model_design_scores(stage_filters, protein_states, ensemble.model_predictions, prediction_state, binder, target)
            accepted_binders.append(ValidatedBinder(predictions, {**metrics, **interface_pdae_scores}, candidate_number, model_metrics))
            if len(accepted_binders) >= accepted_sequence_limit:
                break
    kept = sorted(accepted_binders, key=lambda candidate: -ranking_value(candidate.metrics, RANKING_METRIC))[:kept_sequence_count]
    if candidate_number < candidate_count and len(accepted_binders) < accepted_sequence_limit:
        print(exhausted_redesign_window(candidate_number, candidate_count), flush=True)
    print(redesigns_kept(tuple((candidate.candidate_number, ranking_value(candidate.metrics, RANKING_METRIC)) for candidate in kept), len(accepted_binders), candidate_count, RANKING_METRIC), flush=True)
    return kept
