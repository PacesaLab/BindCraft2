import functools
import inspect
import random
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from typing import Callable, NamedTuple
from jax import Array
from bindcraft.developability import EPITOPE_CORE_LENGTH, HYDROPHOBICITY, MHCPanel, mhc_panels, protease_panel
from bindcraft.protein import AMINO_ACIDS, ATOM_INDEX, BINDER_ALONE, Protein, ProteinStates, ResidueFlags, StructurePredictions, alignment_matrix_product, has_residue_flag, kabsch, real_residue_count, real_residue_mask, real_residue_weights, redesignable_residue_mask

class DesignLoss(NamedTuple):
    function: Callable[[ProteinStates, StructurePredictions], Array]
    weight: float
    required_states: frozenset[str]
    interface_mask: Array | None = None
REGISTERED_LOSSES: dict[str, Callable] = {}
LOSS_TARGET_WEIGHTING: dict[str, str] = {}
BINDER_ROLE_PARAMETERS = frozenset({'binder', 'chain'})
PADDING_ANCHOR_STIFFNESS = 1000.0

def loss(name: str, target_weighting: str='binder_only') -> Callable:
    def register(loss_function: Callable) -> Callable:
        REGISTERED_LOSSES[name] = loss_function
        LOSS_TARGET_WEIGHTING[name] = target_weighting
        return loss_function
    return register

@functools.lru_cache(maxsize=None)
def _bind_metric_parameters(loss_function: Callable, loss_parameters: tuple[tuple[str, object], ...]) -> Callable:
    return functools.partial(loss_function, **dict(loss_parameters))

def _bound_metric(metric_function: Callable, metric_parameters: dict) -> Callable:
    return _bind_metric_parameters(metric_function, tuple(sorted((name, _immutable_metric_parameter(value)) for name, value in metric_parameters.items())))

def _immutable_metric_parameter(value):
    return tuple(_immutable_metric_parameter(item) for item in value) if isinstance(value, (list, tuple)) else value

def resolve_prediction_state(predictions: StructurePredictions, prediction_state: str) -> str:
    return prediction_state if prediction_state in predictions else next((name for name in predictions if name != BINDER_ALONE), prediction_state)

def resolve_target_chain(protein_complex: dict, name: str, state: str) -> str:
    return name if name in protein_complex else f'{name}_{state}' if f'{name}_{state}' in protein_complex else name

def resolve_binder_role(metric_function: Callable, metric_parameters: dict, binder_chain: str) -> dict:
    parameters = inspect.signature(metric_function).parameters
    return {**metric_parameters, **{name: binder_chain for name in parameters.keys() & BINDER_ROLE_PARAMETERS if metric_parameters.get(name, parameters[name].default) == 'binder' != binder_chain}}

def bind_state_metric(metric_function: Callable, metric_settings: dict) -> tuple[Callable, frozenset[str]]:
    metric_parameters = dict(metric_settings.get('params', {}))
    if 'prediction_state' in metric_settings:
        metric_parameters['prediction_state'] = metric_settings['prediction_state']
    prediction_state_parameter = inspect.signature(metric_function).parameters.get('prediction_state')
    if prediction_state_parameter is None:
        metric_parameters.pop('prediction_state', None)
    prediction_state = metric_parameters.get('prediction_state', prediction_state_parameter.default if prediction_state_parameter is not None else ())
    required_states = () if 'prediction_state' not in metric_parameters and prediction_state == 'complex' else (prediction_state,) if isinstance(prediction_state, str) else prediction_state
    return _bound_metric(metric_function, metric_parameters), frozenset(required_states)

STATE_LOSS_PARAMETERS = ('prediction_state', 'reference_state', 'binder_shapes')

def renamed_loss_name(name: str, state_names: dict[str, str]) -> str:
    base, separator, state = name.partition('.')
    return f'{base}.{state_names[state]}' if separator and state in state_names else name

def _renamed_states(value, state_names: dict[str, str]):
    if isinstance(value, str):
        return state_names.get(value, value)
    return tuple(_renamed_states(item, state_names) for item in value) if isinstance(value, tuple) else value

def renamed_state_losses(losses: dict[str, DesignLoss], state_names: dict[str, str]) -> dict[str, DesignLoss]:
    renamed = {}
    for name, entry in losses.items():
        keywords = getattr(entry.function, 'keywords', None)
        function = entry.function if keywords is None else _bind_metric_parameters(entry.function.func, tuple(sorted((key, _renamed_states(value, state_names) if key in STATE_LOSS_PARAMETERS else value) for key, value in keywords.items())))
        renamed[renamed_loss_name(name, state_names)] = DesignLoss(function, entry.weight, frozenset(state_names.get(state, state) for state in entry.required_states), entry.interface_mask)
    return renamed

def resolve_loss_weight(weight, seed: int=0) -> float:
    if not isinstance(weight, (list, tuple)) or not weight:
        return float(weight or 0)
    values = tuple(float(value) for value in weight)
    if len(values) > 2:
        return random.Random(seed).choice(tuple(value for value in values if value) or (0.0,))
    drawn = random.Random(seed).uniform(min(values), max(values))
    return drawn or max(values, key=abs)

def sampled_loss_weights(settings: dict, seed: int=0) -> dict[str, float]:
    return {name: resolve_loss_weight(value, seed) for name, value in settings.items() if name.startswith('weights_') and isinstance(value, (list, tuple))}

PARATOPE_CONFORMATIONS = 'extended', 'folded_back'

def sampled_paratope_conformation(settings: dict, seed: int=0) -> str:
    conformations = settings.get('paratope_conformations') or ()
    named = (conformations,) if isinstance(conformations, str) else tuple(conformations)
    unknown = [name for name in named if name not in PARATOPE_CONFORMATIONS]
    if unknown:
        raise ValueError(f'unknown paratope_conformations {", ".join(unknown)}; a paratope loop is {" or ".join(PARATOPE_CONFORMATIONS)}')
    return random.Random(seed).choice(named) if named else ''

def build_losses(settings: dict, seed: int=0) -> dict[str, DesignLoss]:
    losses = {}
    folded_back = sampled_paratope_conformation(settings, seed) == 'folded_back'
    for name, loss_function in REGISTERED_LOSSES.items():
        weight = resolve_loss_weight(settings.get(f'weights_{name}'), seed)
        if name == 'binder_intra_coldspot' and folded_back:
            weight = -weight
        if weight:
            bound_loss, required_states = bind_state_metric(loss_function, settings.get('losses', {}).get(name, {}))
            losses[name] = DesignLoss(bound_loss, weight, required_states)
    return losses

def weighted_design_loss(losses: dict[str, DesignLoss], protein_states: ProteinStates, predictions: StructurePredictions) -> Array:
    frozen_interfaces = frozen_interface_arguments(losses)
    return sum((entry.weight * entry.function(protein_states, predictions, **frozen_interfaces.get(name, {})) for name, entry in losses.items() if entry.required_states <= set(protein_states)), jnp.asarray(0.0))

def globular_radius(residues: int) -> float:
    return 2.38 * residues ** 0.365

def _masked_mean(values: Array, mask: Array, eps: float=1e-08) -> Array:
    values, mask = jnp.asarray(values, dtype=jnp.float32), jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(jnp.where(mask > 0, values * mask, 0.0)) / (mask.sum() + eps)

def amino_acid_probabilities(sequence: Array) -> Array:
    probabilities = sequence.astype(jnp.float32)
    return jnp.where((probabilities.min(-1, keepdims=True) >= 0) & jnp.isclose(probabilities.sum(-1, keepdims=True), 1.0), probabilities, jax.nn.softmax(probabilities))

def chain_residue_slices(protein_complex: dict[str, Protein]) -> dict[str, slice]:
    chain_slices, residue_start = {}, 0
    for chain_name in sorted(protein_complex):
        chain_slices[chain_name] = slice(residue_start, residue_start + len(protein_complex[chain_name]))
        residue_start += len(protein_complex[chain_name])
    return chain_slices

def binder_copy_chains(protein_complex: dict[str, Protein], binder: str) -> tuple[str, ...]:
    prefix = binder.rsplit('_', 1)[0] if binder.rsplit('_', 1)[-1].isdigit() else binder
    return tuple(name for name in sorted(protein_complex) if name.startswith(f'{prefix}_') and name.rsplit('_', 1)[-1].isdigit()) or (binder,)

def binder_protomer_groups(protein_complex: dict[str, Protein], binder: str, per_protomer: bool) -> tuple[tuple[str, ...], ...]:
    binder_chains = binder_copy_chains(protein_complex, binder)
    return tuple((name,) for name in binder_chains) if per_protomer else (binder_chains,)

def chain_group_rows(protein_complex: dict[str, Protein], chain_names: tuple[str, ...]) -> Array:
    chain_slices = chain_residue_slices(protein_complex)
    return jnp.concatenate([jnp.arange(chain_slices[name].start, chain_slices[name].stop) for name in chain_names])

def chain_group_residue_weights(protein_complex: dict[str, Protein], chain_names: tuple[str, ...]) -> Array:
    return jnp.concatenate([real_residue_weights(protein_complex[name].flags) for name in chain_names])

def complex_residue_weights(protein_complex: dict[str, Protein]) -> Array:
    return jnp.concatenate([real_residue_weights(protein_complex[name].flags) for name in sorted(protein_complex)])

def binder_domain_membership(protein: Protein, domain_ids: tuple[int, ...], domain_count: int) -> Array:
    domains = jnp.pad(jnp.asarray(domain_ids, dtype=jnp.int32), (0, max(len(protein) - len(domain_ids), 0)))[:len(protein)]
    return jax.nn.one_hot(domains, domain_count) * real_residue_weights(protein.flags)[:, None]

def helical_pair_mask(protein: Protein) -> Array:
    residue_mask = real_residue_mask(protein.flags)
    return (binder_sequence_offsets(protein) == 3) & residue_mask[:, None] & residue_mask[None, :]

def chain_residue_mask(residue_count: int, chain_slice: slice) -> Array:
    residue_indices = jnp.arange(residue_count)
    return (residue_indices >= chain_slice.start) & (residue_indices < chain_slice.stop)

def expand_chain_residue_mask(residue_count: int, chain_slice: slice, chain_values: Array, dtype=bool) -> Array:
    return jnp.zeros(residue_count, dtype=dtype).at[chain_slice].set(chain_values)

def pairwise_atom_distances(first: Array, second: Array, eps: float=0.0001) -> Array:
    return jnp.sqrt(jnp.sum((first.astype(jnp.float32)[:, None] - second.astype(jnp.float32)[None, :]) ** 2, axis=-1) + eps)

def chain_atom_coordinates(protein: Protein, atom: str='CA') -> tuple[Array, Array]:
    return protein.atoms[:, ATOM_INDEX[atom]].astype(jnp.float32), protein.atom_mask[:, ATOM_INDEX[atom]].astype(jnp.float32)

def masked_coordinate_centroid(coordinates: Array, mask: Array) -> Array:
    return (coordinates * mask[:, None]).sum(0) / (mask.sum() + 1e-08)

def binder_sequence_offsets(protein: Protein) -> Array:
    offsets = protein.residue_index[:, None] - protein.residue_index[None, :]
    cyclic_distance = jnp.minimum(jnp.abs(offsets), real_residue_count(protein.flags) - jnp.abs(offsets))
    cyclic_offsets = jnp.where(cyclic_distance < jnp.abs(offsets), -cyclic_distance, cyclic_distance) * jnp.sign(offsets)
    cyclic_residues = has_residue_flag(protein.flags, ResidueFlags.CYCLIC) & real_residue_mask(protein.flags)
    return jnp.where(cyclic_residues[:, None] & cyclic_residues[None, :], cyclic_offsets, offsets)

@loss('plddt_loss')
def plddt_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str='binder') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    confidence = predictions[prediction_state].metrics['plddt'][chain_group_rows(protein_complex, binder_chains)]
    return _masked_mean(1 - confidence, jnp.concatenate([has_residue_flag(protein_complex[name].flags, ResidueFlags.DESIGN) for name in binder_chains]))

@loss('target_plddt', target_weighting='every_target')
def target_plddt_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    hotspot_mask = has_residue_flag(protein_states[prediction_state][target].flags, ResidueFlags.HOTSPOT)
    return _masked_mean(1 - predictions[prediction_state].metrics['plddt'][chain_residue_slices(protein_states[prediction_state])[target]], jnp.where(hotspot_mask.any(), hotspot_mask, real_residue_mask(protein_states[prediction_state][target].flags)))

@loss('experimentally_resolved')
def experimentally_resolved_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    resolved = predictions[prediction_state].metrics['experimentally_resolved_ca'][chain_group_rows(protein_complex, binder_chains)]
    return _masked_mean(1 - resolved, jnp.concatenate([has_residue_flag(protein_complex[name].flags, ResidueFlags.DESIGN) for name in binder_chains]))

@loss('sequence_entropy')
def sequence_entropy_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder') -> Array:
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    binder_chains = binder_copy_chains(protein_complex, chain)
    logits = jnp.concatenate([protein_complex[name].sequence for name in binder_chains]).astype(jnp.float32)
    return _masked_mean(-(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1), jnp.concatenate([has_residue_flag(protein_complex[name].flags, ResidueFlags.DESIGN) for name in binder_chains]))

DISTOGRAM_DEPENDENT_LOSSES = ('distogram_cce', 'binder_coldspot', 'binder_intra_hotspot', 'binder_intra_coldspot', 'coldspot_repel', 'interface_contacts', 'non_contact', 'binder_contacts', 'binder_helicity', 'non_helical', 'multidomain')

def chain_pair_pae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, aligned_chains: tuple[str, ...] | None, reference_chains: tuple[str, ...] | None) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    pae = predictions[prediction_state].metrics['pae'] / 31.0
    protein_complex = protein_states[prediction_state]
    if aligned_chains is None and reference_chains is None:
        weights = complex_residue_weights(protein_complex)
        return _masked_mean(pae, weights[:, None] * weights[None, :])
    aligned_rows, reference_rows = chain_group_rows(protein_complex, aligned_chains), chain_group_rows(protein_complex, reference_chains)
    aligned_weights, reference_weights = chain_group_residue_weights(protein_complex, aligned_chains), chain_group_residue_weights(protein_complex, reference_chains)
    return _masked_mean(pae[aligned_rows[:, None], reference_rows[None, :]], aligned_weights[:, None] * reference_weights[None, :])

def soft_maximum(values: Array, temperature: float) -> Array:
    return (values * jax.nn.softmax(values / temperature)).sum()

def mhc_epitope_score(sequence: Array, panel: MHCPanel, coupling_weight: float, hydrophobicity_weight: float, temperature: float, residue_mask: Array | None=None) -> Array:
    if len(panel.anchor_matrices) == 0 or len(sequence) < EPITOPE_CORE_LENGTH:
        return jnp.asarray(0.0)
    anchor_matrices, coupling_matrices = jnp.asarray(panel.anchor_matrices), jnp.asarray(panel.coupling_matrices)
    hydrophobicity, groove_burial = jnp.asarray(HYDROPHOBICITY), jnp.asarray(panel.groove_burial)
    def epitope_core_score(core_start: Array) -> Array:
        core = jax.lax.dynamic_slice(sequence, (core_start, 0), (EPITOPE_CORE_LENGTH, len(AMINO_ACIDS)))
        anchor_match = (anchor_matrices * core).sum((-2, -1))
        anchor_coupling = sum((jnp.einsum('i,aij,j->a', core[first], coupling_matrices[:, pair], core[second]) for pair, (first, second, _) in enumerate(panel.coupled_anchors)))
        groove_burial_match = ((core @ hydrophobicity) * groove_burial).sum()
        return soft_maximum(anchor_match + coupling_weight * anchor_coupling + hydrophobicity_weight * groove_burial_match, temperature)
    core_starts = jnp.arange(len(sequence) - EPITOPE_CORE_LENGTH + 1)
    core_scores = jax.vmap(epitope_core_score)(core_starts)
    if residue_mask is not None:
        core_is_real = jax.vmap(lambda core_start: jax.lax.dynamic_slice(residue_mask.astype(jnp.float32), (core_start,), (EPITOPE_CORE_LENGTH,)).min())(core_starts) > 0
        core_scores = jnp.where(core_is_real, core_scores, -1000000000.0)
    return soft_maximum(core_scores, temperature) / EPITOPE_CORE_LENGTH

@loss('humanization')
def humanization_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', species: str='human', coupling_weight: float=0.5, hydrophobicity_weight: float=0.0, mhc_class_ii_weight: float=1.0, temperature: float=0.1) -> Array:
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    binder_chains = binder_copy_chains(protein_complex, chain)
    worst_epitope = lambda panel: soft_maximum(jnp.stack([mhc_epitope_score(amino_acid_probabilities(protein_complex[name].sequence), panel, coupling_weight, hydrophobicity_weight, temperature, redesignable_residue_mask(protein_complex[name].flags)) for name in binder_chains]), temperature)
    mhc_class_i_panel, mhc_class_ii_panel = mhc_panels(species)
    return worst_epitope(mhc_class_i_panel) + mhc_class_ii_weight * worst_epitope(mhc_class_ii_panel)

def protease_site_score(sequence: Array, residue_mask: Array) -> Array:
    panel = protease_panel()
    p1, p1_block, weights = jnp.asarray(panel.p1), jnp.asarray(panel.p1_block), jnp.asarray(panel.weights)
    p1_probability, block_probability = sequence @ p1.T, sequence @ p1_block.T
    p1_prime_is_real = jnp.concatenate([residue_mask[1:], jnp.zeros((1,))])[:, None]
    block_at_p1_prime = jnp.where(p1_prime_is_real, jnp.concatenate([block_probability[1:], jnp.zeros((1, block_probability.shape[1]))]), 1.0)
    return _masked_mean((p1_probability * (1.0 - block_at_p1_prime)) @ weights, residue_mask)

def pooled_protease_site_score(chains) -> Array:
    scored = [(protease_site_score(sequence, residue_mask), residue_mask.sum()) for sequence, residue_mask in chains]
    rows = sum(count for _score, count in scored)
    return jnp.where(rows > 0, sum(score * count for score, count in scored) / jnp.maximum(rows, 1), 0.0)

@loss('protease_sites')
def protease_site_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder') -> Array:
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    return pooled_protease_site_score([(jax.nn.softmax(protein_complex[name].sequence.astype(jnp.float32)), redesignable_residue_mask(protein_complex[name].flags)) for name in binder_copy_chains(protein_complex, chain)])

def binder_residue_exposure(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, chain: str, radius: float=10.0, contact_temperature: float=2.0, neighbor_threshold: float=16.0, temperature: float=4.0) -> Array:
    predicted_complex = predictions[prediction_state].protein_complex
    complex_coordinates = jnp.concatenate([chain_atom_coordinates(predicted_complex[name])[0] for name in sorted(predicted_complex)])
    complex_mask = complex_residue_weights(protein_states[prediction_state])
    binder_coordinates, _ = chain_atom_coordinates(predicted_complex[chain])
    neighbor_count = (jax.nn.sigmoid((radius - pairwise_atom_distances(binder_coordinates, complex_coordinates)) / contact_temperature) * complex_mask[None, :]).sum(-1) - 1.0
    return jax.nn.sigmoid((neighbor_threshold - neighbor_count) / temperature)

def binder_ca_pseudo_dihedral(coordinates: Array, eps: float=1e-08) -> Array:
    b1, b2, b3 = coordinates[1:-2] - coordinates[:-3], coordinates[2:-1] - coordinates[1:-2], coordinates[3:] - coordinates[2:-1]
    b2_hat = b2 / (jnp.linalg.norm(b2, axis=-1, keepdims=True) + eps)
    v = b1 - jnp.sum(b1 * b2_hat, axis=-1, keepdims=True) * b2_hat
    w = b3 - jnp.sum(b3 * b2_hat, axis=-1, keepdims=True) * b2_hat
    return jnp.arctan2(jnp.sum(jnp.cross(b2_hat, v) * w, axis=-1), jnp.sum(v * w, axis=-1))

def binder_strand_propensity(coordinates: Array, residue_weights: Array, extension_cutoff: float=0.3, temperature: float=0.2, smoothing: int=4) -> Array:
    length = coordinates.shape[0]
    if length < 4:
        return jnp.zeros(length)
    extension = -jnp.cos(binder_ca_pseudo_dihedral(coordinates))
    valid_window = residue_weights[:-3] * residue_weights[3:]
    strand_probability = jax.nn.sigmoid((extension - extension_cutoff) / temperature) * valid_window
    smoothed = jnp.convolve(jnp.concatenate([strand_probability, jnp.zeros(3)]), jnp.ones(smoothing) / smoothing, mode='same')
    return jnp.clip(smoothed, 0.0, 1.0)

def binder_loop_geometry_propensity(coordinates: Array, residue_weights: Array, cutoff: float=7.0, temperature: float=0.8, smoothing: int=4, distinguish_sheets: bool=False, strand_extension_cutoff: float=0.3, strand_temperature: float=0.2) -> Array:
    length = coordinates.shape[0]
    if length < 4:
        return jnp.ones(length)
    valid_window = residue_weights[:-3] * residue_weights[3:]
    close_probability = jax.nn.sigmoid((cutoff - jnp.sqrt(jnp.sum(jnp.square(coordinates[:-3] - coordinates[3:]), -1) + 1e-08)) / temperature) * valid_window
    structured = jnp.clip(jnp.convolve(jnp.concatenate([close_probability, jnp.zeros(3)]), jnp.ones(smoothing) / smoothing, mode='same'), 0.0, 1.0)
    if distinguish_sheets:
        structured = jnp.maximum(structured, binder_strand_propensity(coordinates, residue_weights, strand_extension_cutoff, strand_temperature, smoothing))
    return 1.0 - structured

def chain_loop_susceptibility(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, chain: str, measure: str, plddt_threshold: float, plddt_temperature: float, window: int, distinguish_sheets: bool) -> Array:
    protein = protein_states[prediction_state][chain]
    coordinates, _ = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    residue_weights = real_residue_weights(protein.flags)
    exposure = binder_residue_exposure(protein_states, predictions, prediction_state, chain)
    if measure == 'geometry':
        loopiness = binder_loop_geometry_propensity(coordinates, residue_weights, distinguish_sheets=distinguish_sheets)
    else:
        plddt_loopiness = jax.nn.sigmoid((plddt_threshold - predictions[prediction_state].metrics['plddt'][chain_residue_slices(protein_states[prediction_state])[chain]]) / plddt_temperature)
        loopiness = jnp.maximum(plddt_loopiness, binder_loop_geometry_propensity(coordinates, residue_weights, distinguish_sheets=distinguish_sheets)) if measure == 'both' else plddt_loopiness
    susceptibility = exposure * loopiness * residue_weights
    return susceptibility * jnp.convolve(susceptibility, jnp.ones(window) / window, mode='same')

@loss('exposed_loops')
def exposed_loop_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', measure: str='plddt', plddt_threshold: float=0.85, plddt_temperature: float=0.05, window: int=3, distinguish_sheets: bool=True) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    binder_chains = binder_copy_chains(protein_states[prediction_state], chain)
    susceptibility = [chain_loop_susceptibility(protein_states, predictions, prediction_state, name, measure, plddt_threshold, plddt_temperature, window, distinguish_sheets) for name in binder_chains]
    return _masked_mean(jnp.concatenate(susceptibility), jnp.concatenate([real_residue_mask(protein_states[prediction_state][name].flags) for name in binder_chains]))

def chain_terminal_rows(flags: Array, terminus_length: int) -> Array:
    c_terminus = real_residue_count(flags) - 1
    return jnp.concatenate([jnp.clip(jnp.arange(terminus_length), 0, c_terminus), jnp.clip(c_terminus - jnp.arange(terminus_length), 0, None)])

@loss('exposed_termini')
def exposed_termini_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', terminus_length: int=3) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    exposure, weights = [], []
    for name in binder_copy_chains(protein_states[prediction_state], chain):
        flags = protein_states[prediction_state][name].flags
        terminal_rows = chain_terminal_rows(flags, terminus_length)
        exposure.append(binder_residue_exposure(protein_states, predictions, prediction_state, name)[terminal_rows])
        weights.append(real_residue_weights(flags)[terminal_rows])
    return _masked_mean(jnp.concatenate(exposure), jnp.concatenate(weights))

@loss('binder_pae')
def binder_pae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str | None=None, domain_ids: tuple[int, ...]=(), per_protomer: bool=False) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if domain_ids:
        chain_slice = chain_residue_slices(protein_states[prediction_state])[chain or 'binder']
        binder_pae = predictions[prediction_state].metrics['pae'][chain_slice, chain_slice] / 31.0
        domain_membership = binder_domain_membership(protein_states[prediction_state][chain or 'binder'], domain_ids, max(domain_ids) + 1)
        return _masked_mean(binder_pae, domain_membership @ domain_membership.T)
    if not chain:
        return chain_pair_pae_loss(protein_states, predictions, prediction_state, None, None)
    protomer_groups = binder_protomer_groups(protein_states[prediction_state], chain, per_protomer)
    return sum((chain_pair_pae_loss(protein_states, predictions, prediction_state, group, group) for group in protomer_groups)) / len(protomer_groups)

@loss('interface_pae', target_weighting='binds_target')
def interface_pae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return chain_pair_pae_loss(protein_states, predictions, prediction_state, (binder,), (resolve_target_chain(protein_states[prediction_state], target, prediction_state),))

@loss('compactness')
def radius_of_gyration_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str | None=None, eps: float=1e-08, per_protomer: bool=False) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    designed_chain_groups = binder_protomer_groups(protein_states[prediction_state], chain, per_protomer) if chain else [(name,) for name in sorted(protein_states[prediction_state])]
    radius_loss_sum, designed_chain_count = jnp.asarray(0.0), jnp.asarray(eps)
    for group in designed_chain_groups:
        proteins = [predictions[prediction_state].protein_complex[name] for name in group]
        coordinates = jnp.concatenate([chain_atom_coordinates(protein)[0] for protein in proteins])
        resolved_ca_mask = jnp.concatenate([chain_atom_coordinates(protein)[1] for protein in proteins])
        radius = jnp.sqrt(_masked_mean(jnp.square(coordinates - masked_coordinate_centroid(coordinates, resolved_ca_mask)).sum(-1), resolved_ca_mask) + eps)
        designed_chain_weight = jnp.asarray(1.0) if chain else has_residue_flag(protein_states[prediction_state][group[0]].flags, ResidueFlags.DESIGN).any().astype(jnp.float32)
        radius_loss_sum, designed_chain_count = radius_loss_sum + jax.nn.elu(radius - globular_radius(sum((real_residue_count(protein.flags) for protein in proteins)))) * designed_chain_weight, designed_chain_count + designed_chain_weight
    return radius_loss_sum / designed_chain_count

@loss('iptm_loss', target_weighting='binds_target')
def iptm_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> Array:
    return 1 - predictions[resolve_prediction_state(predictions, prediction_state)].metrics['iptm']

@loss('ptm_loss', target_weighting='every_target')
def ptm_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> Array:
    return 1 - predictions[resolve_prediction_state(predictions, prediction_state)].metrics['ptm']

def receptor_chain_index(protein: Protein) -> Array:
    from bindcraft.protein_preparation import RECEPTOR_CHAIN_BREAK_GAP
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(jnp.diff(protein.residue_index) >= RECEPTOR_CHAIN_BREAK_GAP)])

def template_distance_deviation(template: Protein, predicted: Protein, selected_pairs: Array) -> Array:
    template_coordinates, template_mask = chain_atom_coordinates(template)
    predicted_coordinates, predicted_mask = chain_atom_coordinates(predicted)
    valid_mask = template_mask * predicted_mask
    pair_mask = selected_pairs * valid_mask[:, None] * valid_mask[None, :]
    distance_error = pairwise_atom_distances(predicted_coordinates, predicted_coordinates, 1e-08) - pairwise_atom_distances(template_coordinates, template_coordinates, 1e-08)
    return jnp.sqrt(_masked_mean(jnp.square(distance_error), pair_mask) + 1e-08) * (pair_mask.sum() > 0)

@loss('target_rmsd', target_weighting='every_target')
def target_rmsd_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    return template_distance_deviation(protein_states[prediction_state][target], predictions[prediction_state].protein_complex[target], jnp.asarray(1.0))

@loss('target_rigidity', target_weighting='every_target')
def target_rigidity_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    template = protein_states[prediction_state][target]
    chain_index = receptor_chain_index(template)
    return template_distance_deviation(template, predictions[prediction_state].protein_complex[target], (chain_index[:, None] != chain_index[None, :]).astype(jnp.float32))

def best_contact_mean(values: Array, contact_count: int | float, mask: Array, eps: float=0.0001) -> Array:
    contact_ranking = jax.lax.stop_gradient(jnp.argsort(jnp.where(mask, values, jnp.inf)))
    ranked_values = jnp.take_along_axis(values, contact_ranking, axis=-1)
    selected = (jnp.arange(values.shape[-1]) < contact_count) & jnp.take_along_axis(mask, contact_ranking, axis=-1)
    return jnp.where(selected, ranked_values, 0).sum(-1) / (selected.sum(-1) + eps)

def distogram_bin_distances(num_bins: int) -> Array:
    return jnp.append(0.0, jnp.linspace(2.3125, 21.6875, num_bins - 1))

def distogram_pair_loss(distogram: Array, selected_bins: Array, binary: bool) -> Array:
    distogram = distogram.astype(jnp.float32)
    if binary:
        return -jnp.log((selected_bins * jax.nn.softmax(distogram)).sum(-1) + 1e-08)
    selected_probabilities = jax.nn.softmax(distogram - 10000000.0 * (1 - selected_bins))
    return -(selected_probabilities * jax.nn.log_softmax(distogram)).sum(-1)

def mean_selected_contact_loss(pair_loss: Array, contacts_per_residue: int | float, contact_residue_count: int | float, residue_mask: Array, contact_pair_mask: Array) -> Array:
    return best_contact_mean(best_contact_mean(pair_loss, contacts_per_residue, contact_pair_mask), contact_residue_count, residue_mask)

def distogram_contact_loss(distogram: Array, cutoff: float, contacts_per_residue: int | float, contact_residue_count: int | float, residue_mask: Array, contact_pair_mask: Array, binary: bool=False) -> Array:
    return mean_selected_contact_loss(distogram_pair_loss(distogram, distogram_bin_distances(distogram.shape[-1]) < cutoff, binary), contacts_per_residue, contact_residue_count, residue_mask, contact_pair_mask)

def pseudo_beta_coordinates(protein: Protein) -> tuple[Array, Array]:
    beta_coordinates, beta_mask = chain_atom_coordinates(protein, 'CB')
    alpha_coordinates, alpha_mask = chain_atom_coordinates(protein, 'CA')
    return jnp.where(beta_mask[:, None] > 0, beta_coordinates, alpha_coordinates), jnp.where(beta_mask > 0, beta_mask, alpha_mask)

def nonbinding_residue_repulsion_loss(distogram: Array, residue_mask: Array, partner_residue_mask: Array, cutoff: float, contacts_per_residue: int, eligible_pair_mask: Array | None=None) -> Array:
    pair_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram.shape[-1]) < cutoff, False)
    contact_pair_mask = residue_mask[:, None] & partner_residue_mask[None, :]
    contact_pair_mask = contact_pair_mask if eligible_pair_mask is None else contact_pair_mask & eligible_pair_mask
    contact_loss = mean_selected_contact_loss(pair_loss, contacts_per_residue, float('inf'), residue_mask, contact_pair_mask)
    return jnp.where(contact_pair_mask.any(), -contact_loss, 0.0)

@loss('distogram_cce', target_weighting='every_target')
def distogram_cce_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='target') -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    chain = resolve_target_chain(protein_states[prediction_state], chain, prediction_state)
    chain_slice = chain_residue_slices(protein_states[prediction_state])[chain]
    reference_coordinates, reference_mask = pseudo_beta_coordinates(protein_states[prediction_state][chain])
    distogram = predictions[prediction_state].metrics['distogram'][chain_slice, chain_slice].astype(jnp.float32)
    reference_bins = (pairwise_atom_distances(reference_coordinates, reference_coordinates)[..., None] > distogram_bin_distances(distogram.shape[-1])[1:]).sum(-1)
    cross_entropy = -(jax.nn.one_hot(reference_bins, distogram.shape[-1]) * jax.nn.log_softmax(distogram)).sum(-1)
    return _masked_mean(cross_entropy, reference_mask[:, None] * reference_mask[None, :])

@loss('com_distance', target_weighting='every_target')
def com_distance_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', radius_ratio: float=0.9, eps: float=1e-08) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    predicted_complex = predictions[prediction_state].protein_complex
    binder_coordinates, binder_mask = chain_atom_coordinates(predicted_complex[binder])
    target_coordinates, target_mask = chain_atom_coordinates(predicted_complex[target])
    binder_centroid = masked_coordinate_centroid(binder_coordinates, binder_mask)
    binder_radius = jnp.sqrt(_masked_mean(jnp.square(binder_coordinates - binder_centroid).sum(-1), binder_mask) + eps)
    centroid_distance = jnp.linalg.norm(binder_centroid - masked_coordinate_centroid(target_coordinates, target_mask))
    return jax.nn.relu(centroid_distance - radius_ratio * binder_radius)

@loss('binder_coldspot', target_weighting='every_target')
def binder_coldspot_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=8.0, contacts_per_residue: int=1, designed_only: bool=True) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    protein_complex = protein_states[prediction_state]
    distogram = predictions[prediction_state].metrics['distogram']
    chain_slices, residue_count = chain_residue_slices(protein_complex), distogram.shape[0]
    framework_mask = binder_framework_mask(protein_complex, binder, residue_count, designed_only)
    target_mask = chain_residue_mask(residue_count, chain_slices[target]) & (complex_residue_weights(protein_complex) > 0)
    return nonbinding_residue_repulsion_loss(distogram, target_mask, framework_mask, cutoff, contacts_per_residue)

def binder_distant_pair_mask(protein_complex: dict[str, Protein], binder: str, residue_count: int, sequence_separation: int) -> Array:
    chain_slices = chain_residue_slices(protein_complex)
    distant_pairs = jnp.ones((residue_count, residue_count), dtype=bool)
    for name in binder_copy_chains(protein_complex, binder):
        chain_slice = chain_slices[name]
        distant_pairs = distant_pairs.at[chain_slice, chain_slice].set(jnp.abs(binder_sequence_offsets(protein_complex[name])) >= sequence_separation)
    return distant_pairs

def binder_intra_contact(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, binder: str, cutoff: float, contacts_per_residue: int, sequence_separation: int) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    distogram = predictions[prediction_state].metrics['distogram']
    residue_count = distogram.shape[0]
    framework_mask = binder_framework_mask(protein_complex, binder, residue_count)
    paratope_mask = binder_binding_mask(protein_complex, binder, residue_count)
    distant_pairs = binder_distant_pair_mask(protein_complex, binder, residue_count, sequence_separation)
    return nonbinding_residue_repulsion_loss(distogram, framework_mask, paratope_mask, cutoff, contacts_per_residue, distant_pairs)

@loss('binder_intra_coldspot')
def binder_intra_coldspot_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', cutoff: float=8.0, contacts_per_residue: int=1, sequence_separation: int=20) -> Array:
    return binder_intra_contact(protein_states, predictions, prediction_state, binder, cutoff, contacts_per_residue, sequence_separation)

@loss('binder_intra_hotspot')
def binder_intra_hotspot_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', cutoff: float=8.0, contacts_per_residue: int=1, sequence_separation: int=20) -> Array:
    return -binder_intra_contact(protein_states, predictions, prediction_state, binder, cutoff, contacts_per_residue, sequence_separation)

@loss('coldspot_repel', target_weighting='every_target')
def coldspot_repel_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=8.0, contacts_per_residue: int=1) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    protein_complex = protein_states[prediction_state]
    distogram = predictions[prediction_state].metrics['distogram']
    chain_slices, residue_count = chain_residue_slices(protein_complex), distogram.shape[0]
    coldspot_mask = expand_chain_residue_mask(residue_count, chain_slices[target], has_residue_flag(protein_complex[target].flags, ResidueFlags.COLDSPOT))
    binder_mask = binder_binding_mask(protein_complex, binder, residue_count)
    return nonbinding_residue_repulsion_loss(distogram, coldspot_mask, binder_mask, cutoff, contacts_per_residue)

def binder_binding_mask(protein_complex: dict[str, Protein], binder: str, residue_count: int) -> Array:
    declared = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.CONTACT, residue_count)
    return jnp.where(declared.any(), declared, binder_chain_mask(protein_complex, binder, residue_count)) & (complex_residue_weights(protein_complex) > 0)

def binder_chain_mask(protein_complex: dict[str, Protein], binder: str, residue_count: int) -> Array:
    chain_slices = chain_residue_slices(protein_complex)
    mask = jnp.zeros(residue_count, dtype=bool)
    for name in binder_copy_chains(protein_complex, binder):
        mask = mask | chain_residue_mask(residue_count, chain_slices[name])
    return mask

def binder_chain_flag_mask(protein_complex: dict[str, Protein], binder: str, flag: ResidueFlags, residue_count: int) -> Array:
    chain_slices = chain_residue_slices(protein_complex)
    mask = jnp.zeros(residue_count, dtype=bool)
    for name in binder_copy_chains(protein_complex, binder):
        mask = mask | expand_chain_residue_mask(residue_count, chain_slices[name], has_residue_flag(protein_complex[name].flags, flag))
    return mask

def binder_framework_mask(protein_complex: dict[str, Protein], binder: str, residue_count: int, designed_only: bool=True) -> Array:
    declared = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.CONTACT, residue_count)
    occupied = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.DESIGN, residue_count) if designed_only else binder_chain_mask(protein_complex, binder, residue_count)
    return occupied & ~declared & declared.any() & (complex_residue_weights(protein_complex) > 0)

@loss('interface_contacts', target_weighting='binds_target')
def interface_contacts_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=21.6875, contacts_per_residue: int=1, contact_residue_count: int | float=float('inf')) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    distogram = predictions[prediction_state].metrics['distogram']
    chain_slices, residue_count = chain_residue_slices(protein_states[prediction_state]), distogram.shape[0]
    real_mask = complex_residue_weights(protein_states[prediction_state]) > 0
    binder_mask, target_mask = binder_binding_mask(protein_states[prediction_state], binder, residue_count), chain_residue_mask(residue_count, chain_slices[target]) & real_mask
    hotspot_mask = expand_chain_residue_mask(residue_count, chain_slices[target], has_residue_flag(protein_states[prediction_state][target].flags, ResidueFlags.HOTSPOT))
    pair_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram.shape[-1]) < cutoff, False)
    hotspot_contacts = mean_selected_contact_loss(pair_loss, contacts_per_residue, contact_residue_count, hotspot_mask, hotspot_mask[:, None] & binder_mask[None, :])
    surface_contacts = mean_selected_contact_loss(pair_loss, contacts_per_residue, contact_residue_count, binder_mask, binder_mask[:, None] & target_mask[None, :])
    return jnp.where(hotspot_mask.any(), hotspot_contacts, surface_contacts)

@loss('non_contact', target_weighting='avoids_target')
def non_contact_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=14.0, contacts_per_residue: int=2) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    chain_slices = chain_residue_slices(protein_states[prediction_state])
    distogram = predictions[prediction_state].metrics['distogram'][chain_slices[target], chain_slices[binder]]
    pair_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram.shape[-1]) >= cutoff, False)
    target_mask, binder_mask = real_residue_mask(protein_states[prediction_state][target].flags), real_residue_mask(protein_states[prediction_state][binder].flags)
    return -_masked_mean(best_contact_mean(-pair_loss, contacts_per_residue, target_mask[:, None] & binder_mask[None, :]), target_mask)

@loss('binder_contacts')
def binder_contacts_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str='binder', cutoff: float=14.0, contacts_per_residue: int=2, contact_residue_count: int | float=float('inf'), sequence_separation: int | None=9, per_protomer: bool=False) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    binder_rows = chain_group_rows(protein_complex, binder_chains)
    distogram = predictions[prediction_state].metrics['distogram'][binder_rows[:, None], binder_rows[None, :]]
    residue_mask = jnp.concatenate([real_residue_mask(protein_complex[name].flags) for name in binder_chains])
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    if sequence_separation is not None:
        pair_mask = pair_mask & jnp.logical_not(block_diag(*[jnp.abs(binder_sequence_offsets(protein_complex[name])) < sequence_separation for name in binder_chains]))
    if per_protomer:
        pair_mask = pair_mask & (block_diag(*[jnp.ones((len(protein_complex[name]),) * 2) for name in binder_chains]) > 0)
    return distogram_contact_loss(distogram, cutoff, contacts_per_residue, contact_residue_count, residue_mask, pair_mask)

@loss('binder_helicity')
def binder_helicity_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str='binder', cutoff: float=6.0) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    binder_rows = chain_group_rows(protein_complex, binder_chains)
    distogram = predictions[prediction_state].metrics['distogram'][binder_rows[:, None], binder_rows[None, :]]
    return _masked_mean(distogram_pair_loss(distogram, distogram_bin_distances(distogram.shape[-1]) < cutoff, True), block_diag(*[helical_pair_mask(protein_complex[name]) for name in binder_chains]))

@loss('target_helicity', target_weighting='every_target')
def target_helicity_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target', cutoff: float=6.0) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return binder_helicity_loss(protein_states, predictions, prediction_state, resolve_target_chain(protein_states[prediction_state], target, prediction_state), cutoff)

@loss('non_helical')
def non_helical_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', cutoff: float=6.0) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    binder_rows = chain_group_rows(protein_complex, binder_chains)
    distogram = predictions[prediction_state].metrics['distogram'][binder_rows[:, None], binder_rows[None, :]].astype(jnp.float32)
    close_probability = (jax.nn.softmax(distogram) * (distogram_bin_distances(distogram.shape[-1]) < cutoff)).sum(-1)
    return _masked_mean(close_probability, block_diag(*[helical_pair_mask(protein_complex[name]) for name in binder_chains]))

@loss('termini_distance')
def termini_distance_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', threshold_distance: float=7.0) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    coordinates, mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    c_terminus = real_residue_count(protein_states[prediction_state][chain].flags) - 1
    distance = jnp.sqrt(jnp.square(coordinates[0] - coordinates[c_terminus]).sum() + 1e-08)
    return jax.nn.relu(distance - threshold_distance) * mask[0] * mask[c_terminus]

def terminus_target_direction_cosine(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, binder: str, target: str, terminus: str) -> tuple[Array, Array]:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    binder_coordinates, binder_mask = chain_atom_coordinates(protein_complex[binder])
    target_coordinates, target_mask = chain_atom_coordinates(protein_complex[resolve_target_chain(protein_complex, target, prediction_state)])
    binder_center = masked_coordinate_centroid(binder_coordinates, binder_mask)
    c_terminus = real_residue_count(protein_complex[binder].flags) - 1
    terminus_position = (binder_coordinates[0] + binder_coordinates[c_terminus]) / 2 if terminus == 'both' else binder_coordinates[0 if terminus == 'n' else c_terminus]
    away_vector, terminus_vector = binder_center - masked_coordinate_centroid(target_coordinates, target_mask), terminus_position - binder_center
    denominator = jnp.sqrt(jnp.square(away_vector).sum() + 1e-08) * jnp.sqrt(jnp.square(terminus_vector).sum() + 1e-08)
    valid = (binder_mask.sum() > 0) * (target_mask.sum() > 0) * (binder_mask[0] * binder_mask[c_terminus] if terminus == 'both' else binder_mask[0 if terminus == 'n' else c_terminus])
    return jnp.dot(away_vector, terminus_vector) / (denominator + 1e-08), valid

@loss('termini_angle', target_weighting='every_target')
def termini_angle_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target') -> Array:
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, prediction_state, binder, target, 'both')
    return jnp.arccos(jnp.clip(cosine, -1 + 1e-07, 1 - 1e-07)) * valid

@loss('n_terminus_away', target_weighting='every_target')
def n_terminus_away_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target') -> Array:
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, prediction_state, binder, target, 'n')
    return jax.nn.relu(-cosine) * valid

@loss('c_terminus_away', target_weighting='every_target')
def c_terminus_away_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target') -> Array:
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, prediction_state, binder, target, 'c')
    return jax.nn.relu(-cosine) * valid

def mutual_pair_choice(affinity: Array, temperature: float) -> Array:
    chosen = jax.nn.softmax(affinity / temperature, axis=-1)
    return chosen * chosen.T

@loss('disulfide')
def disulfide_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', distance: float=3.8, sigma: float=1.5, sequence_separation: int=3, temperature: float=0.1) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    sequence = jnp.concatenate([amino_acid_probabilities(protein_complex[name].sequence) for name in binder_chains])
    cysteine_probability = sequence[:, AMINO_ACIDS.index('C')]
    residue_mask = jnp.concatenate([real_residue_mask(protein_complex[name].flags) for name in binder_chains])
    hardness = _masked_mean(jnp.max(sequence, axis=-1), residue_mask)
    coordinates = jnp.concatenate([chain_atom_coordinates(predictions[prediction_state].protein_complex[name], 'CB')[0] for name in binder_chains])
    bond_quality = jnp.exp(-jnp.square((pairwise_atom_distances(coordinates, coordinates, 1e-08) - distance) / sigma))
    sequence_local = block_diag(*[jnp.abs(jnp.arange(len(protein_complex[name]))[:, None] - jnp.arange(len(protein_complex[name]))[None, :]) < sequence_separation for name in binder_chains])
    pairable = jnp.logical_not(sequence_local) * residue_mask[None, :] * residue_mask[:, None]
    partner_bond = bond_quality * jax.lax.stop_gradient(cysteine_probability)[None, :] * pairable
    satisfaction = (partner_bond * mutual_pair_choice(partner_bond, temperature)).sum(-1)
    return hardness * jnp.sum(cysteine_probability * jax.nn.relu(1 - satisfaction) * residue_mask)

def align_binder_coordinates(coordinates: Array, reference_coordinates: Array, valid_mask: Array) -> Array:
    rotation, center, reference_center = kabsch(coordinates, reference_coordinates, valid_mask)
    return alignment_matrix_product(coordinates - center, jax.lax.stop_gradient(rotation).T) + reference_center

def aligned_binder_tm_score(coordinates: Array, reference_coordinates: Array, valid_mask: Array, eps: float=1e-08) -> Array:
    squared_distances = jnp.square(align_binder_coordinates(coordinates, reference_coordinates, valid_mask) - reference_coordinates).sum(-1)
    distance_scale = jnp.maximum(1.24 * jnp.maximum(valid_mask.sum() - 15, eps) ** (1 / 3) - 1.8, 0.5)
    return _masked_mean(1 / (1 + (squared_distances + eps) / distance_scale ** 2), valid_mask)

def bound_and_unbound_binder_coordinates(predictions: StructurePredictions, prediction_state: str, reference_state: str, chain: str) -> tuple[Array, Array, Array]:
    coordinates, mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    reference_coordinates, reference_mask = chain_atom_coordinates(predictions[reference_state].protein_complex[chain])
    return coordinates, reference_coordinates, mask * reference_mask

@loss('induced_fit_global', target_weighting='every_target')
def induced_fit_global_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', reference_state: str=BINDER_ALONE, chain: str='binder', tm_target: float=0.6) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return jnp.asarray(0.0)
    coordinates, reference_coordinates, valid_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, chain)
    return jnp.square(jax.nn.relu(aligned_binder_tm_score(coordinates, reference_coordinates, valid_mask) - tm_target)) * (valid_mask.sum() >= 3)

def binder_interface_mask(protein_states: ProteinStates, predictions: StructurePredictions, chain: str='binder', target: str='target', cutoff: float=8.0) -> Array:
    prediction_state = resolve_prediction_state(predictions, 'complex')
    protein_complex = predictions[prediction_state].protein_complex
    coordinates, _ = chain_atom_coordinates(protein_complex[chain])
    target_coordinates, target_mask = chain_atom_coordinates(protein_complex[resolve_target_chain(protein_states[prediction_state], target, prediction_state)])
    facing_target = ((pairwise_atom_distances(coordinates, target_coordinates) < cutoff) * target_mask[None, :]).any(-1) & real_residue_mask(protein_states[prediction_state][chain].flags)
    return facing_target.astype(jnp.float32)

def interface_mask_residues(interface_mask: Array | None) -> tuple[int, ...]:
    return () if interface_mask is None else tuple(int(residue) for residue in jnp.flatnonzero(interface_mask))

def freeze_induced_fit_interface(losses: dict[str, DesignLoss], protein_states: ProteinStates, predictions: StructurePredictions) -> dict[str, DesignLoss]:
    frozen_losses = dict(losses)
    for name, entry in losses.items():
        if name.split('.')[0] != 'induced_fit_interface':
            continue
        chain = entry.function.keywords.get('chain', 'binder')
        interface_mask = binder_interface_mask(protein_states, predictions, chain, entry.function.keywords.get('target', 'target'), entry.function.keywords.get('cutoff', 8.0))
        interface_residue_count = int(interface_mask.sum())
        core_residues = int(real_residue_count(protein_states[resolve_prediction_state(predictions, 'complex')][chain].flags)) - interface_residue_count
        if interface_residue_count >= 3 and core_residues >= 3:
            frozen_losses[name] = DesignLoss(entry.function, entry.weight, entry.required_states, interface_mask)
    return frozen_losses

def frozen_interface_arguments(losses: dict[str, DesignLoss], padded_chains: dict[str, Protein] | None=None) -> dict[str, dict[str, Array]]:
    frozen = {name: entry.interface_mask for name, entry in losses.items() if entry.interface_mask is not None}
    if padded_chains is None:
        return {name: {'interface_mask': interface_mask} for name, interface_mask in frozen.items()}
    return {name: {'interface_mask': jnp.pad(interface_mask, [0, len(padded_chains[losses[name].function.keywords.get('chain', 'binder')]) - len(interface_mask)])} for name, interface_mask in frozen.items()}

def induced_fit_hinge_names(losses: dict[str, DesignLoss]) -> tuple[str, ...]:
    return tuple(name for name in losses if name.split('.')[0] in ('induced_fit_interface', 'induced_fit_global'))

BINDER_ALONE_LOSS_NAMES = ('plddt_loss', 'binder_pae', 'compactness', 'binder_contacts', 'binder_helicity')
PROTOMER_SCOPED_LOSSES = ('binder_pae', 'compactness', 'binder_contacts')

def induced_fit_binder_alone_losses(losses: dict[str, DesignLoss]) -> dict[str, DesignLoss]:
    binder_alone_losses = {}
    for name, entry in losses.items():
        if name.split('.')[0] in BINDER_ALONE_LOSS_NAMES:
            bound_loss, required_states = bind_state_metric(entry.function.func, {'params': {**entry.function.keywords, 'prediction_state': BINDER_ALONE}})
            binder_alone_losses[f'{name}.{BINDER_ALONE}'] = DesignLoss(bound_loss, entry.weight, required_states)
    return binder_alone_losses

def induced_fit_reference_losses(losses: dict[str, DesignLoss], reference_state: str, prediction_state: str | None=None) -> dict[str, DesignLoss]:
    reference_losses = dict(losses)
    for name in induced_fit_hinge_names(losses):
        entry = losses[name]
        state = prediction_state or entry.function.keywords.get('prediction_state', 'complex')
        parameters = {**entry.function.keywords, 'prediction_state': state, 'reference_state': reference_state}
        function = _bound_metric(entry.function.func, parameters)
        reference_losses[name] = DesignLoss(function, entry.weight, frozenset({state}), entry.interface_mask)
    return reference_losses

def binder_alone_design_losses(losses: dict[str, DesignLoss], reference_state: str) -> dict[str, DesignLoss]:
    losses = induced_fit_reference_losses(losses, reference_state, BINDER_ALONE)
    selected, hinge_names = {}, induced_fit_hinge_names(losses)
    for name, entry in losses.items():
        base_name = name.split('.')[0]
        if name in hinge_names:
            selected[name] = entry
        elif LOSS_TARGET_WEIGHTING.get(base_name) == 'binder_only':
            bound_loss, required_states = bind_state_metric(entry.function.func, {'params': entry.function.keywords, 'prediction_state': BINDER_ALONE})
            selected[base_name] = DesignLoss(bound_loss, entry.weight, required_states)
    return selected

def core_aligned_interface_rmsd(coordinates: Array, reference_coordinates: Array, interface_mask: Array, alignment_mask: Array, eps: float=1e-08) -> Array:
    aligned_coordinates = align_binder_coordinates(coordinates, reference_coordinates, alignment_mask)
    return jnp.sqrt(_masked_mean(jnp.square(aligned_coordinates - reference_coordinates).sum(-1), interface_mask) + eps)

def induced_fit_interface_masks(protein_states: ProteinStates, predictions: StructurePredictions, coordinates: Array, valid_mask: Array, prediction_state: str, target: str, cutoff: float, interface_residues: tuple[int, ...]=(), interface_mask: Array | None=None) -> tuple[Array, Array]:
    if interface_mask is None and interface_residues:
        interface_mask = jnp.zeros(len(coordinates), dtype=jnp.float32).at[jnp.asarray(interface_residues)].set(1.0)
    if interface_mask is None:
        target_coordinates, target_mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[resolve_target_chain(protein_states[prediction_state], target, prediction_state)])
        interface_mask = jax.lax.stop_gradient(((pairwise_atom_distances(coordinates, target_coordinates) < cutoff) * target_mask[None, :]).any(-1).astype(jnp.float32))
    interface_mask = interface_mask * valid_mask
    return interface_mask, (1 - interface_mask) * valid_mask

@loss('induced_fit_interface', target_weighting='every_target')
def induced_fit_interface_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', reference_state: str=BINDER_ALONE, chain: str='binder', target: str='target', interface_rmsd_target: float=3.0, cutoff: float=8.0, interface_mask: Array | None=None) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return jnp.asarray(0.0)
    coordinates, reference_coordinates, valid_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, chain)
    interface_mask, alignment_mask = induced_fit_interface_masks(protein_states, predictions, coordinates, valid_mask, prediction_state, target, cutoff, interface_mask=interface_mask)
    interface_rmsd = core_aligned_interface_rmsd(coordinates, reference_coordinates, interface_mask, alignment_mask)
    return jnp.square(jax.nn.relu(interface_rmsd_target - interface_rmsd)) * (interface_mask.sum() >= 3) * (alignment_mask.sum() >= 3)

@loss('fold_switching')
def fold_switching_loss(protein_states: ProteinStates, predictions: StructurePredictions, binder_shapes: tuple[tuple[str, ...], ...]=(), chain: str='binder', tm_target: float=0.6) -> Array:
    selected_states = [next((state for state in conformation_states if state in predictions), None) for conformation_states in binder_shapes]
    active_conformation_states = [state for state in selected_states if state is not None]
    return induced_fit_global_loss(protein_states, predictions, active_conformation_states[0], active_conformation_states[1], chain, tm_target) if len(active_conformation_states) >= 2 else jnp.asarray(0.0)

def elastic_network_covariance(coordinates: Array, residue_mask: Array, contact_decay: float, damping: float, eps: float=1e-08) -> Array:
    residue_count = coordinates.shape[0]
    offsets = coordinates[None, :, :] - coordinates[:, None, :]
    squared_distances = jnp.square(offsets).sum(-1)
    bond_directions = offsets / jnp.sqrt(squared_distances + eps)[..., None]
    spring_constants = jnp.where(jnp.eye(residue_count, dtype=bool), 0.0, jnp.exp(-squared_distances / contact_decay ** 2) * residue_mask[:, None] * residue_mask[None, :])
    coupling_blocks = -spring_constants[..., None, None] * bond_directions[..., :, None] * bond_directions[..., None, :]
    anchored_blocks = jnp.einsum('ijab->iab', -coupling_blocks) + (1 - residue_mask)[:, None, None] * PADDING_ANCHOR_STIFFNESS * jnp.eye(3)
    hessian = jnp.transpose(coupling_blocks.at[jnp.arange(residue_count), jnp.arange(residue_count)].set(anchored_blocks), (0, 2, 1, 3)).reshape(3 * residue_count, 3 * residue_count)
    centred_coordinates = (coordinates - masked_coordinate_centroid(coordinates, residue_mask)) * residue_mask[:, None]
    still = jnp.zeros_like(residue_mask)
    rigid_body_motions = ((residue_mask, still, still), (still, residue_mask, still), (still, still, residue_mask), (still, -centred_coordinates[:, 2], centred_coordinates[:, 1]), (centred_coordinates[:, 2], still, -centred_coordinates[:, 0]), (-centred_coordinates[:, 1], centred_coordinates[:, 0], still))
    rigid_body_basis, _ = jnp.linalg.qr(jnp.stack([jnp.stack(motion, -1).reshape(-1) for motion in rigid_body_motions], -1))
    internal_projector = jnp.eye(3 * residue_count) - rigid_body_basis @ rigid_body_basis.T
    internal_hessian = internal_projector @ hessian @ internal_projector
    mean_curvature = jnp.clip((jnp.diagonal(internal_hessian) * jnp.repeat(residue_mask, 3)).sum() / jnp.clip(3 * residue_mask.sum() - 6, 1.0, None), eps, None)
    return internal_projector @ jnp.linalg.inv(internal_hessian + damping * mean_curvature * jnp.eye(3 * residue_count)) @ internal_projector

@loss('collective_softness')
def collective_softness_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str='binder', contact_decay: float=8.0, damping: float=0.01, power_iterations: int=12, eps: float=1e-08) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    binder_chains = binder_copy_chains(protein_states[prediction_state], chain)
    chain_coordinates = [chain_atom_coordinates(predictions[prediction_state].protein_complex[name]) for name in binder_chains]
    coordinates, residue_mask = jnp.concatenate([coordinate for coordinate, _ in chain_coordinates]), jnp.concatenate([mask for _, mask in chain_coordinates])
    covariance = elastic_network_covariance(coordinates, residue_mask, contact_decay, damping, eps)
    fluctuations = jnp.clip(jnp.einsum('iaia->i', covariance.reshape(len(coordinates), 3, len(coordinates), 3)), 0.0, None) * residue_mask
    fluctuation_share = fluctuations / (fluctuations.sum() + eps)
    collectivity = jnp.exp(-jnp.where(fluctuation_share > 0, fluctuation_share * jnp.log(fluctuation_share + eps), 0.0).sum()) / (residue_mask.sum() + eps)
    softest_mode = ((coordinates - masked_coordinate_centroid(coordinates, residue_mask)) * residue_mask[:, None]).reshape(-1)
    for _ in range(power_iterations):
        softest_mode = covariance @ softest_mode
        softest_mode = softest_mode / (jnp.sqrt(jnp.square(softest_mode).sum()) + eps)
    concentration = jnp.clip(softest_mode @ (covariance @ softest_mode) / ((jnp.diagonal(covariance) * jnp.repeat(residue_mask, 3)).sum() + eps), 0.0, 1.0)
    return 1 - jnp.clip(collectivity, 0.0, 1.0) * concentration

def binder_domain_ids(binder_length: int, n_domains: int=2, min_domain_size: int=50, max_domain_size: int=180, seed: int=0) -> tuple[int, ...]:
    minimum_size = max(1, min_domain_size)
    domain_count = min(n_domains, binder_length // minimum_size)
    if domain_count < 2:
        return (0,) * binder_length
    random_generator, domain_sizes = random.Random(seed), [minimum_size] * domain_count
    for _ in range(binder_length - minimum_size * domain_count):
        extendable_domains = [index for index, domain_size in enumerate(domain_sizes) if domain_size < max(max_domain_size, minimum_size)]
        domain_sizes[random_generator.choice(extendable_domains) if extendable_domains else random_generator.randrange(domain_count)] += 1
    return tuple(domain_index for domain_index, domain_size in enumerate(domain_sizes) for _ in range(domain_size))

def domain_linker_membership(pae: Array, domain_membership: Array, gap: float, sharpness: float) -> Array:
    residue_indices = jnp.arange(len(pae))
    separated_pairs = jnp.abs(residue_indices[:, None] - residue_indices[None, :]) > 2
    counts = separated_pairs @ domain_membership
    pae_to_domain = pae * separated_pairs @ domain_membership / (counts + 1e-08)
    own_domain_pae = (pae_to_domain * domain_membership).sum(-1)
    empty_domains = domain_membership.sum(0) < 0.5
    nearest_other_pae = jnp.min(pae_to_domain + domain_membership * 1000000000.0 + empty_domains[None, :] * 1000000000.0, axis=-1)
    return jax.nn.sigmoid((gap - (nearest_other_pae - own_domain_pae)) / sharpness)

def multidomain_linker_residues(settings: dict, binder: Protein, binder_pae: Array, seed: int=0) -> Array:
    domain_ids = binder_domain_ids(len(binder), settings.get('n_domains', 2), settings.get('min_domain_size', 50), settings.get('max_domain_size', 180), seed)
    if max(domain_ids) < 1:
        return jnp.zeros(len(binder), dtype=bool)
    domain_membership = binder_domain_membership(binder, domain_ids, max(settings.get('max_domains', 2), settings.get('n_domains', 2), max(domain_ids) + 1))
    linker_membership = domain_linker_membership(binder_pae / 31.0, domain_membership, float(settings.get('domain_linker_gap', 0.1)), float(settings.get('domain_linker_sharpness', 0.03)))
    return (linker_membership >= float(settings.get('domain_linker_fix_cut', 0.5))) & real_residue_mask(binder.flags)

@loss('multidomain')
def multidomain_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', chain: str='binder', domain_ids: tuple[int, ...]=(), n_domains: int=2, min_domain_size: int=50, max_domain_size: int=180, max_domains: int=2, seed: int=0, domain_rg_weight: float=0.3, domain_sep_weight: float=0.0, domain_contact_cutoff: float=8.0, domain_pae_margin: float=0.15, domain_linker_gap: float=0.1, domain_linker_sharpness: float=0.03, domain_linker_helix_weight: float=0.0) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein = protein_states[prediction_state][chain]
    residue_weights = real_residue_weights(protein.flags)
    domain_ids = domain_ids or binder_domain_ids(len(protein), n_domains, min_domain_size, max_domain_size, seed)
    domain_membership = binder_domain_membership(protein, domain_ids, max(max_domains, n_domains, max(domain_ids) + 1))
    same_domain = domain_membership @ domain_membership.T
    different_domain = (1 - same_domain) * residue_weights[:, None] * residue_weights[None, :]
    chain_slice = chain_residue_slices(protein_states[prediction_state])[chain]
    binder_pae = predictions[prediction_state].metrics['pae'][chain_slice, chain_slice] / 31.0
    sequence_hardness = _masked_mean(amino_acid_probabilities(protein.sequence).max(-1), residue_weights)
    linker_membership = domain_linker_membership(binder_pae, domain_membership, domain_linker_gap, domain_linker_sharpness) * residue_weights
    domain_residue_weights = (1 - sequence_hardness * linker_membership) * residue_weights
    domain_pair_weights = domain_residue_weights[:, None] * domain_residue_weights[None, :]
    domain_residue_weight = domain_membership.T @ domain_residue_weights
    coordinates, _ = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    domain_centroids = domain_membership.T @ (coordinates * domain_residue_weights[:, None]) / (domain_residue_weight[:, None] + 1e-08)
    squared_distances = jnp.square(coordinates - domain_membership @ domain_centroids).sum(-1)
    domain_sizes = domain_membership.sum(0)
    domain_radii = jnp.sqrt(domain_membership.T @ (squared_distances * domain_residue_weights) / (domain_residue_weight + 1e-08) + 1e-08)
    radius_loss = _masked_mean(jax.nn.elu(domain_radii - globular_radius(domain_sizes + 1e-08)), domain_sizes > 0.5)
    within_domain_pair_weights, between_domain_pair_weights = same_domain * domain_pair_weights, different_domain * domain_pair_weights
    within_domain_pae, between_domain_pae = _masked_mean(binder_pae, within_domain_pair_weights), _masked_mean(binder_pae, between_domain_pair_weights)
    domain_loss = within_domain_pae + jax.nn.relu(domain_pae_margin - (between_domain_pae - within_domain_pae)) + domain_rg_weight * radius_loss
    if domain_sep_weight:
        domain_loss += domain_sep_weight * _masked_mean(jax.nn.sigmoid((domain_contact_cutoff - pairwise_atom_distances(coordinates, coordinates, 1e-08)) / 2), between_domain_pair_weights)
    if domain_linker_helix_weight:
        distogram = predictions[prediction_state].metrics['distogram'][chain_slice, chain_slice]
        helix_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram.shape[-1]) < 6, True)
        linker_pair_weights = jnp.diagonal(linker_membership[:, None] * linker_membership[None, :], 3)
        domain_loss -= domain_linker_helix_weight * sequence_hardness * _masked_mean(jnp.diagonal(helix_loss, 3), linker_pair_weights)
    return domain_loss * (different_domain.sum() > 0)

def build_design_losses(settings: dict, targets: dict[str, float], binder_length: int, seed: int=0) -> dict[str, DesignLoss]:
    from bindcraft.settings import load_settings, resolve_designed_binder_chain
    settings = load_settings(settings)
    binder_chain = resolve_designed_binder_chain(settings)
    domain_ids = binder_domain_ids(binder_length, settings.get('n_domains', 2), settings.get('min_domain_size', 50), settings.get('max_domain_size', 180), seed) if settings.get('weights_multidomain') else ()
    design_losses = {}
    for name, entry in build_losses(settings, seed).items():
        loss_function = REGISTERED_LOSSES[name]
        parameters = inspect.signature(loss_function).parameters
        loss_parameters = resolve_binder_role(loss_function, dict(entry.function.keywords), binder_chain)
        #multi-domain case
        if domain_ids and 'domain_ids' in parameters:
            loss_parameters['domain_ids'] = domain_ids
        if 'prediction_state' not in parameters:
            #fold switching case
            design_losses[name] = DesignLoss(_bound_metric(loss_function, loss_parameters), entry.weight, entry.required_states, entry.interface_mask)
            continue
        configured_state = loss_parameters.get('prediction_state', 'complex')
        reads_binder_only = configured_state == 'complex' and LOSS_TARGET_WEIGHTING[name] == 'binder_only'
        target_states = {'complex': 1.0} if reads_binder_only else targets if configured_state == 'complex' else {configured_state: targets.get(configured_state, 1.0)}
        for target_name, target_weight in target_states.items():
            target_loss_weight = {'binds_target': max(target_weight, 0), 'avoids_target': max(-target_weight, 0), 'every_target': 1.0, 'binder_only': 1.0}[LOSS_TARGET_WEIGHTING[name]]
            if not target_loss_weight:
                continue
            target_params = {**loss_parameters, 'prediction_state': target_name}
            if 'target' in parameters:
                target_params.setdefault('target', settings.get('target_chain', 'target'))
            required_states = set() if reads_binder_only else {target_name}
            if 'reference_state' in parameters:
                required_states.add(target_params.get('reference_state', parameters['reference_state'].default))
            loss_name = name if len(target_states) == 1 else f'{name}.{target_name}'
            bound_loss = _bound_metric(loss_function, target_params)
            design_losses[loss_name] = DesignLoss(bound_loss, entry.weight * target_loss_weight, frozenset(required_states))
    return design_losses
