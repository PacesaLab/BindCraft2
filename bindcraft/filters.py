import functools
import inspect
import biotite.structure as struc
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, NamedTuple, TYPE_CHECKING
from bindcraft.epitope_targeting import EPITOPE_CUTOFF, epitope_residues
from bindcraft.developability import mhc_panels
from bindcraft.loss import _masked_mean, align_binder_coordinates, aligned_binder_tm_score, bind_state_metric, binder_binding_mask, binder_copy_chains, binder_framework_mask, bound_and_unbound_binder_coordinates, chain_atom_coordinates, chain_pair_pae_loss, chain_residue_slices, complex_residue_weights, core_aligned_interface_rmsd, induced_fit_interface_masks, mhc_epitope_score, pairwise_atom_distances, pooled_protease_site_score, resolve_binder_role, resolve_prediction_state, resolve_target_chain, soft_maximum, terminus_target_direction_cosine
from bindcraft.protein import AMINO_ACIDS, ATOM_INDEX, BINDER_ALONE, Protein, ProteinStates, ResidueFlags, StructurePredictions, build_atom_array, has_residue_flag, output_chain_letters, parse_scaffold_edits, real_residue_count, real_residue_mask, redesignable_residue_mask, structure_chain_names

if TYPE_CHECKING:
    from bindcraft.settings import BinderDesignSettings

SASA_PROBE_POINTS = 100
SURFACE_RELATIVE_SASA = 0.2
HYDROPHOBIC_AMINO_ACIDS = frozenset('ACVILMPFWY')
MAX_RESIDUE_SASA = {'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0, 'C': 148.0, 'Q': 214.0, 'E': 214.0, 'G': 97.0, 'H': 216.0, 'I': 195.0, 'L': 191.0, 'K': 230.0, 'M': 203.0, 'F': 228.0, 'P': 154.0, 'S': 143.0, 'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0}

class DesignFilter(NamedTuple):
    function: Callable[[ProteinStates, StructurePredictions], float | None]
    threshold: float | None
    higher: bool
    required_states: frozenset[str]
    mandatory: bool = True
REGISTERED_FILTER_METRICS: dict[str, Callable] = {}

def filter_metric(name: str):
    def register(function: Callable) -> Callable:
        REGISTERED_FILTER_METRICS[name] = function
        return function
    return register

def build_filters(settings: dict, binder_chain: str='binder') -> dict[str, DesignFilter]:
    design_filters = {}
    for name, function in REGISTERED_FILTER_METRICS.items():
        entry = settings.get(name)
        if not entry or entry.get('threshold') is None:
            continue
        bound_metric, required_states = bind_state_metric(function, {**entry, 'params': resolve_binder_role(function, entry.get('params', {}), binder_chain)})
        design_filters[name] = DesignFilter(function=bound_metric, threshold=float(entry['threshold']), higher=entry.get('higher', True), required_states=required_states, mandatory=entry.get('mandatory', True) is not False)
    return design_filters

def reads_unbound_binder(name: str, design_filter: DesignFilter) -> bool:
    reference_state = inspect.signature(REGISTERED_FILTER_METRICS[name]).parameters.get('reference_state')
    return BINDER_ALONE in design_filter.required_states or (reference_state is not None and reference_state.default == BINDER_ALONE)

def validates_unbound_binder(design_filters: dict[str, DesignFilter]) -> bool:
    return any(reads_unbound_binder(name, design_filter) for name, design_filter in design_filters.items())

def evaluate_design_filters(filters: dict[str, DesignFilter], protein_states: ProteinStates, predictions: StructurePredictions) -> tuple[bool | list[str], dict[str, float]]:
    metrics: dict[str, float] = {}
    failed_filters = []
    for name, design_filter in filters.items():
        value = design_filter.function(protein_states, predictions) if design_filter.required_states <= set(protein_states) & set(predictions) else None
        if value is None:
            if design_filter.mandatory:
                failed_filters.append(f'{name} (not measured)')
            continue
        metrics[name] = float(value)
        if design_filter.threshold is None:
            continue
        passed = metrics[name] >= design_filter.threshold if design_filter.higher else metrics[name] <= design_filter.threshold
        if not passed:
            failed_filters.append(name)
    return True if not failed_filters else failed_filters, metrics

def passes_design_filters(filters: dict[str, DesignFilter], protein_states: ProteinStates, predictions: StructurePredictions) -> bool | list[str]:
    return evaluate_design_filters(filters, protein_states, predictions)[0]

@filter_metric('Unbound_Binder_pLDDT')
def plddt_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str | None=None) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    confidence = predictions[prediction_state].metrics['plddt']
    if chain is None:
        return float(jnp.mean(confidence))
    protein_complex = protein_states[prediction_state]
    chain = resolve_target_chain(protein_complex, chain, prediction_state)
    if chain not in protein_complex:
        return None
    return float(_masked_mean(confidence[chain_residue_slices(protein_complex)[chain]], real_residue_mask(protein_complex[chain].flags)))

filter_metric('Target_pLDDT')(functools.partial(plddt_metric, prediction_state='complex', chain='target'))

@filter_metric('pTM')
def ptm_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> float:
    return float(predictions[resolve_prediction_state(predictions, prediction_state)].metrics['ptm'])

@filter_metric('i_pTM')
@filter_metric('i_pTM_detarget')
def iptm_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> float:
    return float(predictions[resolve_prediction_state(predictions, prediction_state)].metrics['iptm'])

@filter_metric('i_pAE')
@filter_metric('i_pAE_detarget')
def ipae_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target') -> float:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return float(chain_pair_pae_loss(protein_states, predictions, prediction_state, (binder,), (resolve_target_chain(predictions[prediction_state].protein_complex, target, prediction_state),)))

def binder_target_contact_masks(binder: Protein, target: Protein, cutoff: float=4.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    binder_atom_positions, binder_atom_mask = binder.atoms.reshape(-1, 3), binder.atom_mask.reshape(-1)
    target_atom_positions, target_atom_mask = target.atoms.reshape(-1, 3), target.atom_mask.reshape(-1)
    contact = (pairwise_atom_distances(binder_atom_positions, target_atom_positions) <= cutoff) & binder_atom_mask[:, None] & target_atom_mask[None, :]
    return contact.any(-1).reshape(len(binder), -1).any(-1), contact.any(0).reshape(len(target), -1).any(-1)

def binder_assembly_contact_masks(protein_complex: dict[str, Protein], binder: str, target: str, cutoff: float=4.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    contact_masks = [binder_target_contact_masks(protein_complex[name], protein_complex[target], cutoff) for name in binder_copy_chains(protein_complex, binder)]
    return jnp.concatenate([binder_mask for binder_mask, _ in contact_masks]), jnp.stack([target_mask for _, target_mask in contact_masks]).any(0)

@filter_metric('Interface_Residues')
@filter_metric('Interface_Residues_detarget')
def interface_residues_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    return float(binder_assembly_contact_masks(protein_complex, binder, resolve_target_chain(protein_complex, target, prediction_state), cutoff)[0].sum())

def tm_score_distance_scale(partner_count: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(1.24 * (jnp.maximum(partner_count, 19.0) - 15.0) ** (1 / 3) - 1.8, 1.0)

def anchored_interface_tm_scores(pae: jnp.ndarray, contact: jnp.ndarray) -> jnp.ndarray:
    partner_counts = contact.sum(-1).astype(jnp.float32)
    tm_scores = 1.0 / (1.0 + (pae / tm_score_distance_scale(partner_counts)[:, None]) ** 2)
    return jnp.where(contact, tm_scores, 0.0).sum(-1) / jnp.maximum(partner_counts, 1.0)

def interface_pae_directions(predictions: StructurePredictions, prediction_state: str, binder: str, target: str, cutoff: float) -> tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if 'pae' not in predictions[prediction_state].metrics:
        return None
    protein_complex = predictions[prediction_state].protein_complex
    target = resolve_target_chain(protein_complex, target, prediction_state)
    binder_chains = binder_copy_chains(protein_complex, binder)
    target_ca_coordinates, target_resolved_ca = chain_atom_coordinates(protein_complex[target])
    binder_ca = [chain_atom_coordinates(protein_complex[name]) for name in binder_chains]
    contact = jnp.concatenate([(pairwise_atom_distances(coordinates, target_ca_coordinates) <= cutoff) & (resolved[:, None] * target_resolved_ca[None, :] > 0) for coordinates, resolved in binder_ca])
    if not bool(contact.any()):
        return None
    chain_slices = chain_residue_slices(protein_complex)
    interface_pae = jnp.concatenate([predictions[prediction_state].metrics['pae'][chain_slices[name], chain_slices[target]] for name in binder_chains])
    return ((interface_pae, contact), (interface_pae.T, contact.T))

def residue_confidence_tracks(predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=8.0) -> dict[str, np.ndarray]:
    state = resolve_prediction_state(predictions, prediction_state)
    metrics = predictions[state].metrics
    tracks = {}
    if 'plddt' in metrics:
        tracks['pLDDT'] = np.asarray(metrics['plddt'], dtype=np.float32) * 100.0
    if 'pae' in metrics:
        tracks['PAE_mean'] = np.asarray(metrics['pae'], dtype=np.float32).mean(-1)
    protein_complex = predictions[state].protein_complex
    directions = interface_pae_directions(predictions, prediction_state, binder, target, cutoff) if binder in protein_complex and len(protein_complex) > 1 else None
    if directions is not None:
        chain_slices = chain_residue_slices(protein_complex)
        interface = np.full(sum(len(protein) for protein in protein_complex.values()), np.nan, dtype=np.float32)
        for chains, (pae, contact) in zip((binder_copy_chains(protein_complex, binder), (resolve_target_chain(protein_complex, target, state),)), directions):
            anchored = np.asarray(anchored_interface_tm_scores(pae, contact), dtype=np.float32)
            scored, written = np.where(np.asarray(contact).sum(-1) > 0, anchored, np.nan), 0
            for chain in chains:
                interface[chain_slices[chain]] = scored[written:written + len(protein_complex[chain])]
                written += len(protein_complex[chain])
        tracks['i_pDAE'] = interface
    return tracks

@filter_metric('i_pDAE')
def interface_pdae_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=8.0) -> float | None:
    directions = interface_pae_directions(predictions, prediction_state, binder, target, cutoff)
    return None if directions is None else float(max(anchored_interface_tm_scores(pae, contact).max() for pae, contact in directions))

INTERFACE_PDAE_METRICS = {'i_pDAE': interface_pdae_metric}

AF2_CONFIDENCE_METRICS = frozenset({'pLDDT', 'pTM', 'i_pTM', 'Unbound_Binder_pLDDT', 'Target_pLDDT'})

def confidence_stage_filters(stage_filters: dict[str, DesignFilter]) -> dict[str, DesignFilter]:
    return {name: design_filter for name, design_filter in stage_filters.items() if name.partition('.')[0] in AF2_CONFIDENCE_METRICS}

def chain_atom_clash_count(predictions: StructurePredictions, prediction_state: str, cutoff: float, atom_indices: jnp.ndarray) -> float:
    protein_complex = predictions[prediction_state].protein_complex
    chain_names = tuple(sorted(protein_complex))
    atom_coordinates = jnp.concatenate([protein_complex[name].atoms[:, atom_indices].reshape(-1, 3) for name in chain_names])
    resolved_atom_mask = jnp.concatenate([protein_complex[name].atom_mask[:, atom_indices].reshape(-1) for name in chain_names]).astype(bool)
    chain_indices = jnp.concatenate([jnp.full((len(protein_complex[name]) * len(atom_indices),), chain_index, dtype=jnp.int32) for chain_index, name in enumerate(chain_names)])
    atom_distances = pairwise_atom_distances(atom_coordinates, atom_coordinates)
    different_chains = chain_indices[:, None] != chain_indices[None, :]
    resolved_atom_pairs = resolved_atom_mask[:, None] & resolved_atom_mask[None, :]
    unique_atom_pairs = jnp.triu(jnp.ones(atom_distances.shape, dtype=bool), k=1)
    return float(((atom_distances <= cutoff) & resolved_atom_pairs & different_chains & unique_atom_pairs).sum())

@filter_metric('Backbone_Clashes')
def backbone_clashes_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', cutoff: float=2.5) -> float:
    return chain_atom_clash_count(predictions, resolve_prediction_state(predictions, prediction_state), cutoff, jnp.array([ATOM_INDEX['CA']]))

@filter_metric('All_Atom_Clashes')
def all_atom_clashes_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', cutoff: float=2.5) -> float:
    return chain_atom_clash_count(predictions, resolve_prediction_state(predictions, prediction_state), cutoff, jnp.arange(len(ATOM_INDEX)))

@filter_metric('Off_Paratope_Contact_Fraction')
def off_paratope_contact_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    target = resolve_target_chain(protein_complex, target, prediction_state)
    interface_residue_mask, _ = binder_assembly_contact_masks(protein_complex, binder, target, cutoff)
    contact_residue_count = float(interface_residue_mask.sum())
    if contact_residue_count == 0:
        return 0.0
    allowed_contact_residues = jnp.concatenate([has_residue_flag(protein_states[prediction_state][name].flags, ResidueFlags.CONTACT) for name in binder_copy_chains(protein_states[prediction_state], binder)])
    return float((interface_residue_mask & ~allowed_contact_residues).sum()) / contact_residue_count

@filter_metric('Hotspot_Contact_Fraction')
def hotspot_contact_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(predictions[prediction_state].protein_complex, target, prediction_state)
    hotspot_residues = has_residue_flag(protein_states[prediction_state][target].flags, ResidueFlags.HOTSPOT)
    if not bool(hotspot_residues.any()):
        return None
    _, contacted_target_residues = binder_assembly_contact_masks(predictions[prediction_state].protein_complex, binder, target, cutoff)
    return float((contacted_target_residues & hotspot_residues).sum()) / float(hotspot_residues.sum())

@filter_metric('Coldspot_Contact_Fraction')
def coldspot_contact_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(predictions[prediction_state].protein_complex, target, prediction_state)
    coldspot_residues = has_residue_flag(protein_states[prediction_state][target].flags, ResidueFlags.COLDSPOT)
    if not bool(coldspot_residues.any()):
        return None
    _, contacted_target_residues = binder_assembly_contact_masks(predictions[prediction_state].protein_complex, binder, target, cutoff)
    return float((contacted_target_residues & coldspot_residues).sum()) / float(coldspot_residues.sum())

def binder_one_letter_sequence(protein: Protein) -> np.ndarray:
    return np.array([AMINO_ACIDS[index] for index in np.asarray(protein.sequence.argmax(-1))])

def residue_solvent_accessible_area(protein_complex: dict[str, Protein], chains: tuple[str, ...] | None=None) -> np.ndarray | None:
    atom_array = build_atom_array(protein_complex)
    if atom_array is None:
        return None
    per_atom_area = struc.sasa(atom_array, probe_radius=1.4, point_number=SASA_PROBE_POINTS)
    per_residue_area = struc.apply_residue_wise(atom_array, np.nan_to_num(per_atom_area), np.sum)
    if chains is None:
        return per_residue_area
    chain_letters = output_chain_letters(protein_complex)
    residue_chains = atom_array.chain_id[struc.get_residue_starts(atom_array)]
    return per_residue_area[np.isin(residue_chains, [chain_letters[chain] for chain in chains])]

@filter_metric('Interface_BuriedArea')
def interface_buried_area_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    protein_complex = predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex
    assembly = binder_copy_chains(protein_complex, binder)
    unbound_area = residue_solvent_accessible_area({chain: protein_complex[chain] for chain in assembly})
    bound_area = residue_solvent_accessible_area(protein_complex, assembly)
    if unbound_area is None or bound_area is None:
        return None
    return max(0.0, float(unbound_area.sum() - bound_area.sum()))

@filter_metric('Interface_BuriedArea_Fraction')
def interface_buried_area_fraction_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    protein_complex = predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex
    assembly = binder_copy_chains(protein_complex, binder)
    unbound_area = residue_solvent_accessible_area({chain: protein_complex[chain] for chain in assembly})
    bound_area = residue_solvent_accessible_area(protein_complex, assembly)
    if unbound_area is None or bound_area is None:
        return None
    unbound_total = float(unbound_area.sum())
    if unbound_total <= 0.0:
        return None
    return max(0.0, (unbound_total - float(bound_area.sum())) / unbound_total)

@filter_metric('Surface_Hydrophobicity')
def surface_hydrophobicity_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    protein_complex = predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex
    assembly = binder_copy_chains(protein_complex, binder)
    unbound_area = residue_solvent_accessible_area({chain: protein_complex[chain] for chain in assembly})
    if unbound_area is None:
        return None
    sequence = np.concatenate([binder_one_letter_sequence(protein_complex[chain]) for chain in assembly])
    maximum_area = np.array([MAX_RESIDUE_SASA[amino_acid] for amino_acid in sequence])
    surface_residues = unbound_area / maximum_area >= SURFACE_RELATIVE_SASA
    if not surface_residues.any():
        return 0.0
    hydrophobic_residues = np.array([amino_acid in HYDROPHOBIC_AMINO_ACIDS for amino_acid in sequence])
    return float((surface_residues & hydrophobic_residues).sum()) / float(surface_residues.sum())

@filter_metric('Interface_Hydrophobicity')
def interface_hydrophobicity_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    target = resolve_target_chain(protein_complex, target, prediction_state)
    interface_residues = np.asarray(binder_assembly_contact_masks(protein_complex, binder, target, cutoff)[0])
    if not interface_residues.any():
        return None
    sequence = np.concatenate([binder_one_letter_sequence(protein_complex[name]) for name in binder_copy_chains(protein_complex, binder)])
    hydrophobic_residues = np.array([amino_acid in HYDROPHOBIC_AMINO_ACIDS for amino_acid in sequence])
    return float((interface_residues & hydrophobic_residues).sum()) / float(interface_residues.sum())

def binder_secondary_structure(protein_complex: dict[str, Protein], binder: str) -> np.ndarray | None:
    atom_arrays = (build_atom_array({chain: protein_complex[chain]}) for chain in binder_copy_chains(protein_complex, binder))
    annotations = [struc.annotate_sse(atom_array) for atom_array in atom_arrays if atom_array is not None]
    return np.concatenate(annotations) if annotations else None

def secondary_structure_fraction(protein_states: ProteinStates, predictions: StructurePredictions, secondary_structure_code: str, prediction_state: str, binder: str) -> float | None:
    secondary_structure = binder_secondary_structure(predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex, binder)
    if secondary_structure is None or len(secondary_structure) == 0:
        return None
    matching_residues = secondary_structure == secondary_structure_code if secondary_structure_code != 'c' else ~np.isin(secondary_structure, ('a', 'b'))
    return float(matching_residues.sum()) / float(len(secondary_structure))

@filter_metric('Binder_Helix_Fraction')
def binder_helix_fraction_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    return secondary_structure_fraction(protein_states, predictions, 'a', prediction_state, binder)

@filter_metric('Binder_BetaSheet_Fraction')
def binder_beta_sheet_fraction_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    return secondary_structure_fraction(protein_states, predictions, 'b', prediction_state, binder)

@filter_metric('Binder_Loop_Fraction')
def binder_loop_fraction_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    return secondary_structure_fraction(protein_states, predictions, 'c', prediction_state, binder)

@filter_metric('SS_pLDDT')
def structured_residue_plddt_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    secondary_structure = binder_secondary_structure(protein_complex, binder)
    if secondary_structure is None:
        return None
    chain_slices = chain_residue_slices(protein_complex)
    plddt = np.asarray(predictions[prediction_state].metrics['plddt'])
    binder_plddt = np.concatenate([plddt[chain_slices[chain]] for chain in binder_copy_chains(protein_complex, binder)])
    structured_residues = np.isin(secondary_structure, ('a', 'b'))
    if not structured_residues.any():
        return 0.0
    return float(binder_plddt[structured_residues].mean())

def interface_secondary_structure_fraction(protein_states: ProteinStates, predictions: StructurePredictions, secondary_structure_code: str, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    target = resolve_target_chain(protein_complex, target, prediction_state)
    interface_residues = np.asarray(binder_assembly_contact_masks(protein_complex, binder, target, cutoff)[0])
    secondary_structure = binder_secondary_structure(protein_complex, binder)
    if secondary_structure is None or len(secondary_structure) != len(interface_residues) or not interface_residues.any():
        return None
    matching_residues = secondary_structure == secondary_structure_code if secondary_structure_code != 'c' else ~np.isin(secondary_structure, ('a', 'b'))
    return float((interface_residues & matching_residues).sum()) / float(interface_residues.sum())
filter_metric('Interface_Helix_Fraction')(functools.partial(interface_secondary_structure_fraction, secondary_structure_code='a'))
filter_metric('Interface_BetaSheet_Fraction')(functools.partial(interface_secondary_structure_fraction, secondary_structure_code='b'))
filter_metric('Interface_Loop_Fraction')(functools.partial(interface_secondary_structure_fraction, secondary_structure_code='c'))

@filter_metric('Binder_RMSD')
@filter_metric('Induced_Fit_RMSD')
def binder_rmsd_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', reference_state: str=BINDER_ALONE, binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return None
    coordinates, reference_coordinates, valid_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, binder)
    if float(valid_mask.sum()) < 3:
        return None
    aligned_coordinates = align_binder_coordinates(coordinates, reference_coordinates, valid_mask)
    squared_deviation = jnp.square(aligned_coordinates - reference_coordinates).sum(-1)
    return float(jnp.sqrt((squared_deviation * valid_mask).sum() / valid_mask.sum()))

@filter_metric('Target_RMSD')
def target_rmsd_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    if target not in protein_states[prediction_state] or target not in predictions[prediction_state].protein_complex:
        return None
    reference_coordinates, reference_mask = chain_atom_coordinates(protein_states[prediction_state][target])
    coordinates, coordinate_mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[target])
    valid_mask = reference_mask * coordinate_mask
    if float(valid_mask.sum()) < 3:
        return None
    aligned_coordinates = align_binder_coordinates(coordinates, reference_coordinates, valid_mask)
    squared_deviation = jnp.square(aligned_coordinates - reference_coordinates).sum(-1)
    return float(jnp.sqrt((squared_deviation * valid_mask).sum() / valid_mask.sum()))

@filter_metric('Induced_Fit_Interface_RMSD')
def induced_fit_interface_rmsd_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', reference_state: str=BINDER_ALONE, binder: str='binder', target: str='target', cutoff: float=8.0, interface_residues: tuple[int, ...]=()) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return None
    coordinates, reference_coordinates, valid_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, binder)
    interface_mask, alignment_mask = induced_fit_interface_masks(protein_states, predictions, coordinates, valid_mask, prediction_state, target, cutoff, interface_residues)
    if float(interface_mask.sum()) < 3 or float(alignment_mask.sum()) < 3:
        return None
    return float(core_aligned_interface_rmsd(coordinates, reference_coordinates, interface_mask, alignment_mask))

@filter_metric('Induced_Fit_TM')
def induced_fit_tm_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', reference_state: str=BINDER_ALONE, binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return None
    coordinates, reference_coordinates, valid_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, binder)
    if float(valid_mask.sum()) < 3:
        return None
    return float(aligned_binder_tm_score(coordinates, reference_coordinates, valid_mask))

def epitope_contact_masks(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, binder: str, target: str, cutoff: float, epitope_cutoff: float):
    target = resolve_target_chain(predictions[prediction_state].protein_complex, target, prediction_state)
    target_protein = protein_states[prediction_state][target]
    hotspot_residues = np.asarray(has_residue_flag(target_protein.flags, ResidueFlags.HOTSPOT))
    if not hotspot_residues.any() or not np.asarray(target_protein.atom_mask)[hotspot_residues].any():
        return None
    epitope = epitope_residues(np.asarray(target_protein.atoms, dtype=np.float64), np.asarray(target_protein.atom_mask, dtype=bool), hotspot_residues, epitope_cutoff)
    return epitope, np.asarray(binder_assembly_contact_masks(predictions[prediction_state].protein_complex, binder, target, cutoff)[1])

@filter_metric('Off_Epitope_Contact_Fraction')
def off_epitope_contact_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0, epitope_cutoff: float=EPITOPE_CUTOFF) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    epitope_contacts = epitope_contact_masks(protein_states, predictions, prediction_state, binder, target, cutoff, epitope_cutoff)
    if epitope_contacts is None:
        return None
    epitope, contacted_target_residues = epitope_contacts
    if not contacted_target_residues.any():
        return 0.0
    return float((contacted_target_residues & ~epitope).sum()) / float(contacted_target_residues.sum())

@filter_metric('Epitope_Residues_Contacted')
def epitope_residues_contacted_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0, epitope_cutoff: float=EPITOPE_CUTOFF) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    epitope_contacts = epitope_contact_masks(protein_states, predictions, prediction_state, binder, target, cutoff, epitope_cutoff)
    if epitope_contacts is None:
        return None
    epitope, contacted_target_residues = epitope_contacts
    return float((contacted_target_residues & epitope).sum())

@filter_metric('Cyclic_Closure_Distance')
def cyclic_closure_distance_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if not bool(has_residue_flag(protein_states[prediction_state][binder].flags, ResidueFlags.CYCLIC).any()):
        return None
    protein = predictions[prediction_state].protein_complex[binder]
    c_terminus = int(real_residue_count(protein_states[prediction_state][binder].flags)) - 1
    if not (bool(protein.atom_mask[0, ATOM_INDEX['N']]) and bool(protein.atom_mask[c_terminus, ATOM_INDEX['C']])):
        return None
    return float(jnp.linalg.norm(protein.atoms[0, ATOM_INDEX['N']].astype(jnp.float32) - protein.atoms[c_terminus, ATOM_INDEX['C']].astype(jnp.float32)))

def scaffold_framework_chains(protein_complex: dict[str, Protein], binder: str, scaffold: str, scaffold_chain: str) -> list[tuple[str, str]]:
    if not scaffold:
        return []
    scaffold_chains = [scaffold_chain] if scaffold_chain else structure_chain_names(scaffold)
    return [(binder, scaffold_chains[0])] if len(scaffold_chains) == 1 else list(zip(binder_copy_chains(protein_complex, binder), scaffold_chains))

def scaffold_framework_correspondence(protein_states: ProteinStates, prediction_state: str, binder: str, scaffold: str, scaffold_edits: str, scaffold_chain: str) -> tuple[np.ndarray, Protein, np.ndarray] | None:
    design_framework = np.asarray(has_residue_flag(protein_states[prediction_state][binder].flags, ResidueFlags.TEMPLATE))
    scaffold_protein = Protein.from_structure(scaffold, chains=scaffold_chain)[scaffold_chain]
    residue_numbers = np.asarray(scaffold_protein.residue_index)
    scaffold_framework = np.ones(len(scaffold_protein), dtype=bool)
    for start, end, _, _ in parse_scaffold_edits(scaffold_edits, scaffold_chain, jax.random.PRNGKey(0)):
        scaffold_framework &= (residue_numbers < start) | (residue_numbers > end)
    return (design_framework, scaffold_protein, scaffold_framework) if design_framework.sum() == scaffold_framework.sum() else None

@filter_metric('Scaffold_Sequence_Retained_Fraction')
def scaffold_sequence_retained_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', scaffold: str='', scaffold_edits: str='', scaffold_chain: str='') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    held_framework_residues = []
    for binder_chain, chain in scaffold_framework_chains(protein_states[prediction_state], binder, scaffold, scaffold_chain):
        correspondence = scaffold_framework_correspondence(protein_states, prediction_state, binder_chain, scaffold, scaffold_edits, chain)
        if correspondence is None:
            return None
        design_framework, scaffold_protein, scaffold_framework = correspondence
        held_framework_residues.append(binder_one_letter_sequence(protein_states[prediction_state][binder_chain])[design_framework] == binder_one_letter_sequence(scaffold_protein)[scaffold_framework])
    return float(np.concatenate(held_framework_residues).mean()) if held_framework_residues else None

@filter_metric('Scaffold_Framework_RMSD')
def scaffold_framework_rmsd_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', scaffold: str='', scaffold_edits: str='', scaffold_chain: str='') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    framework_deviations, framework_valid = [], []
    for binder_chain, chain in scaffold_framework_chains(protein_states[prediction_state], binder, scaffold, scaffold_chain):
        correspondence = scaffold_framework_correspondence(protein_states, prediction_state, binder_chain, scaffold, scaffold_edits, chain)
        if correspondence is None:
            return None
        design_framework, scaffold_protein, scaffold_framework = correspondence
        coordinates, coordinate_mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[binder_chain])
        reference_coordinates, reference_mask = chain_atom_coordinates(scaffold_protein)
        coordinates, reference_coordinates = coordinates[design_framework], reference_coordinates[scaffold_framework]
        valid_mask = coordinate_mask[design_framework] * reference_mask[scaffold_framework]
        framework_deviations.append(jnp.square(align_binder_coordinates(coordinates, reference_coordinates, valid_mask) - reference_coordinates).sum(-1))
        framework_valid.append(valid_mask)
    if not framework_valid:
        return None
    squared_deviation, valid_mask = jnp.concatenate(framework_deviations), jnp.concatenate(framework_valid)
    if float(valid_mask.sum()) < 3:
        return None
    return float(jnp.sqrt((squared_deviation * valid_mask).sum() / valid_mask.sum()))

@filter_metric('Framework_Packing_Fraction')
def framework_packing_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', cutoff: float=4.5, sequence_separation: int=20) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_slice = chain_residue_slices(protein_complex)[binder]
    residue_count = complex_residue_weights(protein_complex).shape[0]
    framework_residues = np.flatnonzero(np.asarray(binder_framework_mask(protein_complex, binder, residue_count))[binder_slice])
    paratope_residues = np.flatnonzero(np.asarray(binder_binding_mask(protein_complex, binder, residue_count))[binder_slice])
    if not (len(framework_residues) and len(paratope_residues)):
        return None
    protein = predictions[prediction_state].protein_complex[binder]
    atom_count = protein.atoms.shape[1]
    contact = (pairwise_atom_distances(protein.atoms[framework_residues].reshape(-1, 3), protein.atoms[paratope_residues].reshape(-1, 3)) <= cutoff) & protein.atom_mask[framework_residues].reshape(-1)[:, None] & protein.atom_mask[paratope_residues].reshape(-1)[None, :]
    packed = np.asarray(contact.reshape(len(framework_residues), atom_count, len(paratope_residues), atom_count).any((1, 3)))
    return float((packed & (np.abs(framework_residues[:, None] - paratope_residues[None, :]) >= sequence_separation)).any(-1).mean())

def contiguous_domain_split(contacts: np.ndarray, domain_count: int, minimum_domain_size: int) -> list[slice]:
    residues = len(contacts)
    domain_count = max(1, min(domain_count, residues // max(1, minimum_domain_size)))
    integral = np.pad(contacts.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = {0: (0, ())}
    for domain in range(domain_count):
        remaining = (domain_count - domain - 1) * minimum_domain_size
        boundaries: dict[int, tuple[int, tuple[int, ...]]] = {}
        for start, (contact_count, cuts) in best.items():
            for end in range(start + minimum_domain_size, residues - remaining + 1):
                within = int(integral[end, end] - integral[start, end] - integral[end, start] + integral[start, start])
                if end not in boundaries or contact_count + within > boundaries[end][0]:
                    boundaries[end] = (contact_count + within, cuts + (end,))
        best = boundaries
    return [slice(start, end) for start, end in zip((0,) + best[residues][1][:-1], best[residues][1])]

def binder_domain_geometry(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, binder: str, n_domains: int, min_domain_size: int, domain_contact_cutoff: float):
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    residues = int(real_residue_count(protein_states[prediction_state][binder].flags))
    coordinates, mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[binder])
    coordinates, mask = np.asarray(coordinates[:residues], dtype=np.float64), np.asarray(mask[:residues]) > 0
    if residues < 2 * min_domain_size or int(mask.sum()) < 2 * min_domain_size:
        return None
    contacts = (np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1) < domain_contact_cutoff) & mask[:, None] & mask[None, :]
    np.fill_diagonal(contacts, False)
    return coordinates, mask, contacts, contiguous_domain_split(contacts, n_domains, min_domain_size)

@filter_metric('Interdomain_Contact_Fraction')
def interdomain_contact_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', n_domains: int=2, min_domain_size: int=50, domain_contact_cutoff: float=8.0) -> float | None:
    geometry = binder_domain_geometry(protein_states, predictions, prediction_state, binder, n_domains, min_domain_size, domain_contact_cutoff)
    if geometry is None or len(geometry[3]) < 2:
        return None
    _, _, contacts, domains = geometry
    within = sum(int(contacts[domain, domain].sum()) for domain in domains)
    return float(contacts.sum() - within) / float(contacts.sum()) if contacts.sum() else 0.0

@filter_metric('Domain_Separation_Ratio')
def domain_separation_ratio_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', n_domains: int=2, min_domain_size: int=50, domain_contact_cutoff: float=8.0) -> float | None:
    geometry = binder_domain_geometry(protein_states, predictions, prediction_state, binder, n_domains, min_domain_size, domain_contact_cutoff)
    if geometry is None or len(geometry[3]) < 2:
        return None
    coordinates, mask, _, domains = geometry
    centroids = [coordinates[domain][mask[domain]].mean(0) for domain in domains]
    radii = [float(np.sqrt(np.square(coordinates[domain][mask[domain]] - centroid).sum(-1).mean())) for domain, centroid in zip(domains, centroids)]
    return min(float(np.linalg.norm(centroids[index] - centroids[index + 1])) / (radii[index] + radii[index + 1] + 1e-08) for index in range(len(domains) - 1))

@filter_metric('Binder_Chain_Breaks')
def binder_chain_breaks_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', minimum_bond: float=3.3, maximum_bond: float=4.3) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    breaks = None
    for chain in binder_copy_chains(protein_states[prediction_state], binder):
        residues = int(real_residue_count(protein_states[prediction_state][chain].flags))
        coordinates, mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
        coordinates, mask = np.asarray(coordinates[:residues], dtype=np.float64), np.asarray(mask[:residues]) > 0
        bonded = mask[:-1] & mask[1:]
        if not bonded.any():
            continue
        bond_lengths = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=-1)
        breaks = (breaks or 0.0) + float((bonded & ((bond_lengths < minimum_bond) | (bond_lengths > maximum_bond))).sum())
    return breaks

def protomer_identity_fraction(protein_complex: dict[str, Protein], binder: str) -> float | None:
    copies = binder_copy_chains(protein_complex, binder)
    protomers = [binder_one_letter_sequence(protein_complex[name]) for name in copies]
    if len(copies) < 2 or len({len(protomer) for protomer in protomers}) > 1:
        return None
    sequences = np.stack(protomers)
    return float((sequences == sequences[0]).all(0).mean())

@filter_metric('Oligomer_Symmetry_RMSD')
def oligomer_symmetry_rmsd_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    copies = binder_copy_chains(protein_states[prediction_state], binder)
    if len(copies) < 2:
        return None
    protein_complex = predictions[prediction_state].protein_complex
    coordinates = [chain_atom_coordinates(protein_complex[name]) for name in copies]
    if len({len(position) for position, _ in coordinates}) > 1:
        return None
    assembly = jnp.concatenate([position for position, _ in coordinates])
    rotated = jnp.concatenate([position for position, _ in coordinates[1:] + coordinates[:1]])
    valid_mask = jnp.concatenate([mask for _, mask in coordinates]) * jnp.concatenate([mask for _, mask in coordinates[1:] + coordinates[:1]])
    if float(valid_mask.sum()) < 3:
        return None
    squared_deviation = jnp.square(align_binder_coordinates(rotated, assembly, valid_mask) - assembly).sum(-1)
    return float(jnp.sqrt((squared_deviation * valid_mask).sum() / valid_mask.sum()))

@filter_metric('Receptor_Chains_Contacted')
def receptor_chains_contacted_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float | None:
    from bindcraft.protein_preparation import RECEPTOR_CHAIN_BREAK_GAP
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(predictions[prediction_state].protein_complex, target, prediction_state)
    residue_numbers = np.asarray(protein_states[prediction_state][target].residue_index)
    receptor_chain_index = np.concatenate([[0], np.cumsum(np.diff(residue_numbers) >= RECEPTOR_CHAIN_BREAK_GAP)])
    if receptor_chain_index[-1] < 1:
        return None
    contacted_target_residues = np.asarray(binder_assembly_contact_masks(predictions[prediction_state].protein_complex, binder, target, cutoff)[1])
    return float(len(np.unique(receptor_chain_index[contacted_target_residues])))

@filter_metric('Termini_Distance')
def termini_distance_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein = predictions[prediction_state].protein_complex[binder]
    c_terminus = int(real_residue_count(protein_states[prediction_state][binder].flags)) - 1
    coordinates, mask = chain_atom_coordinates(protein)
    if not (float(mask[0]) and float(mask[c_terminus])):
        return None
    return float(jnp.linalg.norm(coordinates[0] - coordinates[c_terminus]))

def terminus_away_cosine_metric(protein_states: ProteinStates, predictions: StructurePredictions, terminus: str='n', prediction_state: str='complex', binder: str='binder', target: str='target') -> float | None:
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, resolve_prediction_state(predictions, prediction_state), binder, target, terminus)
    return float(cosine) if float(valid) > 0 else None
filter_metric('N_Terminus_Away_Cosine')(functools.partial(terminus_away_cosine_metric, terminus='n'))
filter_metric('C_Terminus_Away_Cosine')(functools.partial(terminus_away_cosine_metric, terminus='c'))
filter_metric('Termini_Away_Cosine')(functools.partial(terminus_away_cosine_metric, terminus='both'))

@filter_metric('Target_Crop_Length')
def target_crop_length_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target') -> float:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return float(real_residue_count(protein_states[prediction_state][resolve_target_chain(protein_states[prediction_state], target, prediction_state)].flags))

@filter_metric('MHC_Anchor_Score')
def mhc_anchor_score_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', species: str='human', coupling_weight: float=0.5, hydrophobicity_weight: float=0.0, mhc_class_ii_weight: float=1.0, temperature: float=0.1) -> float:
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    binder_chains = binder_copy_chains(protein_complex, binder)
    worst_epitope = lambda panel: soft_maximum(jnp.stack([mhc_epitope_score(jax.nn.one_hot(protein_complex[name].sequence.argmax(-1), len(AMINO_ACIDS)), panel, coupling_weight, hydrophobicity_weight, temperature, redesignable_residue_mask(protein_complex[name].flags)) for name in binder_chains]), temperature)
    mhc_class_i_panel, mhc_class_ii_panel = mhc_panels(species)
    return float(worst_epitope(mhc_class_i_panel) + mhc_class_ii_weight * worst_epitope(mhc_class_ii_panel))

@filter_metric('Protease_Site_Score')
def protease_site_score_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float:
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    return float(pooled_protease_site_score([(jax.nn.one_hot(protein_complex[name].sequence.argmax(-1), len(AMINO_ACIDS)), redesignable_residue_mask(protein_complex[name].flags)) for name in binder_copy_chains(protein_complex, binder)]))

@filter_metric('Exposed_Loop_Fraction')
def exposed_loop_fraction_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    assembly = binder_copy_chains(protein_complex, binder)
    secondary_structure = binder_secondary_structure(protein_complex, binder)
    area = residue_solvent_accessible_area(protein_complex, assembly)
    if secondary_structure is None or area is None or len(secondary_structure) != len(area):
        return None
    loop_residues = ~np.isin(secondary_structure, ('a', 'b'))
    if not loop_residues.any():
        return 0.0
    maximum_area = np.array([MAX_RESIDUE_SASA[amino_acid] for chain in assembly for amino_acid in binder_one_letter_sequence(protein_complex[chain])])
    exposed = area / maximum_area >= SURFACE_RELATIVE_SASA
    return float((loop_residues & exposed).sum()) / float(loop_residues.sum())

@filter_metric('Terminus_Exposure')
def terminus_exposure_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', terminus_length: int=3) -> float | None:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    assembly = binder_copy_chains(protein_complex, binder)
    area = residue_solvent_accessible_area(protein_complex, assembly)
    sequence = np.concatenate([binder_one_letter_sequence(protein_complex[name]) for name in assembly])
    if area is None or len(area) != len(sequence):
        return None
    relative_area = area / np.array([MAX_RESIDUE_SASA[amino_acid] for amino_acid in sequence])
    terminal_rows, chain_start = [], 0
    for name in assembly:
        c_terminus = int(real_residue_count(protein_states[prediction_state][name].flags)) - 1
        if c_terminus < 0:
            return None
        terminal_rows.append(chain_start + np.concatenate([np.clip(np.arange(terminus_length), 0, c_terminus), np.clip(c_terminus - np.arange(terminus_length), 0, None)]))
        chain_start += len(protein_complex[name])
    return float(relative_area[np.concatenate(terminal_rows)].mean())

def interface_amino_acid_count(protein_states: ProteinStates, predictions: StructurePredictions, amino_acid: str='C', prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0) -> float:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    target = resolve_target_chain(protein_complex, target, prediction_state)
    interface_residues = np.asarray(binder_assembly_contact_masks(protein_complex, binder, target, cutoff)[0])
    binder_sequence = np.concatenate([binder_one_letter_sequence(protein_complex[name]) for name in binder_copy_chains(protein_complex, binder)])
    return float((interface_residues & (binder_sequence == amino_acid)).sum())

def paired_cysteine_count(deviation, pairable) -> int:
    bonded, bonds = set(), 0
    for first, second in sorted(zip(*np.nonzero(pairable)), key=lambda pair: deviation[pair]):
        if first not in bonded and second not in bonded:
            bonded.update((first, second))
            bonds += 1
    return bonds

@filter_metric('Binder_Disulfides')
def binder_disulfide_count_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', distance: float=3.8, tolerance: float=1.0, sequence_separation: int=3) -> float:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    chains = binder_copy_chains(protein_complex, binder)
    coordinates = np.concatenate([np.asarray(chain_atom_coordinates(protein_complex[name], 'CB')[0]) for name in chains])
    cysteines = np.concatenate([binder_one_letter_sequence(protein_complex[name]) == 'C' for name in chains])
    if cysteines.sum() < 2:
        return 0.0
    residue_indices = np.arange(len(cysteines))
    separated = np.abs(residue_indices[:, None] - residue_indices[None, :]) >= sequence_separation
    deviation = np.abs(np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1) - distance)
    pairable = np.triu(cysteines[:, None] & cysteines[None, :] & separated & (deviation <= tolerance))
    return float(paired_cysteine_count(deviation, pairable))

RESIDUE_MASS = {'A': 71.08, 'R': 156.19, 'N': 114.10, 'D': 115.09, 'C': 103.14, 'Q': 128.13, 'E': 129.12, 'G': 57.05, 'H': 137.14, 'I': 113.16, 'L': 113.16, 'K': 128.17, 'M': 131.19, 'F': 147.18, 'P': 97.12, 'S': 87.08, 'T': 101.10, 'W': 186.21, 'Y': 163.18, 'V': 99.13}
WATER_MASS = 18.02
TERMINUS_PKA = {'amino': 9.69, 'carboxyl': 2.34}
SIDE_CHAIN_PKA = {'D': 3.86, 'E': 4.25, 'C': 8.33, 'Y': 10.07, 'H': 6.00, 'K': 10.53, 'R': 12.48}
BASIC_AMINO_ACIDS = frozenset('HKR')
AROMATIC_EXTINCTION = {'W': 5500.0, 'Y': 1490.0}
CYSTINE_EXTINCTION = 125.0
REPORTED_PH = 7.4

def binder_assembly_sequence(protein_complex: dict[str, Protein], binder: str) -> str:
    return ''.join(''.join(binder_one_letter_sequence(protein_complex[name])) for name in binder_copy_chains(protein_complex, binder))

def binder_chain_sequences(protein_complex: dict[str, Protein], binder: str) -> str:
    return '/'.join(''.join(binder_one_letter_sequence(protein_complex[name])) for name in binder_copy_chains(protein_complex, binder))

def sequence_net_charge(sequence: str, ph: float) -> float:
    charge = 1.0 / (1.0 + 10.0 ** (ph - TERMINUS_PKA['amino'])) - 1.0 / (1.0 + 10.0 ** (TERMINUS_PKA['carboxyl'] - ph))
    for amino_acid, pka in SIDE_CHAIN_PKA.items():
        occurrences = sequence.count(amino_acid)
        charge += occurrences * (1.0 / (1.0 + 10.0 ** (ph - pka)) if amino_acid in BASIC_AMINO_ACIDS else -1.0 / (1.0 + 10.0 ** (pka - ph)))
    return charge

def sequence_isoelectric_point(sequence: str) -> float:
    acidic_ph, basic_ph = (0.0, 14.0)
    for _ in range(50):
        middle = (acidic_ph + basic_ph) / 2
        acidic_ph, basic_ph = (middle, basic_ph) if sequence_net_charge(sequence, middle) > 0 else (acidic_ph, middle)
    return (acidic_ph + basic_ph) / 2

def sequence_mass(sequence: str) -> float:
    return (sum(RESIDUE_MASS[amino_acid] for amino_acid in sequence) + WATER_MASS) / 1000.0

def sequence_charge_at_reported_ph(sequence: str) -> float:
    return sequence_net_charge(sequence, REPORTED_PH)

def sequence_cysteine_count(sequence: str) -> float:
    return float(sequence.count('C'))

def binder_sequence_reading(protein_states: ProteinStates, predictions: StructurePredictions, reading, prediction_state: str='complex', binder: str='binder') -> float | None:
    protein_complex = predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex
    sequence = binder_assembly_sequence(protein_complex, binder)
    return float(reading(sequence)) if sequence else None

filter_metric('Binder_Length')(functools.partial(binder_sequence_reading, reading=len))
filter_metric('Binder_Mass_kDa')(functools.partial(binder_sequence_reading, reading=sequence_mass))
filter_metric('Binder_pI')(functools.partial(binder_sequence_reading, reading=sequence_isoelectric_point))
filter_metric('Binder_Net_Charge')(functools.partial(binder_sequence_reading, reading=sequence_charge_at_reported_ph))
filter_metric('Binder_Cysteines')(functools.partial(binder_sequence_reading, reading=sequence_cysteine_count))

@filter_metric('Binder_Extinction')
def binder_extinction_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    protein_complex = predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex
    sequence = binder_assembly_sequence(protein_complex, binder)
    if not sequence:
        return None
    disulfides = binder_disulfide_count_metric(protein_states, predictions, prediction_state=prediction_state, binder=binder)
    return sum(sequence.count(amino_acid) * coefficient for amino_acid, coefficient in AROMATIC_EXTINCTION.items()) + CYSTINE_EXTINCTION * disulfides

@filter_metric('Binder_Free_Cysteines')
def binder_free_cysteine_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder') -> float | None:
    protein_complex = predictions[resolve_prediction_state(predictions, prediction_state)].protein_complex
    sequence = binder_assembly_sequence(protein_complex, binder)
    if not sequence:
        return None
    return max(0.0, sequence.count('C') - 2.0 * binder_disulfide_count_metric(protein_states, predictions, prediction_state=prediction_state, binder=binder))

def contacted_residue_names(protein: Protein, contact_mask) -> str:
    sequence = binder_one_letter_sequence(protein)
    residue_index = np.asarray(protein.residue_index)
    return ','.join(f'{sequence[position]}{int(residue_index[position])}' for position in np.flatnonzero(np.asarray(contact_mask)))

def contacted_target_residues(protein: Protein, contact_mask, receptor_chain_layout: tuple[tuple[str, int, int], ...]=()) -> str:
    """The epitope chain by chain, each receptor chain numbered as its own structure numbers it rather than as the fused target holds it."""
    if not receptor_chain_layout:
        return contacted_residue_names(protein, contact_mask)
    sequence, residue_index, contacts = binder_one_letter_sequence(protein), np.asarray(protein.residue_index), np.asarray(contact_mask)
    epitope, start = [], 0
    for _label, residue_count, first_residue in receptor_chain_layout:
        stop = min(start + residue_count, len(sequence))
        offset = first_residue - int(residue_index[start])
        epitope.append(','.join(f'{sequence[position]}{int(residue_index[position]) + offset}' for position in range(start, stop) if contacts[position]))
        start = stop
    return '/'.join(epitope)

def design_sequence_report(predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=4.0, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> dict[str, str]:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    target = resolve_target_chain(protein_complex, target, prediction_state)
    binder_chains = binder_copy_chains(protein_complex, binder)
    binder_contacts, target_contacts = binder_assembly_contact_masks(protein_complex, binder, target, cutoff)
    chain_offsets = np.cumsum([0] + [len(protein_complex[name]) for name in binder_chains])
    binder_contacts = np.asarray(binder_contacts)
    paratope = [contacted_residue_names(protein_complex[name], binder_contacts[start:end]) for name, start, end in zip(binder_chains, chain_offsets[:-1], chain_offsets[1:])]
    return {'Binder_Sequence': binder_chain_sequences(protein_complex, binder),
            'Interface_Binder_Residues': '/'.join(paratope),
            'Interface_Target_Residues': contacted_target_residues(protein_complex[target], target_contacts, (receptor_chains or {}).get(target, ()))}

for _amino_acid in AMINO_ACIDS:
    filter_metric(f'Interface_{_amino_acid}_Count')(functools.partial(interface_amino_acid_count, amino_acid=_amino_acid))

def state_filter_name(name: str, state_name: str, states: list[str], state_dependent: bool=True) -> str:
    return f'{name}.{state_name}' if state_dependent and len(states) > 1 else name

def design_stage_filters(design_settings: 'BinderDesignSettings', protein_states: ProteinStates, stage: str, iptm: bool=True, plddt: bool=True, campaign_filters: dict[str, DesignFilter] | None=None) -> dict[str, DesignFilter]:
    settings = design_settings.settings
    stage_filters = {}
    #read off the resolved states, for disordered-target crops
    detarget_states = {state.name for state in design_settings.prepared_states if state.objective == 'detarget'}
    complex_states = [name for name in protein_states if name != BINDER_ALONE] or [BINDER_ALONE]
    target_states = [name for name in complex_states if name not in detarget_states and name != BINDER_ALONE] or complex_states
    for state_name in complex_states:
        if iptm and state_name != BINDER_ALONE:
            is_detarget = state_name in detarget_states
            threshold = settings.get(f'max_detarget_iptm_{stage}' if is_detarget else f'min_iptm_{stage}')
            entry = settings.get('losses', {}).get('iptm_loss', {})
            bound_metric, required_states = bind_state_metric(iptm_metric, {**entry, 'prediction_state': state_name})
            stage_filters[state_filter_name('i_pTM', state_name, complex_states)] = DesignFilter(bound_metric, None if threshold is None else float(threshold), not is_detarget, required_states, threshold is not None)
    if plddt and settings.get(f'min_plddt_{stage}') is not None:
        for target_name in target_states:
            entry = settings.get('losses', {}).get('plddt_loss', {})
            bound_metric, required_states = bind_state_metric(plddt_metric, {**entry, 'params': resolve_binder_role(plddt_metric, entry.get('params', {}), design_settings.designed_binder_chain), 'prediction_state': target_name})
            stage_filters[state_filter_name('pLDDT', target_name, complex_states)] = DesignFilter(bound_metric, float(settings[f'min_plddt_{stage}']), True, required_states)
    for name, design_filter in (campaign_filters or {}).items():
        if design_filter.required_states:
            stage_filters[name] = design_filter
            continue
        #a check named for the off-target reads the states the campaign avoids, which is the only place its threshold means anything
        for target_name in sorted(detarget_states) if name.endswith('_detarget') else target_states:
            bound_metric, required_states = bind_state_metric(design_filter.function, {'prediction_state': target_name})
            stage_filters.setdefault(state_filter_name(name, target_name, complex_states, bool(required_states)), design_filter._replace(function=bound_metric, required_states=required_states))
    return stage_filters
