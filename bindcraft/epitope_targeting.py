import itertools
import jax.numpy as jnp
import numpy as np
from bindcraft.protein import AMINO_ACIDS, AMINO_ACID_INDEX, ATOM_INDEX, ATOM_NAMES, Protein, ProteinStates, ResidueFlags, has_residue_flag, target_chain_name

SURFACE_NEIGHBOUR_RADIUS, MAX_SURFACE_NEIGHBOURS = 10.0, 22.0
EPITOPE_CUTOFF = 10.0
CLASH_ALLOWANCE = 0.6
VAN_DER_WAALS_RADII = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
BACKBONE_ATOMS = 'N', 'CA', 'C', 'O', 'OXT'
LYSINE_SIDECHAIN_ATOMS = 'CB', 'CG', 'CD', 'CE', 'NZ'
LYSINE_SIDECHAIN_GEOMETRY = ('CG', 'N', 'CA', 'CB', 1.52, 114.1), ('CD', 'CA', 'CB', 'CG', 1.52, 111.3), ('CE', 'CB', 'CG', 'CD', 1.52, 111.3), ('NZ', 'CG', 'CD', 'CE', 1.489, 111.9)
LYSINE_REBUILT_ATOMS = tuple(name for name, *_ in LYSINE_SIDECHAIN_GEOMETRY)
LYSINE_ROTAMER_CHI_ANGLES = np.asarray(list(itertools.product((-67.0, -177.0, 62.0), (180.0, -65.0, 65.0), (180.0, -65.0, 65.0), (180.0, -65.0, 65.0))))
LYSINE_SIDECHAIN_RADII = np.asarray([VAN_DER_WAALS_RADII[name[0]] for name in LYSINE_REBUILT_ATOMS])
BACKBONE_ATOM_MASK = np.asarray([name in BACKBONE_ATOMS for name in ATOM_NAMES])
LYSINE_SIDECHAIN_MASK = np.asarray([name in LYSINE_SIDECHAIN_ATOMS for name in ATOM_NAMES])
REBUILT_SIDECHAIN_MASK = np.asarray([name not in BACKBONE_ATOMS and name != 'CB' for name in ATOM_NAMES])
ATOM_VDW_RADII = np.asarray([VAN_DER_WAALS_RADII.get(name[0], 1.7) for name in ATOM_NAMES])

def surface_exposed_residues(alpha_carbons: np.ndarray) -> np.ndarray:
    alpha_carbon_distances = np.linalg.norm(alpha_carbons[:, None, :] - alpha_carbons[None, :, :], axis=-1)
    return (alpha_carbon_distances < SURFACE_NEIGHBOUR_RADIUS).sum(-1) - 1 < MAX_SURFACE_NEIGHBOURS

def epitope_residues(atoms: np.ndarray, atom_mask: np.ndarray, hotspot_residues: np.ndarray, epitope_cutoff: float=EPITOPE_CUTOFF) -> np.ndarray:
    atom_coordinates, resolved_atom_mask = atoms.reshape(-1, 3), atom_mask.reshape(-1)
    hotspot_coordinates = atom_coordinates[np.repeat(hotspot_residues, atom_mask.shape[1]) & resolved_atom_mask]
    hotspot_distances = np.linalg.norm(atom_coordinates[:, None, :] - hotspot_coordinates[None, :, :], axis=-1).min(-1)
    return np.where(resolved_atom_mask, hotspot_distances, np.inf).reshape(atom_mask.shape).min(-1) <= epitope_cutoff

def place_sidechain_atom(first: np.ndarray, second: np.ndarray, third: np.ndarray, bond_length: float, bond_angle: float, torsion_angles: np.ndarray) -> np.ndarray:
    bond_angle, torsion_angles = np.radians(bond_angle), np.radians(torsion_angles)[..., None]
    bond_direction = (third - second) / np.linalg.norm(third - second, axis=-1, keepdims=True)
    plane_normal = np.cross(second - first, bond_direction)
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=-1, keepdims=True)
    return third + bond_length * (-bond_direction * np.cos(bond_angle) + np.cross(plane_normal, bond_direction) * (np.sin(bond_angle) * np.cos(torsion_angles)) + plane_normal * (np.sin(bond_angle) * np.sin(torsion_angles)))

def virtual_beta_carbon(nitrogen: np.ndarray, alpha_carbon: np.ndarray, carbon: np.ndarray) -> np.ndarray:
    amide_bond, carbonyl_bond = alpha_carbon - nitrogen, carbon - alpha_carbon
    return -0.58273431 * np.cross(amide_bond, carbonyl_bond) + 0.56802827 * amide_bond - 0.54067466 * carbonyl_bond + alpha_carbon

def build_lysine_sidechain(residue_atoms: np.ndarray, environment_coordinates: np.ndarray, environment_radii: np.ndarray) -> np.ndarray:
    placed_atoms = {name: residue_atoms[ATOM_INDEX[name]] for name in ('N', 'CA', 'CB')}
    for chi_index, (name, first, second, third, bond_length, bond_angle) in enumerate(LYSINE_SIDECHAIN_GEOMETRY):
        placed_atoms[name] = place_sidechain_atom(placed_atoms[first], placed_atoms[second], placed_atoms[third], bond_length, bond_angle, LYSINE_ROTAMER_CHI_ANGLES[:, chi_index])
    rotamer_atoms = np.stack([placed_atoms[name] for name in LYSINE_REBUILT_ATOMS], axis=1)
    rotamer_distances = np.linalg.norm(rotamer_atoms[:, :, None, :] - environment_coordinates[None, None, :, :], axis=-1)
    rotamer_clashes = np.maximum(0.0, LYSINE_SIDECHAIN_RADII[None, :, None] + environment_radii[None, None, :] - CLASH_ALLOWANCE - rotamer_distances)
    return rotamer_atoms[int(rotamer_clashes.sum((1, 2)).argmin())]

def lysinate_target(target: Protein, epitope_cutoff: float=EPITOPE_CUTOFF) -> Protein:
    atoms, atom_mask = np.array(target.atoms, dtype=np.float64), np.array(target.atom_mask, dtype=bool)
    hotspot_residues = np.asarray(has_residue_flag(target.flags, ResidueFlags.HOTSPOT))
    if not hotspot_residues.any():
        raise ValueError('Epitope-focused design needs target hotspots to define the region kept at wild type')
    if not all(atom_mask[:, ATOM_INDEX[name]].all() for name in ('N', 'CA', 'C')):
        raise ValueError('Epitope-focused design needs a target with a complete backbone to identify surface residues')
    epitope_residue_mask = epitope_residues(atoms, atom_mask, hotspot_residues, epitope_cutoff)
    lysinated_residues = surface_exposed_residues(atoms[:, ATOM_INDEX['CA']]) & ~epitope_residue_mask & (np.asarray(target.sequence.argmax(-1)) != AMINO_ACID_INDEX['K'])
    beta_carbons = np.where(atom_mask[:, ATOM_INDEX['CB'], None], atoms[:, ATOM_INDEX['CB']], virtual_beta_carbon(atoms[:, ATOM_INDEX['N']], atoms[:, ATOM_INDEX['CA']], atoms[:, ATOM_INDEX['C']]))
    atoms[lysinated_residues, ATOM_INDEX['CB']] = beta_carbons[lysinated_residues]
    atom_mask[lysinated_residues, ATOM_INDEX['CB']] = True
    environment_atom_mask = (atom_mask & ~(lysinated_residues[:, None] & REBUILT_SIDECHAIN_MASK)).reshape(-1)
    environment_residue_index = np.repeat(np.arange(len(target)), len(ATOM_NAMES))
    environment_radii = np.tile(ATOM_VDW_RADII, len(target))
    flat_atoms = atoms.reshape(-1, 3)
    for residue in np.flatnonzero(lysinated_residues):
        surrounding_atom_mask = environment_atom_mask & (environment_residue_index != residue)
        sidechain_atoms = build_lysine_sidechain(atoms[residue], flat_atoms[surrounding_atom_mask], environment_radii[surrounding_atom_mask])
        for name, position in zip(LYSINE_REBUILT_ATOMS, sidechain_atoms):
            atoms[residue, ATOM_INDEX[name]] = position
        atom_mask[residue] = (atom_mask[residue] & BACKBONE_ATOM_MASK) | LYSINE_SIDECHAIN_MASK
        environment_atom_mask[residue * len(ATOM_NAMES):(residue + 1) * len(ATOM_NAMES)] = atom_mask[residue]
    sequence = np.array(target.sequence, dtype=np.float32)
    sequence[lysinated_residues] = np.eye(len(AMINO_ACIDS), dtype=np.float32)[AMINO_ACID_INDEX['K']]
    print(f'forced targeting epitope={int(epitope_residue_mask.sum())} lysinated={int(lysinated_residues.sum())} residues={len(target)}', flush=True)
    return target.replace(sequence=jnp.asarray(sequence, dtype=target.sequence.dtype), atoms=jnp.asarray(atoms, dtype=target.atoms.dtype), atom_mask=jnp.asarray(atom_mask))

def force_epitope_targeting(protein_states: ProteinStates, settings: dict) -> ProteinStates:
    target_chain_prefix = settings.get('target_chain', 'target')
    return {name: {chain: lysinate_target(protein, settings.get('forced_targeting_shell', EPITOPE_CUTOFF)) if chain == target_chain_name(target_chain_prefix, name) else protein for chain, protein in protein_complex.items()} for name, protein_complex in protein_states.items()}

def restore_wild_type_targets(protein_states: ProteinStates, wild_type_states: ProteinStates, settings: dict) -> ProteinStates:
    target_chain_prefix = settings.get('target_chain', 'target')
    return {name: {chain: wild_type_states[name][chain] if chain == target_chain_name(target_chain_prefix, name) else protein for chain, protein in protein_complex.items()} for name, protein_complex in protein_states.items()}
