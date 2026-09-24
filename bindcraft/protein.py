import functools
import io
import os
import re
import warnings
import biotite.structure as _struc
import biotite.structure.io.pdb as _pdb
import biotite.structure.io.pdbx as _pdbx
import jax
import jax.numpy as jnp
import numpy as _np
import optax
from dataclasses import dataclass, replace
from enum import IntFlag
from itertools import accumulate
from typing import Any, NamedTuple
from jax import Array

AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'
AMINO_ACID_INDEX: dict[str, int] = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
THREE_LETTER_CODE: dict[str, str] = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
ONE_LETTER_CODE: dict[str, str] = {three_letter: one_letter for one_letter, three_letter in THREE_LETTER_CODE.items()}
MODIFIED_RESIDUE_PARENTS: dict[str, str] = {'MSE': 'MET'}
MODIFIED_RESIDUE_ATOM_NAMES: dict[tuple[str, str], str] = {('MSE', 'SE'): 'SD'}
BACKBONE_ATOM_NAMES: tuple[str, ...] = ('N', 'CA', 'C')
ATOM_NAMES: list[str] = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']
ATOM_INDEX: dict[str, int] = {atom_name: index for index, atom_name in enumerate(ATOM_NAMES)}
BINDER_ALONE = 'binder_alone'
BINDER_CHAIN_PREFIX = 'binder'
STAMP_FORMAT_VERSION = 3
QUALITY_METRIC_TYPES: dict[tuple[str, str], str] = {('pLDDT', 'local'): 'pLDDT', ('pLDDT', 'global'): 'pLDDT in [0,1]', ('Target_pLDDT', 'global'): 'pLDDT in [0,1]', ('pTM', 'global'): 'pTM', ('i_pTM', 'global'): 'ipTM'}

def target_chain_name(target_chain_prefix: str, state_name: str) -> str:
    return f'{target_chain_prefix}_{state_name}'

def is_target_chain(chain_name: str, target_chain_prefix: str) -> bool:
    return chain_name.startswith(target_chain_name(target_chain_prefix, ''))

def is_binder_chain(chain_name: str) -> bool:
    return chain_name == BINDER_CHAIN_PREFIX or chain_name.startswith(f'{BINDER_CHAIN_PREFIX}_')

class ResidueFlags(IntFlag):
    NONE = 0
    DESIGN = 1 << 0
    TEMPLATE = 1 << 1
    SEQUENCE = 1 << 2
    CONTACT = 1 << 3
    HOTSPOT = 1 << 4
    COLDSPOT = 1 << 5
    CYCLIC = 1 << 6
    PADDING = 1 << 7

def has_residue_flag(flags: Array, flag: ResidueFlags) -> Array:
    return flags & flag != 0

def real_residue_mask(flags: Array) -> Array:
    return ~has_residue_flag(flags, ResidueFlags.PADDING)

def redesignable_residue_mask(flags: Array) -> Array:
    designed = has_residue_flag(flags, ResidueFlags.DESIGN)
    return real_residue_mask(flags) & jnp.where(designed.any(), designed, True)

def real_residue_count(flags: Array) -> Array:
    return real_residue_mask(flags).sum()

def real_residue_weights(flags: Array) -> Array:
    return real_residue_mask(flags).astype(jnp.float32)

def has_resolved_atom(atom_mask: Array, name: str) -> Array:
    return atom_mask[:, ATOM_INDEX[name]]

def alignment_matrix_product(left: Array, right: Array) -> Array:
    return jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)

def kabsch(coordinates: Array, reference_coordinates: Array, alignment_weights: Array) -> tuple[Array, Array, Array]:
    alignment_weight_sum = alignment_weights.sum() + 1e-08
    coordinate_center = (coordinates * alignment_weights[:, None]).sum(0) / alignment_weight_sum
    reference_center = (reference_coordinates * alignment_weights[:, None]).sum(0) / alignment_weight_sum
    alignment_covariance = jnp.where(alignment_weights.sum() > 0, alignment_matrix_product(((coordinates - coordinate_center) * alignment_weights[:, None]).T, reference_coordinates - reference_center), jnp.eye(3, dtype=coordinates.dtype))
    left_vectors, _, right_vectors = jnp.linalg.svd(alignment_covariance)
    rotation_determinant = jnp.sign(jnp.linalg.det(alignment_matrix_product(right_vectors.T, left_vectors.T)))
    rotation = alignment_matrix_product(right_vectors.T * jnp.array([1.0, 1.0, rotation_determinant], dtype=coordinates.dtype), left_vectors.T)
    return rotation, coordinate_center, reference_center

RELAX_VDW_RADIUS_BY_ELEMENT: dict[str, float] = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
RELAX_BACKBONE_ATOM_NAMES: tuple[str, ...] = ('N', 'CA', 'C', 'O', 'OXT')
ATOM37_VDW_RADIUS: Array = jnp.array([RELAX_VDW_RADIUS_BY_ELEMENT.get(name[0], RELAX_VDW_RADIUS_BY_ELEMENT['C']) for name in ATOM_NAMES])
ATOM37_IS_BACKBONE: Array = jnp.array([name in RELAX_BACKBONE_ATOM_NAMES for name in ATOM_NAMES])
RELAX_ENERGY_TERMS: tuple[str, ...] = ('input_atoms', 'position_weights', 'bond_rows', 'bond_columns', 'bond_lengths', 'clashing_pairs', 'closest_approach', 'weight_bond', 'weight_clash')

#a restrained minimum settles short of the separation it is pushed towards, so the tolerance has to leave a relieved contact clear of the 2.5 A a clash is counted at
def default_relax_parameters() -> dict:
    return {'steps': 200, 'learning_rate': 0.02, 'restraint_backbone': 10.0, 'restraint_sidechain': 0.5,
            'weight_bond': 100.0, 'weight_clash': 5.0, 'overlap_tol': 0.4, 'min_sep': 2.5}

def relax_geometry(protein_complex: dict[str, 'Protein'], parameters: dict) -> dict:
    chain_letters = output_chain_letters(protein_complex)
    chains = sorted(protein_complex, key=chain_letters.__getitem__)
    atoms, chain_position, residue_number, atom_slot, resolved = [], [], [], [], []
    for position, chain in enumerate(chains):
        protein = protein_complex[chain]
        slots = len(protein) * len(ATOM_NAMES)
        atoms.append(_np.asarray(protein.atoms, dtype=_np.float64).reshape(slots, 3))
        resolved.append(_np.asarray(protein.atom_mask).reshape(slots))
        chain_position.append(_np.full(slots, position))
        residue_number.append(_np.repeat(_np.asarray(protein.residue_index), len(ATOM_NAMES)))
        atom_slot.append(_np.tile(_np.arange(len(ATOM_NAMES)), len(protein)))
    atoms, resolved = _np.concatenate(atoms), _np.concatenate(resolved)
    chain_position, residue_number, atom_slot = (_np.concatenate(rows)[resolved] for rows in (chain_position, residue_number, atom_slot))
    atoms = atoms[resolved]
    radii = _np.asarray(ATOM37_VDW_RADIUS)[atom_slot]
    distances = _np.sqrt(_np.maximum(_np.sum(_np.square(atoms[:, None, :] - atoms[None, :, :]), -1), 0.0))
    _np.fill_diagonal(distances, _np.inf)
    same_chain = chain_position[:, None] == chain_position[None, :]
    same_residue = same_chain & (residue_number[:, None] == residue_number[None, :])
    neighbouring_residues = same_chain & (_np.abs(residue_number[:, None] - residue_number[None, :]) <= 1)
    sulfur = (atom_slot == ATOM_INDEX['SG']) | (atom_slot == ATOM_INDEX['SD'])
    disulfide = sulfur[:, None] & sulfur[None, :] & (distances < 2.5)
    carbonyl, amide_nitrogen = atom_slot == ATOM_INDEX['C'], atom_slot == ATOM_INDEX['N']
    amide = ((carbonyl[:, None] & amide_nitrogen[None, :]) | (amide_nitrogen[:, None] & carbonyl[None, :])) & (distances < 1.9)
    bonded = disulfide | amide | (same_residue & (distances < 1.9))
    rows, columns = _np.triu_indices(len(atoms), k=1)
    bond = bonded[rows, columns]
    clashing_pairs = ~neighbouring_residues & ~(disulfide | amide)
    _np.fill_diagonal(clashing_pairs, False)
    return {'input_atoms': jnp.asarray(atoms, dtype=jnp.float32),
            'position_weights': jnp.asarray(_np.where(_np.asarray(ATOM37_IS_BACKBONE)[atom_slot], parameters['restraint_backbone'], parameters['restraint_sidechain']), dtype=jnp.float32),
            'bond_rows': jnp.asarray(rows[bond]), 'bond_columns': jnp.asarray(columns[bond]),
            'bond_lengths': jnp.asarray(distances[rows, columns][bond], dtype=jnp.float32),
            'clashing_pairs': jnp.asarray(clashing_pairs, dtype=jnp.float32),
            'closest_approach': jnp.asarray(_np.maximum(radii[:, None] + radii[None, :] - parameters['overlap_tol'], parameters['min_sep']), dtype=jnp.float32),
            'weight_bond': float(parameters['weight_bond']), 'weight_clash': float(parameters['weight_clash']),
            'resolved': resolved, 'chains': chains}

def relax_energy(atoms: Array, geometry: dict) -> Array:
    distances = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(atoms[:, None, :] - atoms[None, :, :]), -1), 0.0) + 1e-06)
    overlap = jnp.maximum(0.0, geometry['closest_approach'] - distances)
    bond_lengths = jnp.sqrt(jnp.sum(jnp.square(atoms[geometry['bond_rows']] - atoms[geometry['bond_columns']]), -1) + 1e-06)
    return (0.5 * geometry['weight_clash'] * jnp.sum(geometry['clashing_pairs'] * overlap * overlap)
            + geometry['weight_bond'] * jnp.sum(jnp.square(bond_lengths - geometry['bond_lengths']))
            + jnp.sum(geometry['position_weights'] * jnp.sum(jnp.square(atoms - geometry['input_atoms']), -1)))

def relax_protein_complex(protein_complex: dict[str, 'Protein'], parameters: dict | None=None) -> dict[str, 'Protein']:
    parameters = {**default_relax_parameters(), **(parameters or {})}
    geometry = relax_geometry(protein_complex, parameters)
    energy_gradient = jax.jit(jax.grad(relax_energy))
    energy_terms = {term: geometry[term] for term in RELAX_ENERGY_TERMS}
    atoms = geometry['input_atoms']
    optimizer = optax.adam(parameters['learning_rate'])
    optimizer_state = optimizer.init(atoms)
    for _ in range(int(parameters['steps'])):
        updates, optimizer_state = optimizer.update(energy_gradient(atoms, energy_terms), optimizer_state)
        atoms = optax.apply_updates(atoms, updates)
    relaxed_slots = _np.zeros((len(geometry['resolved']), 3), dtype=_np.float32)
    relaxed_slots[geometry['resolved']] = _np.asarray(atoms)
    relaxed_complex, slot = {}, 0
    for chain in geometry['chains']:
        protein = protein_complex[chain]
        chain_atoms = relaxed_slots[slot:slot + len(protein) * len(ATOM_NAMES)].reshape(len(protein), len(ATOM_NAMES), 3)
        slot += len(protein) * len(ATOM_NAMES)
        kept = _np.where(_np.asarray(protein.atom_mask)[..., None], chain_atoms, _np.asarray(protein.atoms, dtype=_np.float32))
        relaxed_complex[chain] = protein.replace(atoms=jnp.asarray(kept, dtype=protein.atoms.dtype))
    return relaxed_complex

@dataclass
class Protein:
    sequence: Array
    atoms: Array
    atom_mask: Array
    flags: Array
    residue_index: Array

    def __len__(self) -> int:
        return int(self.sequence.shape[0])

    def replace(self, **kwargs: Any) -> 'Protein':
        return replace(self, **kwargs)

    def padded_to(self, length: int) -> 'Protein':
        padding = length - len(self)
        if padding <= 0:
            return self
        return self.replace(sequence=jnp.pad(self.sequence, ((0, padding), (0, 0))), atoms=jnp.pad(self.atoms, ((0, padding), (0, 0), (0, 0))), atom_mask=jnp.pad(self.atom_mask, ((0, padding), (0, 0))), flags=jnp.pad(self.flags, (0, padding), constant_values=int(ResidueFlags.PADDING)), residue_index=jnp.concatenate([self.residue_index, self.residue_index[-1] + 1 + jnp.arange(padding, dtype=self.residue_index.dtype)]))

    def trimmed_to(self, length: int) -> 'Protein':
        return self if length >= len(self) else self.replace(sequence=self.sequence[:length], atoms=self.atoms[:length], atom_mask=self.atom_mask[:length], flags=self.flags[:length], residue_index=self.residue_index[:length])

    @staticmethod
    def unresolved_atom_arrays(length: int) -> tuple[Array, Array, Array]:
        return jnp.zeros((length, len(ATOM_NAMES), 3), dtype=jnp.float16), jnp.zeros((length, len(ATOM_NAMES)), dtype=bool), jnp.arange(1, length+1, dtype=jnp.int32)

    @staticmethod
    def empty(length: int, key: Array) -> 'Protein':
        atoms, atom_mask, residue_index = Protein.unresolved_atom_arrays(length)
        return Protein(sequence=0.01 * jax.random.normal(key, (length, len(AMINO_ACIDS)), dtype=jnp.float16), atoms=atoms, atom_mask=atom_mask, flags=jnp.full((length,), ResidueFlags.DESIGN, dtype=jnp.uint8), residue_index=residue_index)

    @staticmethod
    def from_fasta(source: str, chain_letter: str='A', flags: str='') -> 'Protein':
        records = read_fasta_sequences(source)
        if chain_letter in records:
            amino_acid_sequence = records[chain_letter]
        elif len(records) == 1:
            amino_acid_sequence, = records.values()
        else:
            raise ValueError(f'from_fasta: no record {chain_letter!r} in source (found {sorted(records)})')
        amino_acid_sequence = amino_acid_sequence.upper()
        invalid_amino_acids = sorted(set(amino_acid_sequence) - set(AMINO_ACIDS))
        if invalid_amino_acids:
            raise ValueError(f'from_fasta: non-standard amino acid(s) {invalid_amino_acids} in sequence')
        aatype = [AMINO_ACID_INDEX[amino_acid] for amino_acid in amino_acid_sequence]
        atoms, atom_mask, residue_index = Protein.unresolved_atom_arrays(len(aatype))
        return Protein(sequence=jax.nn.one_hot(jnp.asarray(aatype), len(AMINO_ACIDS), dtype=jnp.float16), atoms=atoms, atom_mask=atom_mask, flags=jnp.asarray(annotated_residue_flags(residue_index.tolist(), flags, chain_letter, ResidueFlags.SEQUENCE), dtype=jnp.uint8), residue_index=residue_index)

    @staticmethod
    def from_structure(source: str, chains: str | None=None, flags: str='') -> dict[str, 'Protein']:
        records = read_structure_atoms(source)
        if chains is not None:
            selected_chains = set(selected_chain_names(chains, [atom_record['chain_id'] for atom_record in records]))
            records = [atom_record for atom_record in records if atom_record['chain_id'] in selected_chains]
        records = polymer_atom_records(records, source)
        protein_chains: dict[str, Protein] = {}
        for chain_id, chain_records in group_atoms_by_chain(records).items():
            atom_coordinates, atom_mask, aatype, residue_index = atom37_arrays_from_records(chain_records)
            protein_chains[chain_id] = Protein(sequence=jax.nn.one_hot(jnp.asarray(aatype), len(AMINO_ACIDS), dtype=jnp.float16), atoms=jnp.asarray(atom_coordinates, dtype=jnp.float16), atom_mask=jnp.asarray(atom_mask, dtype=bool), flags=jnp.asarray(annotated_residue_flags(residue_index, flags, chain_id, ResidueFlags.TEMPLATE | ResidueFlags.SEQUENCE), dtype=jnp.uint8), residue_index=jnp.asarray(residue_index, dtype=jnp.int32))
        return protein_chains

    def with_scaffold(self, chain_letter: str, edit_string: str, key: Array) -> 'Protein':
        length_random_key, sequence_random_key = jax.random.split(key)
        scaffold_edits = parse_scaffold_edits(edit_string, chain_letter, length_random_key)
        if not scaffold_edits:
            return self
        scaffold_edits.sort(key=lambda scaffold_edit: scaffold_edit[0])
        residue_index = [int(residue_number) for residue_number in self.residue_index]
        sequence, residue_flags = self.sequence.tolist(), [int(residue_flag) for residue_flag in self.flags]
        atom_positions, atom_mask = self.atoms.tolist(), self.atom_mask.tolist()
        unresolved_atom_positions, unresolved_atom_mask = [[0.0] * 3] * len(ATOM_NAMES), [False] * len(ATOM_NAMES)
        designed_residue_count = sum(scaffold_edit[2] for scaffold_edit in scaffold_edits)
        designed_sequence_logits = (0.01 * jax.random.normal(sequence_random_key, (designed_residue_count, len(AMINO_ACIDS)))).tolist()
        designed_residue_position = 0
        edited_residue_indices, edited_sequence, edited_atom_positions, edited_atom_mask, edited_residue_flags = [], [], [], [], []
        scaffold_residue_position, residue_number_shift = 0, 0
        for start, end, replacement_length, edit_flags in scaffold_edits:
            while scaffold_residue_position < len(residue_index) and residue_index[scaffold_residue_position] < start:
                edited_residue_indices.append(residue_index[scaffold_residue_position] + residue_number_shift)
                edited_sequence.append(sequence[scaffold_residue_position])
                edited_atom_positions.append(atom_positions[scaffold_residue_position])
                edited_atom_mask.append(atom_mask[scaffold_residue_position])
                edited_residue_flags.append(residue_flags[scaffold_residue_position])
                scaffold_residue_position += 1
            replacement_residue_start = start + residue_number_shift
            while scaffold_residue_position < len(residue_index) and residue_index[scaffold_residue_position] <= end:
                scaffold_residue_position += 1
            for replacement_residue_offset in range(replacement_length):
                edited_residue_indices.append(replacement_residue_start + replacement_residue_offset)
                edited_sequence.append(designed_sequence_logits[designed_residue_position])
                edited_atom_positions.append(unresolved_atom_positions)
                edited_atom_mask.append(unresolved_atom_mask)
                edited_residue_flags.append(int(edit_flags))
                designed_residue_position += 1
            residue_number_shift += replacement_length - (end - start + 1)
        while scaffold_residue_position < len(residue_index):
            edited_residue_indices.append(residue_index[scaffold_residue_position] + residue_number_shift)
            edited_sequence.append(sequence[scaffold_residue_position])
            edited_atom_positions.append(atom_positions[scaffold_residue_position])
            edited_atom_mask.append(atom_mask[scaffold_residue_position])
            edited_residue_flags.append(residue_flags[scaffold_residue_position])
            scaffold_residue_position += 1
        return Protein(sequence=jnp.asarray(edited_sequence, dtype=jnp.float16), atoms=jnp.asarray(edited_atom_positions, dtype=jnp.float16), atom_mask=jnp.asarray(edited_atom_mask, dtype=bool), flags=jnp.asarray(edited_residue_flags, dtype=jnp.uint8), residue_index=jnp.asarray(edited_residue_indices, dtype=jnp.int32))

@dataclass
class StructurePrediction:
    protein_complex: dict[str, Protein]
    metrics: dict[str, Array | float | int]
ProteinStates = dict[str, dict[str, Protein]]
StructurePredictions = dict[str, StructurePrediction]

STRUCTURE_PATH_SUFFIXES = ('.pdb', '.cif', '.mmcif', '.ent', '.fasta', '.fa', '.faa')

@functools.lru_cache(maxsize=32)
def read_protein_source(source: str) -> str:
    if '\n' not in source and len(source) < 4096:
        try:
            with open(source) as fh:
                return fh.read()
        except OSError:
            if os.sep in source or source.lower().endswith(STRUCTURE_PATH_SUFFIXES):
                raise
    return source

def read_fasta_sequences(source: str) -> dict[str, str]:
    text = read_protein_source(source)
    records: dict[str, list] = {}
    record_id = ''
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            tokens = line[1:].split()
            record_id = tokens[0] if tokens else ''
            records.setdefault(record_id, [])
        else:
            records.setdefault(record_id, []).append(line)
    return {record_id: ''.join(sequence_lines) for record_id, sequence_lines in records.items()}

def is_mmcif_structure(source: str, text: str) -> bool:
    if source.lower().endswith(('.cif', '.mmcif')):
        return True
    if source.lower().endswith('.pdb'):
        return False
    for line in text.splitlines():
        structure_line = line.strip()
        if structure_line.startswith(('data_', 'loop_', '_')):
            return True
        if structure_line.startswith(('ATOM', 'HETATM')):
            return False
    return False

@functools.lru_cache(maxsize=16)
def read_structure_atoms(source: str) -> tuple[dict, ...]:
    text = read_protein_source(source)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r".*not found within 'atom_site' category.*", category=UserWarning)
        warnings.filterwarnings('ignore', message=r'\d+ elements were guessed from atom name', category=UserWarning)
        if is_mmcif_structure(source, text):
            block = _pdbx.CIFFile.read(io.StringIO(text)).block
            structure = _pdbx.get_structure(block, model=1, extra_fields=['b_factor'])
        else:
            structure = _pdb.PDBFile.read(io.StringIO(text)).get_structure(model=1, extra_fields=['b_factor'])
    b_factors = structure.b_factor if 'b_factor' in structure.get_annotation_categories() else _np.zeros(structure.array_length())
    return tuple({'name': str(structure.atom_name[atom_index]), 'res_name': str(structure.res_name[atom_index]), 'res_id': int(structure.res_id[atom_index]), 'chain_id': str(structure.chain_id[atom_index]), 'insertion_code': str(structure.ins_code[atom_index]).strip(), 'is_hetero': bool(structure.hetero[atom_index]), 'b_factor': float(b_factors[atom_index]), 'coord': [float(coordinate) for coordinate in structure.coord[atom_index]]} for atom_index in range(structure.array_length()))

def structure_source_description(source: str) -> str:
    return source if '\n' not in source and len(source) < 200 else '<inline structure>'

def structure_chain_labels(source: str) -> list[str]:
    return list(dict.fromkeys(atom_record['chain_id'] for atom_record in read_structure_atoms(source)))

def structure_chain_names(source: str) -> list[str]:
    return list(dict.fromkeys(atom_record['chain_id'] for atom_record in polymer_atom_records(read_structure_atoms(source), source)))

def selected_chain_names(chains: str, available_chain_names: list[str]) -> list[str]:
    selected, structure_chains = [], set(available_chain_names)
    for requested_name in chains.split(','):
        requested_name = requested_name.strip()
        if requested_name:
            selected.extend([requested_name] if len(requested_name) == 1 or requested_name in structure_chains else list(requested_name))
    return selected
CHAIN_LETTERS = [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]

def output_chain_letters(chain_names) -> dict[str, str]:
    return dict(zip(sorted(chain_names, key=lambda chain_name: (is_binder_chain(chain_name), chain_name)), CHAIN_LETTERS))

class WrittenChain(NamedTuple):
    """One chain of a written structure: which chain of the complex it came from, and the residues and numbering it carries."""
    complex_chain: str
    label: str
    letter: str
    start: int
    stop: int
    residue_offset: int

def written_chains(protein_complex: dict[str, Protein], receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> list[WrittenChain]:
    """Every chain an output holds, in written order.

    A target spanning several receptor chains is carried through the campaign as one chain, because the
    losses and the filters address one target, and is written back as the chains the receptor is: a
    reader sees two molecules rather than one with a peptide bond across the space between them, and
    each chain keeps the numbering its own structure gave it."""
    segments = []
    for chain in sorted(protein_complex, key=lambda name: (is_binder_chain(name), name)):
        residue_index, residues = protein_complex[chain].residue_index, len(protein_complex[chain])
        layout, start = (receptor_chains or {}).get(chain, ()), 0
        for label, residue_count, first_residue in layout:
            if start >= residues:
                break
            stop = min(start + residue_count, residues)
            segments.append((chain, label, start, stop, first_residue - int(residue_index[start])))
            start = stop
        if start < residues:
            #trailing padding, and any chain the campaign never fused, keep the numbering they are held under
            segments.append((chain, '', start, residues, 0) if start else (chain, '', 0, residues, 0))
    return [WrittenChain(chain, label, CHAIN_LETTERS[position], start, stop, offset) for position, (chain, label, start, stop, offset) in enumerate(segments)]

def build_atom_array(protein_complex: dict[str, Protein], plddt: Array | None=None, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None):
    atom_names, residue_names, residue_ids, chain_ids, atom_coordinates, elements, b_factors = [], [], [], [], [], [], []
    residue_offsets = dict(zip(sorted(protein_complex), accumulate((len(protein_complex[chain]) for chain in sorted(protein_complex)), initial=0)))
    for written in written_chains(protein_complex, receptor_chains):
        chain, chain_letter = written.complex_chain, written.letter
        protein = protein_complex[chain]
        aatype = _np.asarray(protein.sequence.argmax(-1))
        atom_mask = _np.asarray(protein.atom_mask)
        atoms = _np.asarray(protein.atoms)
        residue_index = _np.asarray(protein.residue_index)
        chain_plddt = _np.asarray(plddt[residue_offsets[chain]:residue_offsets[chain] + len(protein)]) if plddt is not None else None
        for residue_position in range(written.start, written.stop):
            residue_name = THREE_LETTER_CODE[AMINO_ACIDS[int(aatype[residue_position])]]
            for atom_index, atom_name in enumerate(ATOM_NAMES):
                if not atom_mask[residue_position, atom_index]:
                    continue
                atom_names.append(atom_name)
                residue_names.append(residue_name)
                residue_ids.append(int(residue_index[residue_position]) + written.residue_offset)
                chain_ids.append(chain_letter)
                atom_coordinates.append(atoms[residue_position, atom_index])
                elements.append(atom_name[0])
                b_factors.append(float(chain_plddt[residue_position]) * 100.0 if chain_plddt is not None else 0.0)
    if not atom_names:
        return None
    atom_array = _struc.AtomArray(len(atom_names))
    atom_array.coord = _np.asarray(atom_coordinates, dtype=_np.float32).reshape(-1, 3)
    atom_array.atom_name = _np.asarray(atom_names)
    atom_array.res_name = _np.asarray(residue_names)
    atom_array.res_id = _np.asarray(residue_ids)
    atom_array.chain_id = _np.asarray(chain_ids)
    atom_array.element = _np.asarray(elements)
    atom_array.set_annotation('b_factor', _np.asarray(b_factors, dtype=_np.float32))
    atom_array.bonds = _struc.connect_via_residue_names(atom_array)
    for first_atom, last_atom in cyclic_backbone_closures(protein_complex, atom_array, receptor_chains):
        atom_array.bonds.add_bond(last_atom, first_atom, _struc.BondType.SINGLE)
    return atom_array

def cyclic_backbone_closures(protein_complex: dict[str, Protein], atom_array, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> list[tuple[int, int]]:
    closures = []
    for written in written_chains(protein_complex, receptor_chains):
        chain = written.complex_chain
        real_residues = real_residue_mask(protein_complex[chain].flags)
        cyclic_residues = has_residue_flag(protein_complex[chain].flags, ResidueFlags.CYCLIC)
        if not bool(real_residues.any()) or not bool(cyclic_residues[real_residues].all()):
            continue
        chain_atoms = _np.flatnonzero(atom_array.chain_id == written.letter)
        if not len(chain_atoms):
            continue
        residue_ids = atom_array.res_id[chain_atoms]
        nitrogen = chain_atoms[(residue_ids == residue_ids.min()) & (atom_array.atom_name[chain_atoms] == 'N')]
        carbon = chain_atoms[(residue_ids == residue_ids.max()) & (atom_array.atom_name[chain_atoms] == 'C')]
        if len(nitrogen) and len(carbon):
            closures.append((int(nitrogen[0]), int(carbon[0])))
    return closures

def written_residue_values(protein_complex: dict[str, Protein], values: _np.ndarray, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> _np.ndarray:
    residue_offsets = dict(zip(sorted(protein_complex), accumulate((len(protein_complex[chain]) for chain in sorted(protein_complex)), initial=0)))
    written = []
    for written_chain in written_chains(protein_complex, receptor_chains):
        atom_mask = _np.asarray(protein_complex[written_chain.complex_chain].atom_mask)
        written.extend(residue_offsets[written_chain.complex_chain] + position for position in range(written_chain.start, written_chain.stop) if atom_mask[position].any())
    return _np.asarray(values)[written]

def declared_quality_metric(definitions: dict, name: str, mode: str) -> str:
    definitions['id'].append(str(len(definitions['id']) + 1))
    definitions['name'].append(name)
    definitions['type'].append(QUALITY_METRIC_TYPES.get((name.partition('.')[0], mode), 'other'))
    definitions['mode'].append(mode)
    return definitions['id'][-1]

def recorded_number(value: float) -> float:
    number = float(value)
    return round(number, 2 if abs(number) <= 1 else 1)

def quality_metric_categories(atom_array, protein_complex: dict[str, Protein], residue_metrics: dict, scalar_metrics: dict, label_positions: dict, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> dict:
    residue_starts = _struc.get_residue_starts(atom_array)
    tracks = {name: written_residue_values(protein_complex, values, receptor_chains) for name, values in residue_metrics.items()}
    tracks = {name: values for name, values in tracks.items() if len(values) == len(residue_starts)}
    definitions = {'id': [], 'name': [], 'type': [], 'mode': []}
    local = {field: [] for field in ('label_asym_id', 'label_comp_id', 'label_seq_id', 'ordinal_id', 'metric_id', 'metric_value', 'model_id')}
    model_wide = {field: [] for field in ('ordinal_id', 'metric_id', 'metric_value', 'model_id')}
    for name, values in tracks.items():
        metric_id = declared_quality_metric(definitions, name, 'local')
        for residue_position, atom_index in enumerate(residue_starts):
            value = float(values[residue_position])
            if value != value:
                continue
            local['label_asym_id'].append(str(atom_array.chain_id[atom_index]))
            local['label_comp_id'].append(str(atom_array.res_name[atom_index]))
            local['label_seq_id'].append(str(label_positions[str(atom_array.chain_id[atom_index]), int(atom_array.res_id[atom_index])]))
            local['ordinal_id'].append(str(len(local['ordinal_id']) + 1))
            local['metric_id'].append(metric_id)
            local['metric_value'].append(str(recorded_number(value)))
            local['model_id'].append('1')
    for name, value in scalar_metrics.items():
        model_wide['ordinal_id'].append(str(len(model_wide['ordinal_id']) + 1))
        model_wide['metric_id'].append(declared_quality_metric(definitions, name, 'global'))
        model_wide['metric_value'].append(str(recorded_number(value)))
        model_wide['model_id'].append('1')
    return {category: columns for category, columns in (('ma_qa_metric', definitions), ('ma_qa_metric_local', local), ('ma_qa_metric_global', model_wide)) if any(columns.values())}

TARGET_FIT_CUTOFF = 4.0
TARGET_FIT_MINIMUM_CORE = 0.4
TARGET_FIT_MINIMUM_IDENTITY = 0.5

def residue_offset_correspondence(sequence: Array, reference_sequence: Array) -> tuple[Array, Array]:
    amino_acids, reference_amino_acids = _np.asarray(sequence.argmax(-1)), _np.asarray(reference_sequence.argmax(-1))
    overlap = lambda shift: (max(0, -shift), min(len(amino_acids), len(reference_amino_acids) - shift))
    identity = lambda shift: int((amino_acids[slice(*overlap(shift))] == reference_amino_acids[overlap(shift)[0] + shift:overlap(shift)[1] + shift]).sum())
    shifts = [shift for shift in range(1 - len(amino_acids), len(reference_amino_acids)) if overlap(shift)[1] - overlap(shift)[0] >= 3]
    if not shifts:
        return _np.arange(0), _np.arange(0)
    shift = max(shifts, key=identity)
    positions = _np.arange(*overlap(shift))
    return positions, positions + shift

def target_fit_quality(target: Protein, reference: Protein) -> tuple[float, float, Array, Array, Array] | None:
    alpha_carbon = ATOM_INDEX['CA']
    positions, reference_positions = residue_offset_correspondence(target.sequence, reference.sequence)
    if len(positions) < 3:
        return None
    paired = target.atom_mask[positions, alpha_carbon] & reference.atom_mask[reference_positions, alpha_carbon]
    coordinates = target.atoms[positions, alpha_carbon].astype(jnp.float32)
    reference_coordinates = reference.atoms[reference_positions, alpha_carbon].astype(jnp.float32)
    core, fit = paired, None
    for _ in range(8):
        if int(core.sum()) < 3:
            return None
        fit = kabsch(coordinates, reference_coordinates, core.astype(jnp.float32))
        deviation = jnp.linalg.norm((coordinates - fit[1]) @ fit[0].T + fit[2] - reference_coordinates, axis=-1)
        tighter = paired & (deviation < TARGET_FIT_CUTOFF)
        if int(tighter.sum()) < 3 or bool((tighter == core).all()):
            break
        core = tighter
    same_residue = target.sequence.argmax(-1)[positions] == reference.sequence.argmax(-1)[reference_positions]
    held = int(core.sum()) / max(1, min(len(target), len(reference)))
    identity = float((same_residue & core).sum()) / max(1, int(core.sum()))
    return held, identity, *fit

def superposed_on_reference_target(target: Protein, reference: Protein) -> Protein | None:
    quality = target_fit_quality(target, reference)
    if quality is None:
        return None
    held, identity, rotation, centre, reference_centre = quality
    if held < TARGET_FIT_MINIMUM_CORE or identity < TARGET_FIT_MINIMUM_IDENTITY:
        return None
    moved = (target.atoms.astype(jnp.float32) - centre) @ rotation.T + reference_centre
    return target.replace(atoms=jnp.where(target.atom_mask[:, :, None], moved, 0.0).astype(target.atoms.dtype))

def superposed_on_binder(protein_complex: dict[str, Protein], reference_complex: dict[str, Protein]) -> dict[str, Protein]:
    binder_chains = [name for name in sorted(protein_complex) if is_binder_chain(name) and name in reference_complex and len(protein_complex[name]) == len(reference_complex[name])]
    if not binder_chains:
        return protein_complex
    alpha_carbon = ATOM_INDEX['CA']
    chain_pairs = [(protein_complex[name], reference_complex[name]) for name in binder_chains]
    coordinates = jnp.concatenate([protein.atoms[:, alpha_carbon] for protein, _ in chain_pairs]).astype(jnp.float32)
    reference_coordinates = jnp.concatenate([reference.atoms[:, alpha_carbon] for _, reference in chain_pairs]).astype(jnp.float32)
    alignment_weights = jnp.concatenate([real_residue_mask(protein.flags) & protein.atom_mask[:, alpha_carbon] & reference.atom_mask[:, alpha_carbon] for protein, reference in chain_pairs]).astype(jnp.float32)
    if float(alignment_weights.sum()) < 3:
        return protein_complex
    rotation, complex_center, reference_center = kabsch(coordinates, reference_coordinates, alignment_weights)
    return {name: protein.replace(atoms=jnp.where(protein.atom_mask[:, :, None], (protein.atoms.astype(jnp.float32) - complex_center) @ rotation.T + reference_center, 0.0).astype(protein.atoms.dtype)) for name, protein in protein_complex.items()}

def write_structure(protein_complex: dict[str, Protein], path: str, plddt: Array | None=None, metadata: dict | None=None, residue_metrics: dict | None=None, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> None:
    atom_array = build_atom_array(protein_complex, plddt, receptor_chains)
    if atom_array is None:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if path.lower().endswith('.pdb'):
        pdb_file = _pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        pdb_file.write(path)
    else:
        cif_file = _pdbx.CIFFile()
        _pdbx.set_structure(cif_file, atom_array, data_block='model')
        label_positions = label_sequence_positions(atom_array)
        renumber_label_sequence_ids(cif_file['model'], label_positions)
        address_bond_partners(cif_file['model'], label_positions)
        declare_polymer_entities(cif_file['model'], atom_array, protein_complex, receptor_chains)
        #a reader, and bindcraft score, should not have to guess which written chain is the binder
        roles = written_chain_roles(protein_complex, receptor_chains)
        cif_file['model']['bindcraft'] = _pdbx.CIFCategory({stamp_keyword(name): str(value) for name, value in {'stamp_format_version': STAMP_FORMAT_VERSION, **roles, **(metadata or {})}.items()})
        scalar_metrics = {name: value for name, value in (metadata or {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        for category, columns in quality_metric_categories(atom_array, protein_complex, residue_metrics or {}, scalar_metrics, label_positions, receptor_chains).items():
            cif_file['model'][category] = _pdbx.CIFCategory(columns)
        cif_file.write(path)

def written_chain_roles(protein_complex: dict[str, Protein], receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> dict[str, str]:
    """Which written chains are the designed binder and which are the target it was designed against."""
    written = written_chains(protein_complex, receptor_chains)
    roles = {'binder_chains': ','.join(chain.letter for chain in written if is_binder_chain(chain.complex_chain)),
             'target_chains': ','.join(chain.letter for chain in written if not is_binder_chain(chain.complex_chain))}
    return {name: value for name, value in roles.items() if value}

def stamp_keyword(name: str) -> str:
    return name.replace('.', '_')

def read_structure_metadata(source: str) -> dict[str, str]:
    text = read_protein_source(source)
    if not is_mmcif_structure(source, text):
        return {}
    block = _pdbx.CIFFile.read(io.StringIO(text)).block
    category = block.get('bindcraft')
    return {name: str(column.as_item()) for name, column in category.items()} if category is not None else {}

def group_atoms_by_chain(records: list[dict]) -> dict[str, list]:
    chain_atom_records: dict[str, list] = {}
    for atom_record in records:
        chain_atom_records.setdefault(atom_record['chain_id'], []).append(atom_record)
    return chain_atom_records

def residue_identity(atom_record: dict) -> tuple[str, int, str]:
    return atom_record['chain_id'], atom_record['res_id'], atom_record['insertion_code']

def residue_atom_names(records: tuple[dict, ...] | list[dict]) -> dict[tuple[str, int, str], set[str]]:
    atom_names_by_residue: dict[tuple[str, int, str], set[str]] = {}
    for atom_record in records:
        atom_names_by_residue.setdefault(residue_identity(atom_record), set()).add(atom_record['name'])
    return atom_names_by_residue

def partly_trimmed_residues(source: str) -> tuple[str, ...]:
    records = [{**atom_record, 'res_name': MODIFIED_RESIDUE_PARENTS.get(atom_record['res_name'], atom_record['res_name'])} for atom_record in read_structure_atoms(source)]
    residue_names = {residue_identity(atom_record): atom_record['res_name'] for atom_record in records}
    return tuple(f'{residue_names[identity]} {identity[0]}/{identity[1]}' for identity, atom_names in residue_atom_names(records).items() if residue_names[identity] in ONE_LETTER_CODE and (not atom_names.issuperset(BACKBONE_ATOM_NAMES)))

def listed_offending_residues(residue_descriptions: list[str], limit: int=5) -> str:
    offenders = list(dict.fromkeys(residue_descriptions))
    return ', '.join(offenders[:limit]) + (f' and {len(offenders) - limit} more' if len(offenders) > limit else '')

def polymer_atom_records(records: tuple[dict, ...] | list[dict], source: str) -> list[dict]:
    polymer_records = [{**atom_record, 'res_name': MODIFIED_RESIDUE_PARENTS.get(atom_record['res_name'], atom_record['res_name']), 'name': MODIFIED_RESIDUE_ATOM_NAMES.get((atom_record['res_name'], atom_record['name']), atom_record['name'])} for atom_record in records]
    atom_names_by_residue = residue_atom_names(polymer_records)
    polymer_records = [atom_record for atom_record in polymer_records if atom_record['res_name'] in ONE_LETTER_CODE or not atom_record['is_hetero'] or atom_names_by_residue[residue_identity(atom_record)].issuperset(BACKBONE_ATOM_NAMES)]
    unsupported_residues = [f"{atom_record['res_name']} {atom_record['chain_id']}/{atom_record['res_id']}" for atom_record in polymer_records if atom_record['res_name'] not in ONE_LETTER_CODE]
    if unsupported_residues:
        raise ValueError(f'from_structure: unsupported residue(s) {listed_offending_residues(unsupported_residues)} in {structure_source_description(source)}; only the 20 standard amino acids and {", ".join(sorted(MODIFIED_RESIDUE_PARENTS))} are supported')
    polymer_records = [atom_record for atom_record in polymer_records if atom_names_by_residue[residue_identity(atom_record)].issuperset(BACKBONE_ATOM_NAMES)]
    inserted_residues = [f"{atom_record['chain_id']} {atom_record['res_id']}{atom_record['insertion_code']}" for atom_record in polymer_records if atom_record['insertion_code']]
    if inserted_residues:
        raise ValueError(f'from_structure: insertion codes are not supported; renumber residues {listed_offending_residues(inserted_residues)} in {structure_source_description(source)} before loading')
    residue_names_by_number: dict[tuple[str, int], str] = {}
    for atom_record in polymer_records:
        residue_name = residue_names_by_number.setdefault((atom_record['chain_id'], atom_record['res_id']), atom_record['res_name'])
        if residue_name != atom_record['res_name']:
            raise ValueError(f"from_structure: residue {atom_record['chain_id']}/{atom_record['res_id']} is both {residue_name} and {atom_record['res_name']} in {structure_source_description(source)}; renumber one of them before loading")
    return polymer_records

def atom37_arrays_from_records(records: list[dict]) -> tuple[list, list, list, list]:
    residue_names = {atom_record['res_id']: atom_record['res_name'] for atom_record in records}
    residue_index = sorted(residue_names)
    residue_positions = {residue_number: residue_position for residue_position, residue_number in enumerate(residue_index)}
    atom_coordinates = [[[0.0, 0.0, 0.0] for _ in ATOM_NAMES] for _ in residue_index]
    atom_mask = [[False for _ in ATOM_NAMES] for _ in residue_index]
    for atom_record in records:
        atom_index = ATOM_INDEX.get(atom_record['name'])
        if atom_index is not None:
            atom_coordinates[residue_positions[atom_record['res_id']]][atom_index] = atom_record['coord']
            atom_mask[residue_positions[atom_record['res_id']]][atom_index] = True
    return atom_coordinates, atom_mask, [AMINO_ACID_INDEX[ONE_LETTER_CODE[residue_names[residue_number]]] for residue_number in residue_index], residue_index
_EDIT_RE = re.compile('(?P<chain>[A-Za-z]+)(?P<start>\\d+)(?:-(?P<end>\\d+))?(?:\\((?P<lengths>[^)]+)\\))?(?P<flags>(?:[+-][A-Za-z]+|[*!])*)')

def parse_residue_length_choices(length_specification: str) -> list:
    length_choices = []
    for length_option in length_specification.split(','):
        length_option = length_option.strip()
        if '-' in length_option:
            minimum_length, maximum_length = (int(length_value) for length_value in length_option.split('-', 1))
            length_choices.extend(range(min(minimum_length, maximum_length), max(minimum_length, maximum_length) + 1))
        else:
            length_choices.append(int(length_option))
    return length_choices

def named_residue_flag(name: str) -> ResidueFlags:
    if name not in ResidueFlags.__members__:
        raise ValueError(f'residue span flag {name!r} is not a residue flag; the flags are {", ".join(flag.name for flag in ResidueFlags)}')
    return ResidueFlags[name]

def scaffold_edit_spans(edit_string: str, chain_letter: str, default_flags: ResidueFlags=ResidueFlags.DESIGN | ResidueFlags.CONTACT) -> list:
    edit_spans, unmatched_text, parsed_position = [], [], 0
    for residue_selection in _EDIT_RE.finditer(edit_string):
        unmatched_text.append(edit_string[parsed_position:residue_selection.start()].strip(', '))
        parsed_position = residue_selection.end()
        if residue_selection.group('chain') != chain_letter:
            continue
        start = int(residue_selection.group('start'))
        end = int(residue_selection.group('end')) if residue_selection.group('end') else start
        start, end = min(start, end), max(start, end)
        replacement_length_choices = residue_selection.group('lengths')
        flags = default_flags
        for flag_delta in re.findall('[+-][A-Za-z]+|[*!]', residue_selection.group('flags')):
            flag = ResidueFlags.CONTACT if flag_delta == '*' else ResidueFlags.DESIGN if flag_delta == '!' else named_residue_flag(flag_delta[1:])
            flags = flags | flag if flag_delta.startswith('+') else flags & ~flag
        edit_spans.append((start, end, parse_residue_length_choices(replacement_length_choices) if replacement_length_choices else None, flags))
    unmatched_text.append(edit_string[parsed_position:].strip(', '))
    if any(unmatched_text):
        raise ValueError(f'residue spans {edit_string!r} carry {" ".join(dict.fromkeys(text for text in unmatched_text if text))!r} outside any span; a span is written "A35", "A35-40", "A35-40(5-7)", "A35*" for a framework position, "A35!" to hold the residue unredesigned, or "A35+COLDSPOT" to set a flag by name')
    return edit_spans

def scaffold_edit_flags(edit_string: str) -> list:
    return [flags for chain_letter in dict.fromkeys(match.group('chain') for match in _EDIT_RE.finditer(edit_string)) for _, _, _, flags in scaffold_edit_spans(edit_string, chain_letter)]

def annotated_residue_flags(residue_index: list, edit_string: str, chain_letter: str, base_flags: ResidueFlags) -> list:
    residue_flags = [base_flags] * len(residue_index)
    for start, end, _, additional_flags in scaffold_edit_spans(edit_string, chain_letter, ResidueFlags.NONE):
        for residue_position, residue_number in enumerate(residue_index):
            if start <= residue_number <= end:
                residue_flags[residue_position] |= additional_flags
    return [int(residue_flag) for residue_flag in residue_flags]

def parse_scaffold_edits(edit_string: str, chain_letter: str, random_key: Array, default_flags: ResidueFlags=ResidueFlags.DESIGN | ResidueFlags.CONTACT) -> list:
    scaffold_edits = []
    for start, end, replacement_length_choices, flags in scaffold_edit_spans(edit_string, chain_letter, default_flags):
        if replacement_length_choices is None:
            replacement_length = end - start + 1
        else:
            random_key, length_random_key = jax.random.split(random_key)
            replacement_length = int(jax.random.choice(length_random_key, jnp.array(replacement_length_choices)))
        scaffold_edits.append((start, end, replacement_length, flags))
    return scaffold_edits

def label_sequence_positions(atom_array) -> dict[tuple[str, int], int]:
    positions, chain_lengths = {}, {}
    for atom_index in _struc.get_residue_starts(atom_array):
        chain = str(atom_array.chain_id[atom_index])
        chain_lengths[chain] = chain_lengths.get(chain, 0) + 1
        positions[chain, int(atom_array.res_id[atom_index])] = chain_lengths[chain]
    return positions

def renumber_label_sequence_ids(block, label_positions: dict) -> None:
    site = block['atom_site']
    chains, residues = site['label_asym_id'].as_array(), site['auth_seq_id'].as_array()
    site['label_seq_id'] = _pdbx.CIFColumn(_np.asarray([str(label_positions[str(chain), int(residue)]) for chain, residue in zip(chains, residues)]))

def address_bond_partners(block, label_positions: dict) -> None:
    if 'struct_conn' not in block:
        return
    connections = block['struct_conn']
    for partner in ('ptnr1', 'ptnr2'):
        chain_column = connections[f'{partner}_label_asym_id'].as_array()
        sequence_column = connections[f'{partner}_label_seq_id'].as_array()
        connections[f'{partner}_auth_asym_id'] = _pdbx.CIFColumn(_np.asarray([str(chain) for chain in chain_column]))
        connections[f'{partner}_auth_seq_id'] = _pdbx.CIFColumn(_np.asarray([str(residue) for residue in sequence_column]))
        connections[f'{partner}_auth_comp_id'] = _pdbx.CIFColumn(_np.asarray([str(residue_name) for residue_name in connections[f'{partner}_label_comp_id'].as_array()]))
        connections[f'{partner}_symmetry'] = _pdbx.CIFColumn(_np.full(len(sequence_column), '1_555'))
        connections[f'{partner}_label_seq_id'] = _pdbx.CIFColumn(_np.asarray([str(label_positions.get((str(chain), int(residue)), residue)) for chain, residue in zip(chain_column, sequence_column)]))

def declare_polymer_entities(block, atom_array, protein_complex: dict[str, Protein], receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> None:
    chain_names = {written.letter: f'{written.complex_chain} {written.label}'.strip() for written in written_chains(protein_complex, receptor_chains)}
    chain_residues: dict[str, list[str]] = {}
    for atom_index in _struc.get_residue_starts(atom_array):
        chain_residues.setdefault(str(atom_array.chain_id[atom_index]), []).append(str(atom_array.res_name[atom_index]))
    sequences = list(dict.fromkeys(tuple(residue_names) for residue_names in chain_residues.values()))
    entity_ids = {chain: str(sequences.index(tuple(residue_names)) + 1) for chain, residue_names in chain_residues.items()}
    block['atom_site']['label_entity_id'] = _pdbx.CIFColumn(_np.asarray([entity_ids[str(chain)] for chain in atom_array.chain_id]))
    entities, polymers, monomers = [], [], []
    for entity_position, residue_names in enumerate(sequences):
        entity_id = str(entity_position + 1)
        chains = sorted(chain for chain, chain_entity in entity_ids.items() if chain_entity == entity_id)
        amino_acid_sequence = ''.join(ONE_LETTER_CODE[residue_name] for residue_name in residue_names)
        entities.append({'id': entity_id, 'type': 'polymer', 'src_method': 'syn', 'pdbx_description': chain_names.get(chains[0], chains[0]), 'pdbx_number_of_molecules': str(len(chains))})
        polymers.append({'entity_id': entity_id, 'type': 'polypeptide(L)', 'nstd_linkage': 'no', 'nstd_monomer': 'no', 'pdbx_seq_one_letter_code': amino_acid_sequence, 'pdbx_seq_one_letter_code_can': amino_acid_sequence, 'pdbx_strand_id': ','.join(chains)})
        monomers.extend({'entity_id': entity_id, 'num': str(residue_position + 1), 'mon_id': residue_name, 'hetero': 'n'} for residue_position, residue_name in enumerate(residue_names))
    structural_chains = [{'id': chain, 'entity_id': entity_id} for chain, entity_id in sorted(entity_ids.items())]
    for category, rows in (('entity', entities), ('entity_poly', polymers), ('entity_poly_seq', monomers), ('struct_asym', structural_chains)):
        block[category] = _pdbx.CIFCategory({field: [row[field] for row in rows] for field in rows[0]})
