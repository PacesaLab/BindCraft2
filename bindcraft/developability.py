import functools
import numpy as np
from typing import NamedTuple
from bindcraft.protein import AMINO_ACIDS, AMINO_ACID_INDEX

EPITOPE_CORE_LENGTH = 9
GROOVE_PACKING_PREFERENCE = {'L': 0.3, 'V': 0.2, 'I': 0.2}
#for humanization only
MHC_CLASS_I_ANCHOR_PREFERENCES: dict[str, tuple[dict[str, float], ...]] = {
    'HLA_A0101': ({'T': 0.5, 'S': 0.3, 'V': 0.3}, {'T': 3.0, 'S': 2.5, 'V': 1.5, 'A': 1.0}, {'L': 0.5, 'I': 0.3, 'V': 0.3, 'M': 0.3}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'K': 0.3}, {'L': 0.5, 'K': 0.5, 'R': 0.3}, {'Y': 3.0, 'F': 2.5, 'W': 2.0}),
    'HLA_A0201': ({'L': 1.0, 'M': 0.8, 'V': 0.5}, {'L': 3.0, 'M': 2.5, 'V': 1.5, 'I': 1.5}, {'V': 0.8, 'L': 0.8, 'I': 0.5, 'A': 0.3}, {'L': 0.5, 'V': 0.5, 'I': 0.3}, {'V': 0.8, 'L': 0.5, 'I': 0.5, 'T': 0.3}, {'V': 1.0, 'T': 0.8, 'A': 0.5, 'I': 0.5}, {'L': 0.5, 'K': 0.5, 'R': 0.3}, {'L': 0.8, 'K': 0.8, 'R': 0.5}, {'V': 3.0, 'L': 2.5, 'I': 2.0, 'A': 1.0}),
    'HLA_A0301': ({'L': 0.5, 'V': 0.3, 'M': 0.3}, {'L': 2.5, 'V': 2.0, 'M': 1.5, 'I': 1.5}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'K': 0.3}, {'K': 0.8, 'R': 0.5}, {'R': 3.0, 'K': 2.5, 'Y': 1.0}),
    'HLA_A1101': ({'L': 0.5, 'V': 0.3, 'T': 0.3}, {'L': 2.5, 'V': 2.0, 'T': 1.5, 'I': 1.0}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'K': 0.3}, {'K': 0.8, 'R': 0.5}, {'R': 3.0, 'K': 2.5, 'Y': 1.0}),
    'HLA_A2402': ({'Y': 0.8, 'F': 0.5, 'I': 0.5}, {'Y': 3.0, 'F': 2.5, 'I': 2.0, 'W': 1.0}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'F': 0.5, 'Y': 0.3}, {'F': 0.8, 'L': 0.5, 'I': 0.5}, {'L': 3.0, 'F': 2.5, 'I': 2.0}),
    'HLA_A2601': ({'E': 0.8, 'V': 0.5}, {'E': 2.5, 'V': 2.0, 'A': 1.0}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'R': 0.5}, {'R': 0.8, 'F': 0.5}, {'R': 2.5, 'F': 2.5, 'L': 1.5}),
    'HLA_B0702': ({'R': 1.0, 'K': 0.8}, {'P': 4.0, 'A': 0.5}, {'L': 0.8, 'I': 0.5, 'V': 0.5}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.8, 'V': 0.5}, {'L': 2.5, 'V': 2.0, 'I': 1.5, 'A': 1.0}),
    'HLA_B0801': ({'K': 0.8, 'R': 0.5}, {'K': 3.0, 'R': 2.5}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'K': 0.5}, {'L': 0.8, 'K': 0.8}, {'L': 2.5, 'K': 2.0, 'R': 1.5}),
    'HLA_B1501': ({'L': 0.8, 'M': 0.5}, {'L': 2.5, 'M': 2.0, 'Q': 1.5}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.8, 'F': 0.5}, {'L': 2.5, 'F': 2.0, 'Y': 1.5}),
    'HLA_B2705': ({'R': 1.0, 'K': 0.5}, {'R': 3.5, 'K': 1.5}, {'L': 0.5, 'V': 0.3, 'I': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'K': 0.5}, {'L': 0.8, 'K': 0.5}, {'L': 2.5, 'K': 2.0, 'R': 1.5}),
    'HLA_B3901': ({'L': 0.5, 'N': 0.5}, {'L': 2.5, 'N': 2.0, 'I': 1.0}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.8, 'V': 0.5}, {'L': 2.5, 'V': 2.0, 'I': 1.5}),
    'HLA_B4001': ({'E': 0.8, 'L': 0.5}, {'E': 3.0, 'L': 2.0}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.8, 'M': 0.5}, {'L': 2.5, 'M': 2.0, 'E': 1.5}),
    'HLA_B5801': ({'A': 0.8, 'T': 0.5, 'S': 0.5}, {'A': 2.5, 'T': 2.0, 'S': 1.5}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'L': 0.5, 'V': 0.3}, {'Y': 0.8, 'F': 0.5}, {'Y': 2.5, 'F': 2.0, 'L': 1.0}),
    'H2_Kb': ({'S': 0.5, 'A': 0.5}, {'I': 2.0, 'A': 1.5, 'G': 1.0, 'V': 1.0}, {'I': 0.5, 'L': 0.5}, {'F': 0.5, 'I': 0.5}, {'F': 4.0, 'Y': 3.5}, {'E': 0.5, 'K': 0.5}, {'K': 0.5}, {'L': 1.0, 'M': 0.8, 'I': 0.8}, {'L': 3.0, 'M': 2.5, 'I': 2.0, 'V': 1.5}),
    'H2_Db': ({'M': 1.5, 'L': 1.0}, {'L': 0.5, 'I': 0.5}, {'L': 0.5}, {'L': 0.5}, {'N': 4.0, 'D': 2.5, 'S': 1.0}, {'L': 0.5}, {'L': 0.5}, {'L': 0.5}, {'L': 3.5, 'M': 3.0, 'I': 2.0, 'V': 1.0}),
    'H2_Kd': ({'L': 0.5}, {'Y': 4.0, 'F': 3.0, 'W': 1.5}, {'L': 0.5}, {'L': 0.5}, {'L': 0.5}, {'L': 0.5}, {'L': 0.5}, {'L': 0.5}, {'I': 3.0, 'L': 2.5, 'V': 1.5, 'M': 1.0})}
MHC_CLASS_II_ANCHOR_PREFERENCES: dict[str, tuple[dict[str, float], ...]] = {
    'HLA_DRB1_0101': ({'Y': 3.0, 'F': 2.8, 'W': 2.5, 'L': 2.0, 'I': 1.5, 'M': 1.5, 'V': 1.0}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'S': 0.8, 'T': 0.8, 'N': 0.8, 'Q': 0.5}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'L': 1.5, 'I': 1.2, 'V': 1.0, 'M': 1.0}),
    'HLA_DRB1_0301': ({'L': 2.5, 'I': 2.0, 'M': 1.8, 'F': 1.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'D': 2.0, 'E': 2.0, 'N': 1.0}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'K': 1.2, 'R': 1.0, 'L': 0.8}),
    'HLA_DRB1_0401': ({'F': 2.8, 'Y': 2.5, 'W': 2.2, 'L': 1.8, 'M': 1.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'Q': 1.5, 'N': 1.0, 'T': 0.8, 'K': 0.5}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'A': 1.2, 'V': 1.0, 'T': 0.8}),
    'HLA_DRB1_0701': ({'F': 2.8, 'Y': 2.5, 'L': 2.0, 'M': 1.8, 'I': 1.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'A': 1.0, 'S': 0.8, 'T': 0.8}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'A': 1.2, 'V': 1.0, 'T': 0.8, 'S': 0.6}),
    'HLA_DRB1_1101': ({'L': 2.5, 'I': 2.0, 'M': 1.8, 'V': 1.5, 'F': 1.2}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'L': 1.0, 'M': 0.8, 'V': 0.8}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'V': 1.2, 'A': 1.0, 'L': 0.8}),
    'HLA_DRB1_1301': ({'L': 2.5, 'I': 2.0, 'V': 1.8, 'M': 1.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'K': 1.5, 'R': 1.5, 'Q': 0.8}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'A': 1.0, 'V': 0.8}),
    'HLA_DRB1_1501': ({'F': 2.8, 'Y': 2.5, 'W': 2.2, 'L': 1.8, 'M': 1.5, 'I': 1.2}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'V': 1.2, 'I': 1.0, 'L': 1.0}, GROOVE_PACKING_PREFERENCE, {'G': 1.0, 'A': 1.0, 'S': 0.8, 'P': 0.5}, GROOVE_PACKING_PREFERENCE, GROOVE_PACKING_PREFERENCE, {'V': 1.2, 'I': 1.0, 'L': 0.8})}
HUMAN_MHC_CLASS_I_ALLELES = tuple(name for name in MHC_CLASS_I_ANCHOR_PREFERENCES if name.startswith('HLA'))
MOUSE_MHC_CLASS_I_ALLELES = 'H2_Kb', 'H2_Db', 'H2_Kd'
MHC_CLASS_I_SPECIES_ALLELES: dict[str, tuple[str, ...]] = {'human': HUMAN_MHC_CLASS_I_ALLELES, 'mouse': MOUSE_MHC_CLASS_I_ALLELES, 'mouse_c57bl6': ('H2_Kb', 'H2_Db'), 'mouse_balbc': ('H2_Kd',), 'both': HUMAN_MHC_CLASS_I_ALLELES + MOUSE_MHC_CLASS_I_ALLELES}
MHC_CLASS_II_SPECIES_ALLELES: dict[str, tuple[str, ...]] = {'human': tuple(MHC_CLASS_II_ANCHOR_PREFERENCES), 'mouse': (), 'mouse_c57bl6': (), 'mouse_balbc': (), 'both': tuple(MHC_CLASS_II_ANCHOR_PREFERENCES)}
MHC_CLASS_I_COUPLED_ANCHORS = (1, 8, 1.0), (0, 1, 0.4), (2, 8, 0.4)
MHC_CLASS_II_COUPLED_ANCHORS = (0, 8, 1.0), (0, 3, 0.5), (3, 8, 0.5)
MHC_CLASS_I_GROOVE_BURIAL = np.asarray([0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1], dtype=np.float32)
MHC_CLASS_II_GROOVE_BURIAL = np.asarray([0.1, 0.3, 0.8, 1.0, 1.0, 1.0, 0.8, 0.3, 0.1], dtype=np.float32)
KYTE_DOOLITTLE_HYDROPATHY = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
_hydropathy = np.asarray([KYTE_DOOLITTLE_HYDROPATHY[amino_acid] for amino_acid in AMINO_ACIDS], dtype=np.float32)
HYDROPHOBICITY = np.maximum(0.0, _hydropathy) / _hydropathy.max()

class MHCPanel(NamedTuple):
    anchor_matrices: np.ndarray
    coupling_matrices: np.ndarray
    coupled_anchors: tuple[tuple[int, int, float], ...]
    groove_burial: np.ndarray

def anchor_preference_matrix(position_preferences: tuple[dict[str, float], ...]) -> np.ndarray:
    matrix = np.zeros((len(position_preferences), len(AMINO_ACIDS)), dtype=np.float32)
    for position, preferences in enumerate(position_preferences):
        for amino_acid, preference in preferences.items():
            matrix[position, AMINO_ACID_INDEX[amino_acid]] = preference
    return matrix

def build_mhc_panel(anchor_preferences: dict[str, tuple[dict[str, float], ...]], allele_names: tuple[str, ...], coupled_anchors: tuple[tuple[int, int, float], ...], groove_burial: np.ndarray) -> MHCPanel:
    anchor_matrices = np.asarray([anchor_preference_matrix(anchor_preferences[name]) for name in allele_names], dtype=np.float32).reshape(len(allele_names), EPITOPE_CORE_LENGTH, len(AMINO_ACIDS))
    coupling_matrices = np.asarray([[np.outer(matrix[first], matrix[second]) * scale for first, second, scale in coupled_anchors] for matrix in anchor_matrices], dtype=np.float32).reshape(len(allele_names), len(coupled_anchors), len(AMINO_ACIDS), len(AMINO_ACIDS))
    return MHCPanel(anchor_matrices, coupling_matrices, coupled_anchors, groove_burial)

@functools.lru_cache(maxsize=None)
def mhc_panels(species: str='human') -> tuple[MHCPanel, MHCPanel]:
    if species not in MHC_CLASS_I_SPECIES_ALLELES:
        raise ValueError(f'unknown humanization species {species!r}, expected one of {sorted(MHC_CLASS_I_SPECIES_ALLELES)}')
    return (build_mhc_panel(MHC_CLASS_I_ANCHOR_PREFERENCES, MHC_CLASS_I_SPECIES_ALLELES[species], MHC_CLASS_I_COUPLED_ANCHORS, MHC_CLASS_I_GROOVE_BURIAL), build_mhc_panel(MHC_CLASS_II_ANCHOR_PREFERENCES, MHC_CLASS_II_SPECIES_ALLELES[species], MHC_CLASS_II_COUPLED_ANCHORS, MHC_CLASS_II_GROOVE_BURIAL))

#for protease resistance only
PROTEASES: dict[str, tuple[float, dict[str, float], set[str]]] = {
    'trypsin': (1.0, {'K': 1.0, 'R': 1.0}, {'P'}),
    'chymotrypsin': (1.0, {'F': 1.0, 'Y': 1.0, 'W': 1.0, 'L': 0.5, 'M': 0.5}, {'P'}),
    'elastase': (0.4, {'A': 1.0, 'V': 1.0, 'G': 0.6, 'S': 0.6, 'L': 0.4, 'I': 0.4}, {'P'}),
    'pepsin': (0.5, {'F': 1.0, 'L': 1.0, 'W': 1.0, 'Y': 1.0}, set())}

class ProteasePanel(NamedTuple):
    weights: np.ndarray
    p1: np.ndarray
    p1_block: np.ndarray

def build_protease_panel(names: tuple[str, ...]) -> ProteasePanel:
    weights = np.asarray([PROTEASES[name][0] for name in names], dtype=np.float32)
    p1 = np.zeros((len(names), len(AMINO_ACIDS)), dtype=np.float32)
    p1_block = np.zeros((len(names), len(AMINO_ACIDS)), dtype=np.float32)
    for row, name in enumerate(names):
        for amino_acid, preference in PROTEASES[name][1].items():
            p1[row, AMINO_ACID_INDEX[amino_acid]] = preference
        for amino_acid in PROTEASES[name][2]:
            p1_block[row, AMINO_ACID_INDEX[amino_acid]] = 1.0
    return ProteasePanel(weights, p1, p1_block)

@functools.lru_cache(maxsize=None)
def protease_panel() -> ProteasePanel:
    return build_protease_panel(tuple(PROTEASES))
