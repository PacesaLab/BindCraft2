"""Geometry-preserving relaxation of an AF2-predicted protein complex.

  * every torsion-invariant intra-residue distance -- which is exactly every 1-2 bond,
    every 1-3 distance (hence every bond angle) and every within-ring 1-4/1-5 distance.
    Invariant under psi as well as chi: chi alone would also freeze N-O and O-CB, which
    pins psi wherever no peptide unit does;
  * the signed volume of every set of four mutually torsion-invariant atoms, which adds the
    one thing the distances leave free, namely handedness: planarity of sp2 groups and
    rings, and chirality at CA and at the beta-branched centres;
  * the proline ring closure, which AF2's torsion parameterisation leaves open;
  * the peptide unit and every disulfide, found by DISTANCE and frozen as a rigid pose.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bindcraft.af.alphafold.common import residue_constants as rc

__all__ = ['relax', 'default_relax_parameters']

N_SLOT, CA_SLOT, C_SLOT, O_SLOT = 0, 1, 2, 3
SG_SLOT = rc.restype_name_to_atom14_names['CYS'].index('SG')
BACKBONE_SLOTS = (N_SLOT, CA_SLOT, C_SLOT, O_SLOT)
PRO_TYPE = rc.restype_order['P']
CYS_TYPE = rc.restype_order['C']
_PRO = rc.restype_name_to_atom14_names['PRO']
PRO_RING_SLOTS = tuple(_PRO.index(n) for n in ('N', 'CA', 'CB', 'CG', 'CD'))
PRO_CLOSURE_PAIRS = tuple((_PRO.index(a), _PRO.index(b))
                          for a, b in (('N', 'CD'), ('N', 'CG'), ('CA', 'CD')))
PRO_VOLUMES = tuple(tuple(_PRO.index(n) for n in base) + (_PRO.index(apex),) for apex, base in (
    ('N', ('CA', 'CB', 'CD')), ('CA', ('N', 'CB', 'CD')), ('CB', ('N', 'CA', 'CD')),
    ('CG', ('N', 'CB', 'CD')), ('CD', ('N', 'CB', 'CG'))))

PEPTIDE_PAIRS = ((C_SLOT, N_SLOT), (CA_SLOT, N_SLOT), (C_SLOT, CA_SLOT),
                 (O_SLOT, N_SLOT), (CA_SLOT, CA_SLOT), (O_SLOT, CA_SLOT))
PEPTIDE_PLANE = ((0, CA_SLOT), (0, C_SLOT), (0, O_SLOT), (1, N_SLOT), (1, CA_SLOT))

BOND_CN_CUTOFF = 1.9
BOND_SS_CUTOFF = 2.5
GAP_JUNCTION_CUTOFF = 3.0
_CORROBORATE = (1.8, 3.2)
_RESIDUE_GAP = 64 
_NEIGHBOUR_MARGIN = 2.0
_MAX_VDW_SUM = 2.0 * max(rc.van_der_waals_radius.values())
_CN_LENGTH = rc.between_res_bond_length_c_n
_CN_STDDEV = rc.between_res_bond_length_stddev_c_n
_CA_C_N = rc.between_res_cos_angles_ca_c_n
_C_N_CA = rc.between_res_cos_angles_c_n_ca


# --------------------------------------------------------------------------- ideal geometry

def _torsion_invariant() -> np.ndarray:
    """(20, 14, 14) bool: the intra-residue distances no AF2 torsion can change.

    A rotation moves exactly the atoms of the rigid groups it feeds.  ``psi`` feeds group
    3, the carbonyl O and nothing else; ``chi_k`` feeds group 4+k and, through the
    chi2..chi4 chaining, every group past it; ``pre_omega`` and ``phi`` place no heavy
    atom at all.  A distance survives a rotation when both its atoms move, when neither
    does, or when the stationary one sits ON the axis -- the clause that keeps CA-CG and
    CB-CD.  The axis is the torsion's middle bond, so "on the axis" is a NAME test rather
    than a geometric one: over all 20 types the two inner atoms of ``chi_angles_atoms``
    are the only atoms within 1e-6 A of a chi axis, and the group-3 frame's x axis is the
    CA->C bond to 0.00 degrees.
    """
    occupied = rc.restype_atom14_mask[:20] > 0
    invariant = np.zeros((20, 14, 14), bool)
    for t in range(20):
        three = rc.restype_1to3[rc.restypes[t]]
        names = rc.restype_name_to_atom14_names[three]
        group = rc.restype_atom14_to_rigid_group[t]
        turns = [(group == 3, ('CA', 'C'))]                       # psi, about the CA->C bond
        turns += [(group >= 4 + k, (atoms[1], atoms[2]))          # chi_k, about its middle bond
                  for k, atoms in enumerate(rc.chi_angles_atoms[three])]
        keep = occupied[t][:, None] & occupied[t][None, :] & ~np.eye(14, dtype=bool)
        for moves, axis in turns:
            on_axis = ~moves & np.array([name in axis for name in names])
            keep &= ~(moves[:, None] ^ moves[None, :]) | on_axis[:, None] | on_axis[None, :]
        invariant[t] = keep
    return invariant


def _ideal_pose() -> np.ndarray:
    """(20, 14, 3) atom14 positions of every residue type at zero torsion.

    ``all_atom.torsion_angles_to_frames`` composed with
    ``frames_and_literature_positions_to_atom14_pos``, which at zero torsion collapses to
    a bare product of AF2's default frames -- so no sin/cos table, and no ~20 s jit of the
    vendored pair. All four atoms of a clique are mutually torsion-invariant.
    """
    frame = rc.restype_rigid_group_default_frame[:20].astype(np.float64).copy()
    for g in (5, 6, 7):                      # chi2..chi4 compose onto the preceding group
        frame[:, g] = frame[:, g - 1] @ frame[:, g]
    frame = frame[np.arange(20)[:, None], rc.restype_atom14_to_rigid_group[:20]]
    literature = rc.restype_atom14_rigid_group_positions[:20].astype(np.float64)
    return np.einsum('tkij,tkj->tki', frame[..., :3, :3], literature) + frame[..., :3, 3]


def _maximal_cliques(adjacency: dict[int, set[int]], nodes: list[int]) -> list[list[int]]:
    """Bron-Kerbosch with pivoting."""
    out: list[list[int]] = []

    def expand(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            out.append(sorted(r))
            return
        pivot = max(p | x, key=lambda v: len(adjacency[v] & p))
        for v in list(p - adjacency[pivot]):
            expand(r | {v}, p & adjacency[v], x & adjacency[v])
            p, x = p - {v}, x | {v}

    expand(set(), set(nodes), set())
    return out


def _best_triple(points: np.ndarray, others: list[int]) -> tuple:
    """The three of ``others`` spanning the largest triangle, which conditions the volume best.

    ``points[v]`` may be a STACK of positions rather than one, so that a whole set of
    equivalent groups -- the individual peptide links -- is scored at once; the areas are
    then averaged.  
    """
    def area(p, q, r):
        return float(np.mean(0.5 * np.linalg.norm(
            np.cross(points[q] - points[p], points[r] - points[p]), axis=-1)))

    return max(itertools.combinations(others, 3), key=lambda t: area(*t))


def _build_tables() -> dict:
    """Per-residue-type restraint topology. Deterministic; built once, then cached."""
    occupied = rc.restype_atom14_mask[:20] > 0
    invariant = _torsion_invariant()
    ideal = _ideal_pose()

    pairs, volumes = {}, {}
    for t in range(20):
        pairs[t] = [(a, b) for a in range(14) for b in range(a + 1, 14) if invariant[t, a, b]]
        adjacency = {v: set(np.nonzero(invariant[t, v])[0].tolist()) for v in range(14)}
        volumes[t] = [(*_best_triple(ideal[t], [o for o in clique if o != apex]), apex)
                      for clique in _maximal_cliques(adjacency,
                                                     np.nonzero(occupied[t])[0].tolist())
                      if len(clique) >= 4 for apex in clique]

    exempt = invariant | np.eye(14, dtype=bool)[None]
    for a, b in PRO_CLOSURE_PAIRS:
        exempt[PRO_TYPE, a, b] = exempt[PRO_TYPE, b, a] = True

    radius = np.array([[rc.van_der_waals_radius[name[0]] if name else 0.0
                        for name in rc.restype_name_to_atom14_names[rc.restype_1to3[r]]]
                       for r in rc.restypes[:20]])
    return {'pairs': pairs, 'volumes': volumes, 'pro_volumes': PRO_VOLUMES,
            'exempt': exempt, 'radius': radius}


_TABLES: dict | None = None


def _tables() -> dict:
    global _TABLES
    if _TABLES is None:
        _TABLES = _build_tables()
    return _TABLES


# --------------------------------------------------------------------------- parameters

def default_relax_parameters() -> dict:
    return {'steps': 900,
            'learning_rate': 0.02,
            'learning_rate_final_fraction': 0.6,
            'restraint_backbone': 10.0,
            'restraint_sidechain': 0.25,
            'weight_clash': 15.0,
            'overlap_tol': 0.4,
            'weight_intra': 5.0,
            'intra_tol': 1.5,
            'weight_peptide': 1.0,
            'peptide_tolerance': 12.0,
            'weight_geometry': 1000.0,
            'weight_covalent': 10000.0,
            'neighbour_refresh': 25}


def resolve_parameters(parameters: dict | None) -> dict:
    par = dict(default_relax_parameters())
    for name, value in (parameters or {}).items():
        if name in par and value is not None:
            par[name] = value
    for name, value in par.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'relax parameter {name!r} must be a finite number, got {value!r}')
        if value < 0.0:
            raise ValueError(f'relax parameter {name!r} must not be negative, got {value!r}')
    for name in ('steps', 'neighbour_refresh'):
        par[name] = int(par[name])
        if par[name] < 1:
            raise ValueError(f'relax parameter {name!r} must be at least 1, got {par[name]!r}')
    if par['learning_rate'] <= 0.0:
        raise ValueError("relax parameter 'learning_rate' must be positive")
    return par


# --------------------------------------------------------------------------- flattening

def flatten(protein_complex: dict) -> dict:
    """dict[str, Protein] -> concatenated atom14 arrays plus bookkeeping.

    ``index`` is synthetic and strictly increasing, with a difference of 1 only between
    residues consecutive within one chain. AF2's clash losses mask pairs with
    ``index[i] < index[j]`` over the full grid with no chain term, so per-chain author
    numbering -- which restarts at 1 for every chain -- would make every cross-chain pair
    sharing a residue number invisible.
    """
    from bindcraft.protein import output_chain_letters

    letters = output_chain_letters(protein_complex)
    chains = sorted(protein_complex, key=letters.__getitem__)
    parts: dict[str, list] = {k: [] for k in ('aatype', 'x37', 'm37', 'numbering', 'chain')}
    lengths = []
    for k, chain in enumerate(chains):
        protein = protein_complex[chain]
        parts['aatype'].append(np.asarray(protein.sequence).argmax(-1).astype(np.int32))
        parts['x37'].append(np.asarray(protein.atoms, dtype=np.float32))
        parts['m37'].append(np.asarray(protein.atom_mask).astype(np.float32))
        parts['numbering'].append(np.asarray(protein.residue_index).astype(np.int64))
        parts['chain'].append(np.full(len(protein), k, dtype=np.int32))
        lengths.append(len(protein))
    flat = {k: np.concatenate(v) for k, v in parts.items()}

    aatype = flat['aatype']
    slot37 = rc.restype_atom14_to_atom37[aatype]
    flat['slot37'] = slot37
    flat['x14'] = np.take_along_axis(flat['x37'], slot37[..., None], 1).astype(np.float32)
    flat['m14'] = (rc.restype_atom14_mask[aatype]
                   * np.take_along_axis(flat['m37'], slot37, 1)).astype(np.float32)

    n = len(aatype)
    step = np.zeros(n, np.int64)
    if n > 1:
        same = (flat['chain'][1:] == flat['chain'][:-1]) & (np.diff(flat['numbering']) == 1)
        step[1:] = np.where(same, 1, _RESIDUE_GAP)
    flat['index'] = np.cumsum(step)
    flat['chains'], flat['lengths'] = chains, lengths
    return flat


# --------------------------------------------------------------------------- covalent bonds

def covalent_bonds(x14: np.ndarray, m14: np.ndarray, index: np.ndarray,
                   is_cys: np.ndarray) -> dict:
    """Covalent links found by GEOMETRY, as ``protein.py:117-121`` does.

    Returns three sets, deliberately distinct:
      ``peptide``    C(i)-N(j) links corroborated as real peptide units, which get frozen;
      ``amide``      every inter-residue C...N contact under the cutoff, which is what the
                     shipped code refuses to push apart. A superset: BC2 inputs contain
                     stretched junctions at numbering gaps (C...N 1.87-2.85 A with CA-CA
                     4.5-5.6 A) that are not peptide units and must not be frozen as such,
                     but must not be pulled apart either;
      ``disulfide``  SG...SG links between two CYS residues.

    ``is_cys`` is not optional. atom14 slot 5 is SG only for cysteine; for every other
    residue it is whatever that type's sixth atom happens to be (CG, OG, CD, ...).
    """
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if len(x14) < 2:
        return {'peptide': empty, 'amide': empty, 'disulfide': empty}
    present = m14 > 0

    d = np.linalg.norm(x14[:, C_SLOT][:, None, :] - x14[:, N_SLOT][None, :, :], axis=-1)
    hit = (d < BOND_CN_CUTOFF) & present[:, C_SLOT][:, None] & present[:, N_SLOT][None, :]
    np.fill_diagonal(hit, False)
    i, j = (v.astype(np.int64) for v in np.nonzero(hit))

    lo, hi = _CORROBORATE
    d_ca_n = np.linalg.norm(x14[i, CA_SLOT] - x14[j, N_SLOT], axis=-1)
    d_c_ca = np.linalg.norm(x14[i, C_SLOT] - x14[j, CA_SLOT], axis=-1)
    keep = (((d_ca_n > lo) & (d_ca_n < hi) & (d_c_ca > lo) & (d_c_ca < hi))
            | (index[j] - index[i] == 1))

    #slot 5 is SG only for CYS, so the search runs over the cysteines alone
    cys = np.nonzero(is_cys & present[:, SG_SLOT])[0]
    sg = x14[cys, SG_SLOT]
    close = np.linalg.norm(sg[:, None, :] - sg[None, :, :], axis=-1) < BOND_SS_CUTOFF
    a, b = np.nonzero(np.triu(close, 1))
    return {'peptide': (i[keep], j[keep]), 'amide': (i, j),
            'disulfide': (cys[a].astype(np.int64), cys[b].astype(np.int64))}


# --------------------------------------------------------------------------- neighbour list

def residue_neighbours(x14: np.ndarray, m14: np.ndarray, index: np.ndarray,
                       margin: float, pad: int | None = None) -> tuple[np.ndarray, int]:
    """Padded (N, K) list of residues j with ``index[j] > index[i]``, lossless by construction.

    The bound is exact: atoms of residues i and j are at least
    ``d(centre_i, centre_j) - reach_i - reach_j`` apart, so nothing within ``margin`` is
    excluded.
    """
    present = (m14 > 0).astype(np.float64)
    centre = (x14 * present[..., None]).sum(1) / np.maximum(present.sum(1, keepdims=True), 1.0)
    reach = (np.linalg.norm(x14 - centre[:, None, :], axis=-1) * present).max(1)
    separation = np.linalg.norm(centre[:, None, :] - centre[None, :, :], axis=-1)
    candidate = ((separation < reach[:, None] + reach[None, :] + margin)
                 & (index[:, None] < index[None, :]))

    counts = candidate.sum(1)
    k = max(int(counts.max()) if len(counts) else 0, 1, int(pad or 0))
    neighbours = np.full((len(x14), k), -1, dtype=np.int32)
    rows, cols = np.nonzero(candidate)
    if len(rows):
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        neighbours[rows, np.arange(len(rows)) - np.repeat(starts, counts)] = cols
    if int((neighbours >= 0).sum()) != int(candidate.sum()):
        raise RuntimeError('neighbour list dropped pairs')
    return neighbours, k


# --------------------------------------------------------------------------- restraint assembly

def _signed_volume(x: np.ndarray, a, b, c, d) -> np.ndarray:
    return np.einsum('ij,ij->i', np.cross(x[b] - x[a], x[c] - x[a]), x[d] - x[a])


class _Restraints:
    """Accumulates the two restraint kinds. Every target is read off the input.

    A restraint is a tuple of flat atom-index columns plus the parameter name holding its
    weight: two columns make a frozen distance, four a frozen signed volume.
    """

    def __init__(self, flat_x: np.ndarray, weights: dict):
        self.x, self.w = flat_x, weights
        #a zero-length seed row, so the concatenations in pack() need no empty special case
        self.d = [(np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float32),)]
        self.v = [(np.zeros(0, np.int64),) * 4 + (np.zeros(0, np.float32),)]

    def add(self, *columns_and_key):
        *columns, key = columns_and_key
        columns = tuple(np.asarray(c, np.int64).ravel() for c in columns)
        if len(columns[0]):
            (self.d if len(columns) == 2 else self.v).append(
                columns + (np.full(len(columns[0]), self.w[key], np.float32),))

    def pack(self) -> dict:
        di, dj, dw = (np.concatenate(col) for col in zip(*self.d))
        va, vb, vc, vd, vw = (np.concatenate(col) for col in zip(*self.v))
        #normalise the volume by twice its base-triangle area, so the residual is a distance
        area = 0.5 * np.linalg.norm(np.cross(self.x[vb] - self.x[va],
                                             self.x[vc] - self.x[va]), axis=-1)
        return {'di': di.astype(np.int32), 'dj': dj.astype(np.int32), 'dw': dw,
                'd0': np.linalg.norm(self.x[di] - self.x[dj], axis=-1).astype(np.float32),
                'va': va.astype(np.int32), 'vb': vb.astype(np.int32),
                'vc': vc.astype(np.int32), 'vd': vd.astype(np.int32), 'vw': vw,
                'v0': _signed_volume(self.x, va, vb, vc, vd).astype(np.float32),
                'vs': np.maximum(2.0 * area, 0.5).astype(np.float32)}


def prepare(protein_complex: dict, par: dict) -> dict:
    tab = _tables()
    flat = flatten(protein_complex)
    aatype, x14, m14, index = flat['aatype'], flat['x14'], flat['m14'], flat['index']
    n = len(aatype)
    present = m14 > 0
    flat_x = x14.reshape(-1, 3)
    cell = lambda rows, slot: np.asarray(rows, np.int64) * 14 + slot

    is_cys = aatype == CYS_TYPE
    bonds = covalent_bonds(x14, m14, index, is_cys)
    src, dst = bonds['peptide']
    ss_i, ss_j = bonds['disulfide']
    R = _Restraints(flat_x, par)

    def restrain(key, table, *rowsets):
        """Emit one restraint per (row of ``table``, residue), skipping unresolved atoms.

        ``table`` carries one atom14 slot per vertex -- 2 for a distance, 4 for a signed
        volume -- and ``rowsets`` the residue rows those slots sit in: a single set shared
        by every vertex for an intra-residue restraint, or one per vertex across a link.
        """
        if not len(table) or not len(rowsets[0]):
            return
        table, m = np.asarray(table, np.int64), len(rowsets[0])
        #one row set per vertex, or a single set shared by all of them
        rowsets = rowsets if len(rowsets) == table.shape[1] else rowsets * table.shape[1]
        rows = np.stack([np.tile(r, len(table)) for r in rowsets])
        slots = np.repeat(table.T, m, 1)
        keep = np.all(present[rows, slots], 0)
        R.add(*cell(rows, slots)[:, keep], key)

    # intra-residue: every torsion-invariant distance, and every rigid clique's handedness
    for t in range(20):
        rows = np.nonzero(aatype == t)[0]
        restrain('weight_geometry', tab['pairs'][t], rows)
        restrain('weight_geometry', tab['volumes'][t], rows)

    # proline ring closure and pucker, which the torsion parameterisation leaves open.
    # The ring volumes want all five ring atoms, not just the four they name
    pro = np.nonzero(aatype == PRO_TYPE)[0]
    restrain('weight_geometry', PRO_CLOSURE_PAIRS, pro)
    restrain('weight_geometry', tab['pro_volumes'],
             pro[np.all(present[np.ix_(pro, list(PRO_RING_SLOTS))], 1)])

    # the peptide unit and the disulfides, as rigid poses. covalent_bonds resolved only the
    restrain('weight_covalent', PEPTIDE_PAIRS, src, dst)
    restrain('weight_covalent', [(SG_SLOT, SG_SLOT)], ss_i, ss_j)

    #stretched junctions
    gap = np.nonzero((np.diff(index) != 1) & (flat['chain'][1:] == flat['chain'][:-1]))[0]
    junction = gap[present[gap, C_SLOT] & present[gap + 1, N_SLOT]
                   & (np.linalg.norm(x14[gap, C_SLOT] - x14[gap + 1, N_SLOT], axis=-1)
                      < GAP_JUNCTION_CUTOFF)
                   & ~np.isin(gap * n + gap + 1, src * n + dst)]   # not already a peptide unit
    restrain('weight_covalent', [(C_SLOT, N_SLOT)], junction, junction + 1)

    # the peptide plane, as five rigid handednesses over the link's five atoms
    keep = np.all([present[(src, dst)[w], s] for w, s in PEPTIDE_PLANE], 0)
    if keep.any():
        group = np.stack([cell((src, dst)[w][keep], s) for w, s in PEPTIDE_PLANE])
        for apex in range(5):
            triple = _best_triple(flat_x[group], [k for k in range(5) if k != apex])
            R.add(*group[list(triple)], group[apex], 'weight_covalent')
    terms = R.pack()

    # clash floors clamped
    clamp_lo, clamp_hi = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]

    def clamp(rows_i, rows_j, slots_i=np.arange(14), slots_j=np.arange(14)):
        """Clamp every (slot_i, slot_j) combination of every residue pair i, j.

        The slots default to all 196 combinations, which is what a covalent link protects;
        a bare pair of slots clamps that one contact. ``lo``/``hi`` are ordered by ``index``,
        matching the direction energy()'s dense twin charges the pair in.
        """
        rows_i, rows_j = np.asarray(rows_i, np.int64), np.asarray(rows_j, np.int64)
        si = np.repeat(slots_i, np.size(slots_j))        # the slot grid of one residue pair
        sj = np.tile(slots_j, np.size(slots_i))
        forward = (index[rows_i] < index[rows_j])[:, None]
        alive = (present[rows_i[:, None], si] & present[rows_j[:, None], sj]
                 & (forward | (index[rows_i] > index[rows_j])[:, None]))
        a, b = cell(rows_i[:, None], si), cell(rows_j[:, None], sj)
        clamp_lo.append(np.where(forward, a, b)[alive])
        clamp_hi.append(np.where(forward, b, a)[alive])

    nxt = np.nonzero(np.diff(index) == 1)[0]
    amide_i, amide_j = bonds['amide']
    cys = np.nonzero(is_cys & present[:, SG_SLOT])[0]
    tri_a, tri_b = np.triu_indices(len(cys), 1)
    clamp(np.concatenate([src, ss_i, junction]),        # every link, in all 196 slot pairs
          np.concatenate([dst, ss_j, junction + 1]))
    clamp(np.concatenate([nxt, amide_i]),               # every consecutive C-N, every amide
          np.concatenate([nxt + 1, amide_j]), C_SLOT, N_SLOT)
    clamp(cys[tri_a], cys[tri_b], SG_SLOT, SG_SLOT)     # every SG-SG, bonded or not

    r14 = tab['radius'][aatype]
    lo, hi = np.concatenate(clamp_lo), np.concatenate(clamp_hi)
    _, keep = np.unique(lo * (14 * max(n, 1)) + hi, return_index=True)
    lo, hi = lo[keep], hi[keep]
    floor = r14[lo // 14, lo % 14] + r14[hi // 14, hi % 14] - par['overlap_tol']
    d_in = np.linalg.norm(flat_x[lo] - flat_x[hi], axis=-1)
    exempt = ((index[hi // 14] - index[lo // 14] == 1)
              & (lo % 14 == C_SLOT) & (hi % 14 == N_SLOT))
    exempt |= ((lo % 14 == SG_SLOT) & (hi % 14 == SG_SLOT)
               & is_cys[lo // 14] & is_cys[hi // 14])
    terms.update(ci=lo.astype(np.int32), cj=hi.astype(np.int32),
                 c_floor=floor.astype(np.float32),
                 c_clamped=np.minimum(floor, d_in).astype(np.float32),
                 c_charged=(~exempt).astype(np.float32))

    d_intra = np.linalg.norm(x14[:, :, None, :] - x14[:, None, :, :], axis=-1)
    lower = np.where(tab['exempt'][aatype], 0.0,
                     np.minimum(r14[:, :, None] + r14[:, None, :] - par['intra_tol'], d_intra))

    position = np.where(np.isin(np.arange(14), BACKBONE_SLOTS)[None, :],
                        par['restraint_backbone'], par['restraint_sidechain'])
    terms.update(
        flat=flat, n_residue=n, aatype=aatype, n_bond=float(len(nxt)),
        x0=x14, m14=m14, index=index.astype(np.int32),
        radius=(r14 * m14).astype(np.float32),
        lower=np.maximum(lower, 0.0).astype(np.float32),
        position_weight=(position * m14).astype(np.float32))
    return {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in terms.items()}


# --------------------------------------------------------------------------- energy

def _distance(flat_x, i, j):
    return jnp.sqrt(jnp.sum(jnp.square(flat_x[i] - flat_x[j]), -1) + 1e-8)


def _volume(flat_x, a, b, c, d):
    return jnp.einsum('ij,ij->i', jnp.cross(flat_x[b] - flat_x[a], flat_x[c] - flat_x[a]),
                      flat_x[d] - flat_x[a])


def _peptide_guard(x, terms, tolerance):
    """AF2's ``between_residue_bond_loss``, the three gap-masked means, times the bond count.
    """
    c, n_ = x[:-1, C_SLOT], x[1:, N_SLOT]
    ca, next_ca = x[:-1, CA_SLOT], x[1:, CA_SLOT]
    m = terms['m14']
    gap = (terms['index'][1:] - terms['index'][:-1] == 1).astype(x.dtype)
    mask = m[:-1, C_SLOT] * m[1:, N_SLOT] * m[:-1, CA_SLOT] * m[1:, CA_SLOT] * gap

    to_pro = (terms['aatype'][1:] == PRO_TYPE).astype(x.dtype)
    hinge = lambda v, mean, dev: jax.nn.relu(
        jnp.sqrt(1e-6 + jnp.square(v - mean)) - tolerance * dev)

    cn = hinge(jnp.sqrt(1e-6 + jnp.sum(jnp.square(c - n_), -1)),
               (1.0 - to_pro) * _CN_LENGTH[0] + to_pro * _CN_LENGTH[1],
               (1.0 - to_pro) * _CN_STDDEV[0] + to_pro * _CN_STDDEV[1])
    unit = lambda v: v / jnp.sqrt(1e-6 + jnp.sum(jnp.square(v), -1))[:, None]
    c_ca, c_n, n_ca = unit(ca - c), unit(n_ - c), unit(next_ca - n_)
    a1 = hinge(jnp.sum(c_ca * c_n, -1), _CA_C_N[0], _CN_STDDEV[0])
    a2 = hinge(jnp.sum(-c_n * n_ca, -1), _C_N_CA[0], _C_N_CA[1])
    return jnp.sum(mask * (cn + a1 + a2)) / (jnp.sum(mask) + 1e-6) * terms['n_bond']


def energy(x, terms, par, neighbours):
    """Sum of: sparse inter-residue clash, intra-residue clash, frozen distances,
    frozen signed volumes, and the positional restraint."""
    flat_x = x.reshape(-1, 3)

    valid = neighbours >= 0
    nb = jnp.where(valid, neighbours, 0)
    m, r = terms['m14'], terms['radius']
    d = jnp.sqrt(1e-10 + jnp.sum(jnp.square(x[:, None, :, None, :] - x[nb][:, :, None, :, :]), -1))
    one_hot = lambda a, b, gate: (gate[:, :, None, None]
                                  * jnp.zeros((14, 14), x.dtype).at[a, b].set(1.0))
    cys = terms['aatype'] == CYS_TYPE
    mask = (m[:, None, :, None] * m[nb][:, :, None, :] * valid[:, :, None, None]
            * (1.0
               - one_hot(C_SLOT, N_SLOT,
                         ((terms['index'][:, None] + 1) == terms['index'][nb]).astype(x.dtype))
               - one_hot(SG_SLOT, SG_SLOT, (cys[:, None] & cys[nb]).astype(x.dtype))))
    floor = r[:, None, :, None] + r[nb][:, :, None, :] - par['overlap_tol']
    e = jnp.sum(mask * jax.nn.relu(floor - d))

    # replace the floor of every clamped pair by min(AF2 floor, input distance)
    if terms['ci'].size:
        dc = _distance(flat_x, terms['ci'], terms['cj'])
        e = e + jnp.sum(jax.nn.relu(terms['c_clamped'] - dc)
                        - terms['c_charged'] * jax.nn.relu(terms['c_floor'] - dc))
    e = par['weight_clash'] * e

    d_intra = jnp.sqrt(1e-10 + jnp.sum(jnp.square(x[:, :, None, :] - x[:, None, :, :]), -1))
    intra = m[:, :, None] * m[:, None, :] * jax.nn.relu(terms['lower'] - d_intra)
    e = e + par['weight_intra'] * jnp.sum(intra)

    if terms['di'].size:
        e = e + jnp.sum(terms['dw'] * jnp.square(
            _distance(flat_x, terms['di'], terms['dj']) - terms['d0']))
    if terms['va'].size:
        residual = (_volume(flat_x, terms['va'], terms['vb'], terms['vc'], terms['vd'])
                    - terms['v0']) / terms['vs']
        e = e + jnp.sum(terms['vw'] * jnp.square(residual))

    if terms['n_bond'] > 0 and par['weight_peptide'] > 0:
        e = e + par['weight_peptide'] * _peptide_guard(x, terms, par['peptide_tolerance'])

    return e + jnp.sum(terms['position_weight'] * jnp.sum(jnp.square(x - terms['x0']), -1))


# --------------------------------------------------------------------------- driver

def _minimise(terms: dict, par: dict) -> np.ndarray:
    x = terms['x0']
    mask = terms['m14'][..., None]
    m14, index = np.asarray(terms['m14']), np.asarray(terms['index'])
    margin = _NEIGHBOUR_MARGIN + _MAX_VDW_SUM - par['overlap_tol']
    gradient = jax.jit(jax.grad(lambda xx, nb: energy(xx, terms, par, nb)))

    schedule = optax.cosine_decay_schedule(par['learning_rate'], par['steps'],
                                           par['learning_rate_final_fraction'])
    optimizer = optax.adam(schedule)
    state = optimizer.init(x)
    pad = None
    for step in range(par['steps']):
        if step % par['neighbour_refresh'] == 0:
            neighbours, pad = residue_neighbours(np.asarray(x), m14, index, margin, pad)
            nb = jnp.asarray(neighbours)        # host -> device once per refresh, not per step
        updates, state = optimizer.update(gradient(x, nb) * mask, state)
        x = optax.apply_updates(x, updates)
    return np.asarray(x)


def _write_back(protein_complex: dict, flat: dict, m14: np.ndarray, x14: np.ndarray) -> dict:
    """atom14 -> atom37 -> a new dict of new Protein objects, per the BC2 contract."""
    x37 = flat['x37'].copy()
    for s in range(14):
        sel = np.nonzero(m14[:, s] > 0)[0]
        if len(sel):
            x37[sel, flat['slot37'][sel, s]] = x14[sel, s]

    #OXT has no atom14 slot and restype_atom37_to_atom14 aliases it onto N, so an atom14
    oxt = rc.atom_order.get('OXT')
    rows = np.nonzero(flat['m37'][:, oxt] > 0)[0] if oxt is not None else np.zeros(0, int)
    if len(rows):
        frame = [N_SLOT, CA_SLOT, C_SLOT]
        a, b = flat['x14'][rows][:, frame, :], x14[rows][:, frame, :]
        ca, cb = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
        u, _, vt = np.linalg.svd(np.einsum('nij,nik->njk', a - ca, b - cb))
        flip = np.repeat(np.eye(3)[None], len(rows), 0)
        flip[:, 2, 2] = np.sign(np.linalg.det(np.einsum('nij,njk->nik', u, vt)))
        rotation = np.einsum('nij,njk,nkl->nil', u, flip, vt)
        relative = flat['x37'][rows, oxt] - ca[:, 0, :]
        x37[rows, oxt] = np.einsum('ni,nij->nj', relative, rotation) + cb[:, 0, :]

    out, offset = {}, 0
    for chain, length in zip(flat['chains'], flat['lengths']):
        protein = protein_complex[chain]
        block = x37[offset:offset + length]
        offset += length
        kept = np.where(np.asarray(protein.atom_mask)[..., None], block,
                        np.asarray(protein.atoms, dtype=np.float32))
        out[chain] = protein.replace(atoms=jnp.asarray(kept, dtype=protein.atoms.dtype))
    return out


def relax(protein_complex: dict, parameters: dict | None = None) -> dict:
    """Relax a complex, returning a new dict of new Protein objects.

    Matches ``protein.relax_protein_complex``'s observable contract: the outputs share the
    inputs' ``sequence``, ``atom_mask``, ``flags`` and ``residue_index`` arrays, only
    ``atoms`` is replaced and cast back to the input dtype, coordinates are restored
    wherever ``atom_mask`` is false, and the input complex is not mutated.
    """
    if not isinstance(protein_complex, dict):
        raise ValueError('relax expects a dict[str, Protein]')
    par = resolve_parameters(parameters)
    if not protein_complex:
        return {}
    terms = prepare(protein_complex, par)
    if terms['n_residue'] == 0:
        return {chain: protein.replace(atoms=protein.atoms)
                for chain, protein in protein_complex.items()}
    x14 = _minimise(terms, par)
    if not np.all(np.isfinite(x14)):
        raise RuntimeError('relax produced non-finite coordinates')
    return _write_back(protein_complex, terms['flat'], np.asarray(terms['m14']), x14)
