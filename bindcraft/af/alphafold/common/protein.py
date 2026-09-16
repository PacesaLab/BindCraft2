# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import io
import warnings
from typing import Any, Mapping, Optional, Type, Sequence
from bindcraft.af.alphafold.common import residue_constants
from functools import partial
import numpy as np

import biotite.structure as _struc
import biotite.structure.io.pdb as _biopdb
import biotite.structure.io.pdbx as _biopdbx

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

_EMPTY_UNOBS = lambda: {'aatype': np.array([], dtype=int),
                        'residue_index': np.array([], dtype=int),
                        'chains': np.array([], dtype=str)}


def _unobserved_from_block(block):
  """Unobserved / zero-occupancy residues from a biotite CIF block's pdbx category (mmCIF only;
  empty for PDB)."""
  cat = block.get('pdbx_unobs_or_zero_occ_residues') if block is not None else None
  if cat is None or 'auth_comp_id' not in cat:
    return _EMPTY_UNOBS()
  resname_3 = [str(r) for r in cat['auth_comp_id'].as_array()]
  resname_1 = [residue_constants.restype_3to1.get(r, 'X') for r in resname_3]
  aatype = [residue_constants.restype_order.get(r, residue_constants.restype_num) for r in resname_1]
  return {'aatype': np.array(aatype, dtype=int),
          'residue_index': np.array([int(x) for x in cat['auth_seq_id'].as_array()], dtype=int),
          'chains': np.array([str(x) for x in cat['auth_asym_id'].as_array()])}


def merge_structure_unobserved(
  unobserved,
  atom_positions,
  aatype,
  atom_mask,
  residue_index,
  b_factors
  ):
  # keep the atom37 dimensions explicit so an empty observed set (every residue dropped)
  # cannot collapse the mask/positions to 1-D (which later blows up on `mask[:,0]`).
  A = residue_constants.atom_type_num
  observed = {
    'atom_positions': np.asarray(atom_positions, dtype=float).reshape(-1, A, 3),
    'aatype': np.array(aatype, dtype=int),
    'atom_mask': np.asarray(atom_mask, dtype=float).reshape(-1, A),
    'residue_index': np.array(residue_index, dtype=int),
    'b_factors': np.asarray(b_factors, dtype=float).reshape(-1, A)
  }
  unobserved_dim = unobserved['residue_index'].shape[0]
  unobserved['atom_positions'] = np.zeros((unobserved_dim, A, 3))
  unobserved['atom_mask'] = np.zeros((unobserved_dim, A))
  unobserved['b_factors'] = np.zeros((unobserved_dim, A))
  merged = {k: np.concatenate([observed[k], unobserved[k]]) for k in observed.keys()}
  order = np.argsort(merged['residue_index'])
  merged = {k: v[order] for k, v in merged.items()}

  return merged


@dataclasses.dataclass(frozen=True)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]


# Backbone atoms a residue must have to be placeable. A residue missing any of them
# is a fragment, not a residue -- see the skip in from_string() below.
_BACKBONE_IDX = np.array([residue_constants.atom_order[a] for a in ("N", "CA", "C")])
_warned_partial = set()


_ATOM_SET = set(residue_constants.atom_types)          # atom37 names (fast membership)


def _warn_partial_once(chain_id, resid, resname, present):
  '''Report a dropped partial residue once per structure position (targets are re-parsed for
  every reprediction, so this would otherwise spam the log).'''
  key = (chain_id, resid, resname)
  if key not in _warned_partial:
    _warned_partial.add(key)
    print(f"WARNING: dropping partial residue {chain_id}{resid} ({str(resname).strip()}) "
          f"-- no complete backbone, only {sorted(present)} present.")


def _first_block(cif):
  try:
    return cif.block                                     # biotite: the sole block
  except Exception:
    return next(iter(cif.values()))


def _load_biotite(st):
  """Parse a PDB/CIF string into (single-model AtomArray with b_factor, CIF-block-or-None)."""
  s = st.lstrip()
  looks_cif = s.startswith(('data_', 'loop_', '_', '#')) or '_atom_site.' in st
  # mmCIFs that carry only label_* (not auth_*) atom_site columns make biotite warn once per
  # attribute about the label_* fallback, and a PDB with no element column makes it warn that it
  # guessed the elements from the atom names. Both fallbacks are correct (verified: identical
  # atom37) and both are re-emitted on every structure load, so silence just these two: hIL7RA.pdb
  # has no element column and read_structure_atoms already drops the pair on the other read path.
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*not found within 'atom_site' category.*",
                            category=UserWarning)
    warnings.filterwarnings("ignore", message=r'\d+ elements were guessed from atom name',
                            category=UserWarning)
    if looks_cif:
      block = _first_block(_biopdbx.CIFFile.read(io.StringIO(st)))
      return _biopdbx.get_structure(block, model=1, extra_fields=['b_factor']), block
    pdb = _biopdb.PDBFile.read(io.StringIO(st))
    return pdb.get_structure(model=1, extra_fields=['b_factor']), None


def from_string(st: str, chain_ids: Optional[Sequence[str]] = None) -> Mapping[str, Protein]:
  """Parse a PDB/CIF string into {chain_id: Protein} using biotite.

  Reproduces the vendored AlphaFold parser exactly: standard residues -> aatype index,
  non-standard -> UNK (20), non-atom37 atoms ignored, insertion codes rejected, hetero residues
  skipped, partial (no complete N/CA/C backbone) residues dropped, and mmCIF unobserved residues
  merged in and sorted by residue index. Produces atom37 atom_positions/atom_mask/b_factors.
  """
  atoms, block = _load_biotite(st)
  unobserved = _unobserved_from_block(block)
  cats = atoms.get_annotation_categories()
  has_ins, has_het, has_b = ('ins_code' in cats), ('hetero' in cats), ('b_factor' in cats)

  if chain_ids is not None:
    chains = list(chain_ids)
  else:
    chains = list(dict.fromkeys(atoms.chain_id.tolist()))   # unique, file order

  proteins = dict()
  for cid in chains:
    sel = atoms[atoms.chain_id == cid]
    starts = _struc.get_residue_starts(sel)
    ends = np.append(starts[1:], sel.array_length())
    atom_positions, aatype, atom_mask, residue_index, b_factors = [], [], [], [], []

    for s, e in zip(starts, ends):
      if has_ins and str(sel.ins_code[s]).strip():
        raise ValueError(f'Structure file contains an insertion code at chain {cid} and residue '
                         f'index {int(sel.res_id[s])}. These are not supported.')
      if has_het and bool(sel.hetero[s]):                # hetero residue (e.g. water / ligand)
        continue
      resname = str(sel.res_name[s])
      restype_idx = residue_constants.restype_order.get(
          residue_constants.restype_3to1.get(resname, 'X'), residue_constants.restype_num)
      pos = np.zeros((residue_constants.atom_type_num, 3))
      mask = np.zeros((residue_constants.atom_type_num,))
      res_b = np.zeros((residue_constants.atom_type_num,))
      for i in range(int(s), int(e)):
        name = str(sel.atom_name[i])
        if name not in _ATOM_SET:
          continue
        ao = residue_constants.atom_order[name]
        pos[ao] = sel.coord[i]
        mask[ao] = 1.
        res_b[ao] = float(sel.b_factor[i]) if has_b else 0.
      if mask[_BACKBONE_IDX].sum() < len(_BACKBONE_IDX):
        # no usable backbone (trimmed residue / stray side-chain atoms) -> drop; unobserved merge
        # will treat the position as unobserved instead.
        if np.sum(mask) >= 0.5:
          _warn_partial_once(cid, int(sel.res_id[s]), resname,
                             [str(sel.atom_name[i]) for i in range(int(s), int(e))])
        continue
      aatype.append(restype_idx)
      atom_positions.append(pos)
      atom_mask.append(mask)
      residue_index.append(int(sel.res_id[s]))
      b_factors.append(res_b)

    mask_c = unobserved['chains'] == cid
    unobs_c = {k: v[mask_c] for k, v in unobserved.items()}
    proteins[cid] = Protein(**merge_structure_unobserved(
        unobserved=unobs_c, atom_positions=atom_positions, aatype=aatype,
        atom_mask=atom_mask, residue_index=residue_index, b_factors=b_factors))
  return proteins


# live path: biotite (from_string ignores the format hint -- it auto-detects PDB vs CIF).
from_pdb_string = from_string
from_mmcif_string = from_string

def to_pdb(prot: Protein) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  restypes = residue_constants.restypes + ['X']
  res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
  atom_types = residue_constants.atom_types

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  b_factors = prot.b_factors

  if np.any(aatype > residue_constants.restype_num):
    raise ValueError('Invalid aatypes.')

  pdb_lines.append('MODEL     1')
  atom_index = 1
  chain_id = 'A'
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_id:>1}'
                   f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the chain.
  chain_end = 'TER'
  chain_termination_line = (
      f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} '
      f'{chain_id:>1}{residue_index[-1]:>4}')
  pdb_lines.append(chain_termination_line)
  pdb_lines.append('ENDMDL')

  pdb_lines.append('END')
  pdb_lines.append('')
  return '\n'.join(pdb_lines)


def ideal_atom_mask(prot: Protein) -> np.ndarray:
  """Computes an ideal atom mask.

  `Protein.atom_mask` typically is defined according to the atoms that are
  reported in the PDB. This function computes a mask according to heavy atoms
  that should be present in the given sequence of amino acids.

  Args:
    prot: `Protein` whose fields are `numpy.ndarray` objects.

  Returns:
    An ideal atom mask.
  """
  return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(features: FeatureDict, result: ModelOutput,
                    b_factors: Optional[np.ndarray] = None) -> Protein:
  """Assembles a protein from a prediction.

  Args:
    features: Dictionary holding model inputs.
    result: Dictionary holding model outputs.
    b_factors: (Optional) B-factors to use for the protein.

  Returns:
    A protein instance.
  """
  fold_output = result['structure_module']
  if b_factors is None:
    b_factors = np.zeros_like(fold_output['final_atom_mask'])

  return Protein(
      aatype=features['aatype'][0],
      atom_positions=fold_output['final_atom_positions'],
      atom_mask=fold_output['final_atom_mask'],
      residue_index=features['residue_index'][0] + 1,
      b_factors=b_factors)
