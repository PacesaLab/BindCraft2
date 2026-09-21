"""CPU-only regression tests; no AlphaFold weights are required."""

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bindcraft.af2 import MONOMER_CHAIN_GAP, monomer_chain_break_indices


class MonomerChainBreakIndicesTests(unittest.TestCase):
    def assert_chain_gaps_preserved(self, chains):
        lengths = tuple(len(chain) for chain in chains)
        original = jnp.asarray(np.concatenate(chains), dtype=jnp.int32)
        for predict in (
            partial(monomer_chain_break_indices, lengths),
            jax.jit(partial(monomer_chain_break_indices, lengths)),
        ):
            indices = np.asarray(predict(original))
            self.assertEqual(indices.shape, original.shape)
            self.assertEqual(indices.dtype, np.dtype('int32'))
            self.assertEqual(indices[0], 0)
            start = 0
            for chain in chains:
                stop = start + len(chain)
                np.testing.assert_array_equal(
                    np.diff(indices[start:stop]), np.diff(chain)
                )
                if start:
                    self.assertEqual(
                        indices[start] - indices[start - 1], MONOMER_CHAIN_GAP + 1
                    )
                start = stop

    def test_multichain_receptor_preserves_peptide_mhc_break(self):
        # One logical target contains a 10-residue peptide and 181-residue MHC.
        self.assert_chain_gaps_preserved([
            np.arange(137),
            np.r_[np.arange(1, 11), np.arange(59, 240)],
        ])

    def test_continuous_chains_keep_previous_numbering(self):
        original = jnp.asarray(np.r_[np.arange(137), np.arange(1, 192)])
        actual = monomer_chain_break_indices((137, 191), original)
        np.testing.assert_array_equal(
            actual, np.r_[np.arange(137), np.arange(186, 377)]
        )

    def test_additional_breaks_and_padding(self):
        cases = {
            'three receptor chains': [
                np.arange(137),
                np.r_[np.arange(1, 11), np.arange(59, 240), np.arange(288, 387)],
            ],
            'multiple logical chains and missing residues': [
                np.array([5, 6, 55, 56]),
                np.array([100, 102, 151, 152]),
                np.arange(7),
            ],
            'padded binder and target': [
                np.arange(160), np.r_[np.arange(1, 11), np.arange(59, 241)],
            ],
            'single logical chain': [np.array([30, 31, 80, 81])],
        }
        for label, chains in cases.items():
            with self.subTest(label=label):
                self.assert_chain_gaps_preserved(chains)


if __name__ == '__main__':
    unittest.main()
