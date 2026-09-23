import os
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
from jax import Array
from bindcraft.prediction import ProteinPredictor, CompiledModelCache, residue_chain_ids, concatenate_chain_arrays, split_residue_arrays_by_chain
from bindcraft.model_weights import MPNN_WEIGHT_VARIANTS, mpnn_variant_directory
from bindcraft.af2 import DEFAULT_LENGTH_BUCKET, padded_prediction_complex
from bindcraft.mpnn.modules import ProteinMPNN
from bindcraft.sequence_optimization import OMITTED_AMINO_ACID_LOGIT
from bindcraft.protein import AMINO_ACIDS, ATOM_NAMES, StructurePrediction, StructurePredictions, Protein, ResidueFlags, ProteinStates, has_residue_flag, has_resolved_atom

_PROTEIN_ALPHABET = AMINO_ACIDS + 'X'
_MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
_TO_MPNN = np.asarray([_PROTEIN_ALPHABET.index(a) for a in _MPNN_ALPHABET])
_FROM_MPNN = np.asarray([_MPNN_ALPHABET.index(a) for a in _PROTEIN_ALPHABET])
_BACKBONE_ATOMS = tuple(ATOM_NAMES.index(name) for name in ('N', 'CA', 'C', 'O'))
TIED_STATE_SEPARATION_ANGSTROMS = 1000.0
CHECKPOINT_PARAMETER_SEPARATOR = '|'

def sequence_to_mpnn_alphabet(sequence_values: Array) -> Array:
    if sequence_values.shape[-1] == 20:
        sequence_values = jnp.pad(sequence_values, [(0, 0)] * (sequence_values.ndim - 1) + [(0, 1)])
    return sequence_values[..., _TO_MPNN]

def sequence_from_mpnn_alphabet(sequence_values: Array) -> Array:
    return sequence_values[..., _FROM_MPNN]

def proteinmpnn_input_features(atoms: Array, resolved_residue_mask: Array, residue_index: Array, chain_indices: Array, fixed_residue_mask: Array, sequence: Array, temperature: Array, key: Array, tied_residue_groups: Array | None=None, redesigned_amino_acid_bias: Array | None=None) -> dict[str, Array]:
    decoding_priorities = jax.random.uniform(key, resolved_residue_mask.shape)
    decoding_priorities = jnp.where(resolved_residue_mask.astype(bool), decoding_priorities, decoding_priorities + 1)
    decoding_priorities = jnp.where(fixed_residue_mask, decoding_priorities - 1, decoding_priorities)
    redesigned_sequence_bias = 0.0 if redesigned_amino_acid_bias is None else sequence_to_mpnn_alphabet(redesigned_amino_acid_bias)
    sequence_bias = jnp.where(fixed_residue_mask[:, None], 10000000.0 * sequence_to_mpnn_alphabet(sequence), redesigned_sequence_bias)
    features = {'X': atoms[:, _BACKBONE_ATOMS, :].astype(jnp.float32), 'mask': resolved_residue_mask, 'residue_idx': residue_index, 'chain_idx': chain_indices, 'bias': sequence_bias, 'temperature': temperature}
    if tied_residue_groups is None:
        return {**features, 'decoding_order': decoding_priorities.argsort()}
    decoding_order = tied_residue_groups[decoding_priorities[tied_residue_groups[:, 0]].argsort()]
    decoding_step = jnp.zeros(resolved_residue_mask.shape, dtype=jnp.int32).at[decoding_order].set(jnp.arange(decoding_order.shape[0], dtype=jnp.int32)[:, None])
    return {**features, 'decoding_order': decoding_order, 'ar_mask': (decoding_step[:, None] > decoding_step[None, :]).astype(jnp.float32)}

def tied_chain_residue_groups(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...], chain_groups: tuple[tuple[str, ...], ...]) -> tuple[tuple[int, ...], ...]:
    chain_offsets, residue_count = {}, 0
    for name, length in zip(chain_names, chain_lengths):
        chain_offsets[name], residue_count = residue_count, residue_count + length
    tied_groups = tuple(tuple(chain_offsets[name] + position for name in group) for group in chain_groups if set(group) <= set(chain_names) for position in range(chain_lengths[chain_names.index(group[0])]))
    if not tied_groups:
        return ()
    tied_rows = {row for group in tied_groups for row in group}
    group_width = max((len(group) for group in tied_groups))
    return tuple(sorted(tied_groups + tuple((row,) * group_width for row in range(residue_count) if row not in tied_rows)))

def read_mpnn_checkpoint(path: str) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    with np.load(path, allow_pickle=False) as checkpoint:
        model_parameters: dict[str, dict[str, np.ndarray]] = {}
        for key in checkpoint.files:
            module_name, separator, parameter_name = key.rpartition(CHECKPOINT_PARAMETER_SEPARATOR)
            if separator:
                model_parameters.setdefault(module_name, {})[parameter_name] = checkpoint[key]
        return model_parameters, int(checkpoint['num_edges'])


class ProteinMPNNSequenceModel(ProteinPredictor):
    def __init__(self, data_dir: str, model_name: str='v_48_020', temperature: float=0.1, key: Array | None=None, max_cache_size: int=8, variant: str='neutral', omitted_amino_acids: str='', amino_acid_bias: dict[str, float] | None=None, multi_chain_binders: tuple[tuple[str, ...], ...]=(), length_bucket_size: int=DEFAULT_LENGTH_BUCKET, target_pad_length: int=0, redesign_max_positions: int | None=None, redesign_position_temperature: float=1.0):
        self.model_name = model_name
        self.variant = variant
        #None leaves a decode unconstrained, a number keeps only that many substitutions and reverts the rest
        self.redesign_max_positions = None if redesign_max_positions is None else int(redesign_max_positions)
        self.redesign_position_temperature = float(redesign_position_temperature)
        self.length_bucket_size = length_bucket_size
        self.target_pad_length = target_pad_length
        self.multi_chain_binders = multi_chain_binders
        self.temperature = temperature
        self.key = jax.random.PRNGKey(0) if key is None else key
        amino_acid_bias = amino_acid_bias or {}
        self.redesigned_amino_acid_bias = jnp.asarray([OMITTED_AMINO_ACID_LOGIT if amino_acid in omitted_amino_acids else amino_acid_bias.get(amino_acid, 0.0) for amino_acid in AMINO_ACIDS], dtype=jnp.float32)
        path = os.path.join(mpnn_variant_directory(data_dir, variant), f'{model_name}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'MPNN weights not found: {path}')
        checkpoint_parameters, neighbor_count = read_mpnn_checkpoint(path)
        self.model_parameters = jax.tree_util.tree_map(jnp.asarray, checkpoint_parameters)
        self.mpnn_config = {'num_letters': 21, 'node_features': 128, 'edge_features': 128, 'hidden_dim': 128, 'num_encoder_layers': 3, 'num_decoder_layers': 3, 'augment_eps': 0.0, 'k_neighbors': neighbor_count, 'dropout': 0.0}
        def sample_backbone_sequence(inputs: dict[str, Array]):
            return ProteinMPNN(**self.mpnn_config).sample(inputs)
        self.mpnn_sampler = hk.transform(sample_backbone_sequence)
        self.prediction_compile_cache = CompiledModelCache(max_cache_size)

    def capped_amino_acids(self, key: Array, amino_acid_log_probabilities: Array, sampled_amino_acids: Array, original_amino_acids: Array, redesigned_residue_mask: Array) -> Array:
        #only the most improving substitutions are applied, so a candidate stays a short hop from the sequence it started from
        residue_positions = jnp.arange(sampled_amino_acids.shape[0])
        improvement = amino_acid_log_probabilities[residue_positions, sampled_amino_acids] - amino_acid_log_probabilities[residue_positions, original_amino_acids]
        #Gumbel-top-k: ranked on improvement alone the same few positions win every draw, so the cap would return one variant however many were asked for
        ranked = improvement if not self.redesign_position_temperature else improvement / self.redesign_position_temperature + jax.random.gumbel(jax.random.fold_in(key, 0), improvement.shape)
        substituted = redesigned_residue_mask & (sampled_amino_acids != original_amino_acids)
        #fewer substitutions than the cap needs no guard: top_k then lands on positions where the sample already is the original, so writing it back changes nothing
        kept = jax.lax.top_k(jnp.where(substituted, ranked, -jnp.inf), min(self.redesign_max_positions, sampled_amino_acids.shape[0]))[1]
        return original_amino_acids.at[kept].set(sampled_amino_acids[kept])

    def _compiled_complex_prediction(self, chain_lengths: tuple[int, ...], tied_residue_groups: tuple=()) -> Callable:
        cache_key = chain_lengths, tied_residue_groups
        compiled_sequence_prediction = self.prediction_compile_cache.get(cache_key)
        if compiled_sequence_prediction is None:
            chain_indices = residue_chain_ids(chain_lengths)
            grouped_residues = jnp.asarray(tied_residue_groups, dtype=jnp.int32) if tied_residue_groups else None
            def predict_mpnn_sequence(model_parameters: Array, key: Array, atoms: Array, atom_mask: Array, residue_index: Array, sequence: Array, fixed_residue_mask: Array, temperature: Array):
                resolved_ca_mask = has_resolved_atom(atom_mask, 'CA').astype(jnp.float32)
                mpnn_inputs = proteinmpnn_input_features(atoms, resolved_ca_mask, residue_index, chain_indices, fixed_residue_mask, sequence, temperature, key, grouped_residues, self.redesigned_amino_acid_bias)
                mpnn_prediction = self.mpnn_sampler.apply(model_parameters, key, mpnn_inputs)
                sampled_amino_acids = sequence_from_mpnn_alphabet(mpnn_prediction['S'])[..., :20].argmax(-1)
                amino_acid_log_probabilities = jax.nn.log_softmax(sequence_from_mpnn_alphabet(mpnn_prediction['logits']), axis=-1)[..., :20]
                redesigned_residue_mask = jnp.logical_not(fixed_residue_mask)
                original_amino_acids = sequence.argmax(-1)
                selected_amino_acids = jnp.where(redesigned_residue_mask, sampled_amino_acids, original_amino_acids) if self.redesign_max_positions is None else self.capped_amino_acids(key, amino_acid_log_probabilities, sampled_amino_acids, original_amino_acids, redesigned_residue_mask)
                residue_negative_log_likelihood = -jnp.take_along_axis(amino_acid_log_probabilities, selected_amino_acids[:, None], axis=-1)[:, 0]
                sequence_negative_log_likelihood = (residue_negative_log_likelihood * resolved_ca_mask).sum() / (resolved_ca_mask.sum() + 1e-08)
                recovered_residue_mask = (selected_amino_acids == original_amino_acids).astype(jnp.float32)
                sequence_recovery = (recovered_residue_mask * resolved_ca_mask).sum() / (resolved_ca_mask.sum() + 1e-08)
                updated_sequence = jax.nn.one_hot(selected_amino_acids, 20).astype(jnp.float16)
                return updated_sequence, sequence_negative_log_likelihood, sequence_recovery
            compiled_sequence_prediction = jax.jit(jax.vmap(predict_mpnn_sequence, in_axes=(None, 0, None, None, None, None, None, None)))
            self.prediction_compile_cache.set(cache_key, compiled_sequence_prediction)
        return compiled_sequence_prediction

    def predict(self, protein_states: ProteinStates, model: str | None=None) -> StructurePredictions:
        return self.predict_candidates(protein_states, model)[0]

    def predict_candidates(self, protein_states: ProteinStates, model: str | None=None, candidate_count: int=1) -> list[StructurePredictions]:
        if model not in (None, self.model_name):
            raise ValueError(f'ProteinMPNN loaded {self.model_name!r} and cannot redesign with {model!r}')
        decoded = {name: self._predict_complex(protein_complex, candidate_count=candidate_count) for name, protein_complex in protein_states.items()}
        return [{name: candidates[candidate_index] for name, candidates in decoded.items()} for candidate_index in range(candidate_count)]

    def predict_tied_candidates(self, protein_states: ProteinStates, candidate_count: int=1) -> list[StructurePredictions]:
        state_names = tuple(sorted(protein_states))
        shared_chain_names = tuple(sorted(set.intersection(*(set(protein_complex) for protein_complex in protein_states.values()))))
        separated_complex = {f'{chain_name}.{state_name}': protein.replace(atoms=protein.atoms.astype(jnp.float32) + jnp.asarray([state_index * TIED_STATE_SEPARATION_ANGSTROMS, 0.0, 0.0], dtype=jnp.float32)) for state_index, state_name in enumerate(state_names) for chain_name, protein in protein_states[state_name].items()}
        decode_chain_groups = tuple(tuple(f'{chain_name}.{state_name}' for chain_name in chain_group for state_name in state_names) for chain_group in (self.multi_chain_binders or tuple((name,) for name in shared_chain_names)))
        decoded = self._predict_complex(separated_complex, chain_groups=decode_chain_groups, candidate_count=candidate_count)
        return [{state_name: StructurePrediction(protein_complex={chain_name: protein.replace(sequence=candidate.protein_complex[f'{chain_name}.{state_name}'].sequence) for chain_name, protein in protein_states[state_name].items()}, metrics=candidate.metrics) for state_name in state_names} for candidate in decoded]

    def _predict_complex(self, protein_complex: dict[str, Protein], tied_residue_groups: tuple=(), chain_groups: tuple[tuple[str, ...], ...]=(), candidate_count: int=1) -> list[StructurePrediction]:
        chain_names = tuple(sorted(protein_complex))
        true_lengths = tuple(len(protein_complex[name]) for name in chain_names)
        padded_complex = padded_prediction_complex(protein_complex, self.length_bucket_size, self.target_pad_length)
        chain_lengths = tuple(len(padded_complex[name]) for name in chain_names)
        chain_arrays = concatenate_chain_arrays(chain_names, padded_complex, 'atoms', 'atom_mask', 'residue_index', 'flags', 'sequence')
        atoms, atom_mask, residue_index, flags = chain_arrays['atoms'], chain_arrays['atom_mask'], chain_arrays['residue_index'], chain_arrays['flags']
        sequence = chain_arrays['sequence'].astype(jnp.float32)
        fixed_residue_mask = jnp.logical_not(has_residue_flag(flags, ResidueFlags.DESIGN))
        self.key, *sampling_random_keys = jax.random.split(self.key, candidate_count + 1)
        compiled_sequence_prediction = self._compiled_complex_prediction(chain_lengths, tied_residue_groups or tied_chain_residue_groups(chain_names, chain_lengths, chain_groups or self.multi_chain_binders))
        updated_sequence, sequence_negative_log_likelihood, sequence_recovery = compiled_sequence_prediction(self.model_parameters, jnp.stack(sampling_random_keys), atoms, atom_mask, residue_index, sequence, fixed_residue_mask, jnp.asarray(self.temperature))
        redesigned_chain_arrays = [split_residue_arrays_by_chain(chain_names, chain_lengths, sequence=updated_sequence[candidate_index]) for candidate_index in range(candidate_count)]
        return [StructurePrediction(protein_complex={name: protein_complex[name].replace(sequence=candidate[name]['sequence'][:true_length]) for name, true_length in zip(chain_names, true_lengths)}, metrics={'score': sequence_negative_log_likelihood[candidate_index], 'seqid': sequence_recovery[candidate_index]}) for candidate_index, candidate in enumerate(redesigned_chain_arrays)]
