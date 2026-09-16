from typing import Callable, Protocol, runtime_checkable
import jax
import jax.numpy as jnp
import optax
from jax import Array
from bindcraft.loss import pairwise_atom_distances, pseudo_beta_coordinates
from bindcraft.prediction import residue_chain_ids, update_shared_sequences, concatenate_chain_arrays, collect_shared_chains, split_residue_arrays_by_chain
from bindcraft.protein import StructurePrediction, StructurePredictions, Protein, ResidueFlags, ProteinStates, has_residue_flag, is_binder_chain, real_residue_mask

_sum_state_gradients: Callable[[Array], Array] = lambda state_gradients: jnp.sum(state_gradients, axis=0)

@runtime_checkable
class SequenceOptimizer(Protocol):
    def update_sequence(self, protein_states: ProteinStates, accumulated_gradients: dict[str, list[Array]]) -> ProteinStates:
        ...

    def sequence_parameters(self) -> tuple[Array, Array, Array, Array]:
        ...

@runtime_checkable
class SequenceMutationSampler(Protocol):
    best_protein_states: ProteinStates | None

    def propose_sequence_mutation(self, protein_states: ProteinStates) -> ProteinStates:
        ...

    def select_best_sequence(self, design_loss: Array, predictions: StructurePredictions) -> ProteinStates:
        ...

def combine_sequence_gradients(accumulated_gradients: dict[str, list[Array]], combine_state_gradients: Callable[[Array], Array]=_sum_state_gradients) -> dict[str, Array]:
    return {name: combine_state_gradients(jnp.stack(state_gradients)) for name, state_gradients in accumulated_gradients.items()}

OMITTED_AMINO_ACID_LOGIT = -10000.0

def omitted_amino_acid_mask(sequence: Array) -> Array:
    return sequence <= OMITTED_AMINO_ACID_LOGIT / 2

def straight_through_one_hot(probabilities: Array) -> Array:
    one_hot_sequence = jax.nn.one_hot(jnp.argmax(probabilities, axis=-1), probabilities.shape[-1])
    return jax.lax.stop_gradient(one_hot_sequence - probabilities) + probabilities

def sequence_features_from_logits(logits: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array) -> Array:
    sequence_probabilities = jax.nn.softmax(logits * logit_scale / temperature)
    blended_logits = jnp.where(omitted_amino_acid_mask(logits), 0.0, logits)
    sequence_features = softmax_weight * sequence_probabilities + (1 - softmax_weight) * blended_logits
    return one_hot_weight * straight_through_one_hot(sequence_probabilities) + (1 - one_hot_weight) * sequence_features

def normalize_sequence_gradient(sequence_gradient: Array) -> Array:
    designed_residue_count = (jnp.square(sequence_gradient).sum(-1, keepdims=True) > 0).sum(-2, keepdims=True)
    return sequence_gradient * jnp.sqrt(designed_residue_count) / (jnp.linalg.norm(sequence_gradient) + 1e-07)

def _linear_sequence_schedule(start_weight: float, end_weight: float, iteration: int, iterations: int) -> Array:
    return jnp.asarray(start_weight + (end_weight - start_weight) * ((iteration + 1) / iterations))

def _quadratic_temperature_schedule(start_temperature: float, end_temperature: float, iteration: int, iterations: int) -> Array:
    return jnp.asarray(end_temperature + (start_temperature - end_temperature) * (1 - (iteration + 1) / iterations) ** 2)

class GradientSequenceOptimizer(SequenceOptimizer):
    def __init__(self, iterations: int, sequence_optimizer: optax.GradientTransformation | None=None, learning_rate: float=0.1, logit_scale: float=2.0, multi_chain_binders: tuple[tuple[str, ...], ...]=(), combine_state_gradients: Callable[[Array], Array]=_sum_state_gradients):
        self.iterations = iterations
        self.gradient_transform = sequence_optimizer if sequence_optimizer is not None else optax.sgd(learning_rate)
        self.logit_scale = jnp.asarray(logit_scale)
        self.optimizer_state = None
        self.sequence_update_count = 0
        self.multi_chain_binders = multi_chain_binders
        self.combine_state_gradients = combine_state_gradients
        self.designed_chain_cache: dict[tuple[str, ...], tuple[str, ...]] = {}

    def _sequence_parameters_at_step(self, sequence_update_count: int) -> tuple[Array, Array, Array]:
        raise NotImplementedError

    def sequence_parameters(self) -> tuple[Array, Array, Array, Array]:
        softmax_weight, one_hot_weight, temperature = self._sequence_parameters_at_step(self.sequence_update_count)
        return softmax_weight, one_hot_weight, temperature, self.logit_scale

    def update_sequence(self, protein_states: ProteinStates, accumulated_gradients: dict[str, list[Array]]) -> ProteinStates:
        chain_names, shared_chains = collect_shared_chains(protein_states)
        designed_chain_names = self.designed_chain_cache.get(chain_names)
        if designed_chain_names is None:
            designed_chain_names = tuple(name for name in chain_names if bool(has_residue_flag(shared_chains[name].flags, ResidueFlags.DESIGN).any()))
            self.designed_chain_cache[chain_names] = designed_chain_names
        chain_gradients = combine_sequence_gradients(accumulated_gradients, self.combine_state_gradients)
        for binder_chain_names in self.multi_chain_binders:
            shared_chain_gradient = self.combine_state_gradients(jnp.stack([chain_gradients[chain_name] for chain_name in binder_chain_names]))
            chain_gradients.update({name: shared_chain_gradient for name in binder_chain_names})
        sequence_logits = {name: shared_chains[name].sequence.astype(jnp.float32) for name in designed_chain_names}
        if self.optimizer_state is None:
            self.optimizer_state = self.gradient_transform.init(sequence_logits)
        softmax_weight, _, temperature = self._sequence_parameters_at_step(self.sequence_update_count)
        masked_chain_gradients = {name: jnp.where(has_residue_flag(shared_chains[name].flags, ResidueFlags.DESIGN)[:, None], chain_gradients[name].astype(jnp.float32), 0.0) for name in designed_chain_names}
        chain_lengths = tuple(len(shared_chains[name]) for name in designed_chain_names)
        normalized_gradient = normalize_sequence_gradient(jnp.concatenate([masked_chain_gradients[name] for name in designed_chain_names], axis=0))
        normalized_gradient = normalized_gradient * (1.0 - softmax_weight + softmax_weight * temperature)
        scaled_chain_gradients = {name: chain_arrays['flat'] for name, chain_arrays in split_residue_arrays_by_chain(designed_chain_names, chain_lengths, flat=normalized_gradient).items()}
        updates, self.optimizer_state = self.gradient_transform.update(scaled_chain_gradients, self.optimizer_state, sequence_logits)
        updated_logits = {name: jnp.where(omitted_amino_acid_mask(sequence_logits[name]), OMITTED_AMINO_ACID_LOGIT, sequence_logits[name] + updates[name]) for name in designed_chain_names}
        self.sequence_update_count += 1
        updated_chains = {name: shared_chains[name].replace(sequence=updated_logits[name].astype(shared_chains[name].sequence.dtype)) for name in designed_chain_names}
        return update_shared_sequences(protein_states, updated_chains)

class LogitSequenceOptimizer(GradientSequenceOptimizer):
    def __init__(self, iterations: int, start_softmax_weight: float=0.0, end_softmax_weight: float=1.0, **kwargs):
        super().__init__(iterations=iterations, **kwargs)
        self.start_softmax_weight, self.end_softmax_weight = start_softmax_weight, end_softmax_weight

    def _sequence_parameters_at_step(self, sequence_update_count: int) -> tuple[Array, Array, Array]:
        iteration = min(sequence_update_count, self.iterations - 1)
        return _linear_sequence_schedule(self.start_softmax_weight, self.end_softmax_weight, iteration, self.iterations), jnp.asarray(0.0), jnp.asarray(1.0)

class SequenceAnnealingOptimizer(GradientSequenceOptimizer):
    def __init__(self, iterations: int, start_temperature: float=1.0, end_temperature: float=0.01, **kwargs):
        super().__init__(iterations=iterations, **kwargs)
        self.start_temperature, self.end_temperature = start_temperature, end_temperature

    def _sequence_parameters_at_step(self, sequence_update_count: int) -> tuple[Array, Array, Array]:
        iteration = min(sequence_update_count, self.iterations - 1)
        return jnp.asarray(1.0), jnp.asarray(0.0), _quadratic_temperature_schedule(self.start_temperature, self.end_temperature, iteration, self.iterations)

class OneHotSequenceOptimizer(GradientSequenceOptimizer):
    def __init__(self, iterations: int, temperature: float=0.01, **kwargs):
        super().__init__(iterations=iterations, **kwargs)
        self.temperature = temperature

    def _sequence_parameters_at_step(self, sequence_update_count: int) -> tuple[Array, Array, Array]:
        return jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(self.temperature)

def split_chain_confidence(prediction: StructurePrediction, key: str) -> dict[str, Array]:
    chain_names = tuple(sorted(prediction.protein_complex))
    chain_lengths = tuple(len(prediction.protein_complex[name]) for name in chain_names)
    chain_confidence_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, metric=prediction.metrics[key])
    return {name: chain_arrays['metric'] for name, chain_arrays in chain_confidence_arrays.items()}

INTERFACE_CUTOFF = 8.0

def pseudo_beta_interface_mask(protein_complex: dict[str, Protein], cutoff: float=INTERFACE_CUTOFF) -> Array:
    binder_chains = tuple(name for name in sorted(protein_complex) if is_binder_chain(name))
    target_chains = tuple(name for name in sorted(protein_complex) if not is_binder_chain(name))
    target_coordinates, target_mask = (jnp.concatenate(atom_arrays) for atom_arrays in zip(*(pseudo_beta_coordinates(protein_complex[name]) for name in target_chains)))
    facing_target = {name: ((pairwise_atom_distances(pseudo_beta_coordinates(protein_complex[name])[0], target_coordinates) < cutoff) * target_mask[None, :]).any(-1) & real_residue_mask(protein_complex[name].flags) for name in binder_chains}
    return jnp.concatenate([facing_target.get(name, jnp.zeros(len(protein_complex[name]), dtype=bool)) for name in sorted(protein_complex)])

def interface_confidence_weights(prediction: StructurePrediction, cutoff: float=INTERFACE_CUTOFF) -> dict[str, Array] | None:
    protein_complex = prediction.protein_complex
    binder_chains = tuple(name for name in protein_complex if is_binder_chain(name))
    if 'iptm_per_residue' not in prediction.metrics or not binder_chains or len(binder_chains) == len(protein_complex):
        return None
    interface_mask = pseudo_beta_interface_mask(protein_complex, cutoff)
    interface_confidence = jnp.where(interface_mask, 1.0 - prediction.metrics['iptm_per_residue'], 0.0)
    interface_mean = interface_confidence.sum() / jnp.maximum(interface_mask.sum(), 1.0)
    weights = jnp.where(interface_mask & (interface_mean > 0), interface_confidence / jnp.maximum(interface_mean, 1e-08), 1.0)
    chain_names = tuple(sorted(protein_complex))
    return {name: chain_arrays['weights'] for name, chain_arrays in split_residue_arrays_by_chain(chain_names, tuple(len(protein_complex[name]) for name in chain_names), weights=weights).items()}

class SemigreedySequenceSampler(SequenceMutationSampler):
    def __init__(self, key: Array | None=None, multi_chain_binders: tuple[tuple[str, ...], ...]=(), mutation_weighting: str='plddt'):
        self.key = jax.random.PRNGKey(0) if key is None else key
        self.best_protein_states: ProteinStates | None = None
        self.best_design_loss: Array | None = None
        self.chain_plddt: dict[str, Array] = {}
        self.chain_interface_weights: dict[str, Array] = {}
        self.omitted_amino_acid_masks: dict[str, Array] = {}
        self.multi_chain_binders = multi_chain_binders
        self.mutation_weighting = mutation_weighting

    def multi_chain_binder_group(self, chain_name: str) -> tuple[str, ...]:
        return next((group for group in self.multi_chain_binders if chain_name in group), (chain_name,))

    def interface_weights(self, chain_names: tuple[str, ...], shared_chains: dict[str, Protein]) -> Array:
        if self.mutation_weighting != 'interface_iptm' or not self.chain_interface_weights:
            return jnp.asarray(1.0)
        weights = {name: self.chain_interface_weights.get(name, jnp.ones(len(shared_chains[name]), dtype=jnp.float32)) for name in chain_names}
        for binder_chain_names in self.multi_chain_binders:
            protomer_weights = jnp.mean(jnp.stack([weights[name] for name in binder_chain_names]), axis=0)
            weights.update({name: protomer_weights if name == binder_chain_names[0] else jnp.ones_like(protomer_weights) for name in binder_chain_names})
        return jnp.concatenate([weights[name] for name in chain_names])

    def propose_sequence_mutation(self, protein_states: ProteinStates) -> ProteinStates:
        chain_names, shared_chains = collect_shared_chains(protein_states)
        chain_lengths = tuple(len(shared_chains[name]) for name in chain_names)
        flags = concatenate_chain_arrays(chain_names, shared_chains, 'flags')['flags']
        designed_residue_mask = has_residue_flag(flags, ResidueFlags.DESIGN)
        chain_plddt = {name: self.chain_plddt.get(name, jnp.zeros(len(shared_chains[name]), dtype=jnp.float32)) for name in chain_names}
        #for oligomers and multi-chain binders
        for binder_chain_names in self.multi_chain_binders:
            protomer_plddt = jnp.mean(jnp.stack([chain_plddt[name] for name in binder_chain_names]), axis=0)
            chain_plddt.update({name: protomer_plddt if name == binder_chain_names[0] else jnp.ones_like(protomer_plddt) for name in binder_chain_names})
        plddt = jnp.concatenate([chain_plddt[name] for name in chain_names])
        residue_mutation_probabilities = jnp.where(designed_residue_mask, (1.0 - plddt) * self.interface_weights(chain_names, shared_chains), 0.0)
        residue_mutation_probabilities = residue_mutation_probabilities / (residue_mutation_probabilities.sum() + 1e-08)
        self.key, residue_random_key, amino_acid_random_key = jax.random.split(self.key, 3)
        mutation_residue_index = int(jax.random.choice(residue_random_key, residue_mutation_probabilities.shape[0], p=residue_mutation_probabilities))
        mutation_chain_index = int(residue_chain_ids(chain_lengths)[mutation_residue_index])
        mutation_chain_name = chain_names[mutation_chain_index]
        chain_mutation_index = mutation_residue_index - sum(chain_lengths[:mutation_chain_index])
        sequence = shared_chains[mutation_chain_name].sequence
        amino_acid_count = sequence.shape[-1]
        chain_omitted_amino_acid_mask = self.omitted_amino_acid_masks.setdefault(mutation_chain_name, omitted_amino_acid_mask(sequence).any(0))
        current_amino_acid = jnp.argmax(sequence[chain_mutation_index])
        exclude_current_amino_acid = jax.nn.one_hot(current_amino_acid, amino_acid_count) * -100000000.0
        mutation_probabilities = jax.nn.softmax(jnp.where(chain_omitted_amino_acid_mask, OMITTED_AMINO_ACID_LOGIT, sequence[chain_mutation_index] + exclude_current_amino_acid))
        mutated_amino_acid = jax.random.choice(amino_acid_random_key, amino_acid_count, p=mutation_probabilities)
        mutated_residue_sequence = jax.nn.one_hot(mutated_amino_acid, amino_acid_count).astype(sequence.dtype)
        mutated_chains = {name: shared_chains[name].replace(sequence=shared_chains[name].sequence.at[chain_mutation_index].set(mutated_residue_sequence)) for name in self.multi_chain_binder_group(mutation_chain_name)}
        return update_shared_sequences(protein_states, mutated_chains)

    def select_best_sequence(self, design_loss: Array, predictions: StructurePredictions) -> ProteinStates:
        predicted_chain_confidence: dict[str, list[Array]] = {}
        for prediction in predictions.values():
            for name, plddt in split_chain_confidence(prediction, 'plddt').items():
                predicted_chain_confidence.setdefault(name, []).append(plddt)
        self.chain_plddt.update({chain_name: jnp.mean(jnp.stack(state_confidence), axis=0) for chain_name, state_confidence in predicted_chain_confidence.items()})
        if self.mutation_weighting == 'interface_iptm':
            predicted_interface_weights: dict[str, list[Array]] = {}
            for prediction in predictions.values():
                for name, weights in (interface_confidence_weights(prediction) or {}).items():
                    predicted_interface_weights.setdefault(name, []).append(weights)
            self.chain_interface_weights.update({chain_name: jnp.mean(jnp.stack(state_weights), axis=0) for chain_name, state_weights in predicted_interface_weights.items()})
        if self.best_design_loss is None or design_loss < self.best_design_loss:
            self.best_protein_states = {name: prediction.protein_complex for name, prediction in predictions.items()}
            self.best_design_loss = design_loss
        return self.best_protein_states
