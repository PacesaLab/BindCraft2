from collections import OrderedDict
from typing import Protocol, runtime_checkable
import jax.numpy as jnp
from jax import Array
from bindcraft.loss import DesignLoss
from bindcraft.protein import StructurePredictions, Protein, ProteinStates

@runtime_checkable
class ProteinPredictor(Protocol):
    def predict(self, protein_states: ProteinStates, model: str | None=None) -> StructurePredictions:
        ...

@runtime_checkable
class DifferentiableProteinPredictor(ProteinPredictor, Protocol):
    def sequence_gradients(self, protein_states: ProteinStates, losses: dict[str, DesignLoss], model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=0.0, temperature: float=1.0, logit_scale: float=2.0, reference_predictions: StructurePredictions | None=None) -> tuple[StructurePredictions, dict[str, Array], Array]:
        ...

def collect_shared_chains(protein_states: ProteinStates) -> tuple[tuple[str, ...], dict[str, Protein]]:
    shared_chains: dict[str, Protein] = {}
    for protein_complex in protein_states.values():
        for chain_name, protein in protein_complex.items():
            shared_chains.setdefault(chain_name, protein)
    return tuple(sorted(shared_chains)), shared_chains

def update_shared_sequences(protein_states: ProteinStates, updated_chains: dict[str, Protein]) -> ProteinStates:
    return {state_name: {chain_name: protein.replace(sequence=updated_chains[chain_name].sequence) if chain_name in updated_chains else protein for chain_name, protein in protein_complex.items()} for state_name, protein_complex in protein_states.items()}

class CompiledModelCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.compiled_functions: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self.compiled_functions:
            return None
        self.compiled_functions.move_to_end(key)
        return self.compiled_functions[key]

    def set(self, key, value) -> None:
        self.compiled_functions[key] = value
        self.compiled_functions.move_to_end(key)
        if len(self.compiled_functions) > self.max_size:
            self.compiled_functions.popitem(last=False)

def residue_chain_ids(chain_lengths: tuple[int, ...]) -> Array:
    return jnp.concatenate([jnp.full((chain_length,), chain_index, dtype=jnp.int32) for chain_index, chain_length in enumerate(chain_lengths)])

def split_residue_arrays_by_chain(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...], **fields: Array) -> dict[str, dict[str, Array]]:
    chain_arrays, residue_start = {}, 0
    for chain_name, chain_length in zip(chain_names, chain_lengths):
        chain_arrays[chain_name] = {field: value[residue_start:residue_start + chain_length] for field, value in fields.items()}
        residue_start += chain_length
    return chain_arrays

def concatenate_chain_arrays(chain_names: tuple[str, ...], protein_complex: dict[str, Protein], *fields: str) -> dict[str, Array]:
    return {field: jnp.concatenate([getattr(protein_complex[name], field) for name in chain_names], axis=0) for field in fields}
