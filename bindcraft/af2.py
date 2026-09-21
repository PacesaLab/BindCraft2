import copy
import fcntl
import hashlib
import os
from contextlib import contextmanager
from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array
from bindcraft.af.alphafold.common import confidence, residue_constants
from bindcraft.af.alphafold.model import config as af_config, data as af_data, model as af_model, modules as af_modules
from bindcraft.af import accel
from bindcraft.prediction import DifferentiableProteinPredictor, CompiledModelCache, residue_chain_ids, concatenate_chain_arrays, collect_shared_chains, split_residue_arrays_by_chain
from bindcraft.loss import DesignLoss, frozen_interface_arguments, renamed_loss_name, renamed_state_losses
from bindcraft.sequence_optimization import sequence_features_from_logits
from bindcraft.protein import ATOM_INDEX, ATOM_NAMES, BINDER_ALONE, BINDER_CHAIN_PREFIX, StructurePrediction, StructurePredictions, Protein, ResidueFlags, ProteinStates, has_residue_flag, is_binder_chain, is_target_chain, kabsch, real_residue_weights

@contextmanager
def one_worker_compiles(compile_shape: tuple):
    cache_directory = os.environ.get('JAX_COMPILATION_CACHE_DIR')
    if not cache_directory:
        yield
        return
    os.makedirs(cache_directory, exist_ok=True)
    shape_digest = hashlib.sha256(repr(compile_shape).encode()).hexdigest()[:16]
    with open(os.path.join(cache_directory, f'compile_{shape_digest}.lock'), 'w') as compile_lock:
        fcntl.flock(compile_lock, fcntl.LOCK_EX)
        yield

def align_prediction_to_target_template(predicted_atom_positions: Array, predicted_atom_mask: Array, template_atom_positions: Array, template_atom_mask: Array, flags: Array) -> Array:
    ca_atom_index = ATOM_INDEX['CA']
    template_residue_mask = has_residue_flag(flags, ResidueFlags.TEMPLATE) & ~has_residue_flag(flags, ResidueFlags.DESIGN)
    alignment_weights = (template_residue_mask & template_atom_mask[:, ca_atom_index]).astype(jnp.float32)
    predicted_ca_coordinates = jax.lax.stop_gradient(predicted_atom_positions[:, ca_atom_index]).astype(jnp.float32)
    template_ca_coordinates = template_atom_positions[:, ca_atom_index].astype(jnp.float32)
    alignment_rotation, prediction_center, template_center = kabsch(predicted_ca_coordinates, template_ca_coordinates, alignment_weights)
    aligned_atom_positions = (predicted_atom_positions.astype(jnp.float32) - prediction_center) @ alignment_rotation.T + template_center
    return jnp.where(predicted_atom_mask[:, :, None], aligned_atom_positions, 0.0).astype(predicted_atom_positions.dtype)

def prepare_design_sequence_features(sequence: Array, flags: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array) -> tuple[Array, Array]:
    designed_residue_mask = has_residue_flag(flags, ResidueFlags.DESIGN)[:, None]
    sequence_features = jnp.where(designed_residue_mask, sequence_features_from_logits(sequence, softmax_weight, one_hot_weight, temperature, logit_scale), sequence)
    one_hot_weight_value = jnp.ones((), dtype=jnp.result_type(softmax_weight))
    sequence_profile = jnp.where(designed_residue_mask, sequence_features_from_logits(sequence, softmax_weight, one_hot_weight_value, temperature, logit_scale), sequence)
    return sequence_features, sequence_profile

DEFAULT_LENGTH_BUCKET = 32

def campaign_length_bucket(settings: dict) -> int:
    return int(settings.get('length_bucket_size') or DEFAULT_LENGTH_BUCKET)

def padded_prediction_length(residue_count: int, bucket_size: int=DEFAULT_LENGTH_BUCKET) -> int:
    return -(-residue_count // bucket_size) * bucket_size

def padded_prediction_complex(protein_complex: dict[str, Protein], bucket_size: int=DEFAULT_LENGTH_BUCKET, target_pad_length: int=0) -> dict[str, Protein]:
    return {chain_name: protein.padded_to(padded_prediction_length(len(protein), bucket_size) if bool(has_residue_flag(protein.flags, ResidueFlags.DESIGN).any()) else target_pad_length) for chain_name, protein in protein_complex.items()}

def pad_design_chains(protein_states: ProteinStates, bucket_size: int=DEFAULT_LENGTH_BUCKET, target_pad_length: int=0) -> ProteinStates:
    return {state_name: padded_prediction_complex(protein_complex, bucket_size, target_pad_length) for state_name, protein_complex in protein_states.items()}

def shared_target_pad_length(target_states: ProteinStates, target_chain: str='target', bucket_size: int=DEFAULT_LENGTH_BUCKET) -> int:
    target_lengths = [len(protein) for protein_complex in target_states.values() for name, protein in protein_complex.items() if is_target_chain(name, target_chain)]
    return padded_prediction_length(max(target_lengths), bucket_size) if len(set(target_lengths)) > 1 else 0

def canonical_state_names(protein_states: ProteinStates) -> dict[str, str]:
    return {name: name if name == BINDER_ALONE else f'state_{index}' for index, name in enumerate(sorted(protein_states))}

def canonical_chain_names(protein_states: ProteinStates, state_names: dict[str, str]) -> dict[str, str]:
    chain_names = {}
    for state_name, protein_complex in protein_states.items():
        suffix = f'_{state_name}'
        for chain_name in protein_complex:
            chain_names[chain_name] = f'{chain_name[:-len(suffix)]}_{state_names[state_name]}' if chain_name.endswith(suffix) else chain_name
    return chain_names

def renamed_protein_states(protein_states: ProteinStates, state_names: dict[str, str], chain_names: dict[str, str]) -> ProteinStates:
    return {state_names[state_name]: {chain_names[chain_name]: protein for chain_name, protein in protein_complex.items()} for state_name, protein_complex in protein_states.items()}

def real_residue_positions(chain_lengths: tuple[int, ...], true_lengths: tuple[int, ...]) -> Array:
    positions, chain_start = [], 0
    for chain_length, true_length in zip(chain_lengths, true_lengths):
        positions.append(jnp.arange(chain_start, chain_start + true_length, dtype=jnp.int32))
        chain_start += chain_length
    return jnp.concatenate(positions)

def trim_padded_metrics(metrics: dict[str, Array], real_positions: Array) -> dict[str, Array]:
    return {name: value if value.ndim == 0 else value[real_positions] if value.ndim == 1 else value[real_positions[:, None], real_positions[None, :]] for name, value in metrics.items()}

SIDECHAIN_ATOMS = tuple(atom_name not in ('N', 'CA', 'C', 'CB', 'O') for atom_name in ATOM_NAMES)

def flexible_target_residues(residue_index: Array, target_template_mask: Array, target_flexibility: float) -> Array:
    return target_template_mask & (jnp.floor((residue_index + 1) * target_flexibility) > jnp.floor(residue_index * target_flexibility))

def target_template_features(aatype: Array, atoms: Array, atom_mask: Array, flags: Array, residue_index: Array, target_flexibility: float=0.0) -> dict[str, Array]:
    template_residue_mask = has_residue_flag(flags, ResidueFlags.TEMPLATE)
    flexible_mask = flexible_target_residues(residue_index, template_residue_mask & ~has_residue_flag(flags, ResidueFlags.DESIGN), target_flexibility) if target_flexibility else jnp.zeros_like(template_residue_mask)
    template_aatype = jnp.where(template_residue_mask & ~flexible_mask, aatype, 21)
    template_all_atom_mask = atom_mask & template_residue_mask[:, None] & ~(flexible_mask[:, None] & jnp.asarray(SIDECHAIN_ATOMS))
    template_all_atom_positions = jnp.where(template_residue_mask[:, None, None], atoms, 0.0)
    template_beta_coordinates, template_beta_mask = af_modules.pseudo_beta_fn(template_aatype, template_all_atom_positions, template_all_atom_mask)
    return {'template_aatype': template_aatype[None].astype(jnp.int32), 'template_all_atom_positions': template_all_atom_positions[None].astype(jnp.float32), 'template_all_atom_mask': template_all_atom_mask[None].astype(jnp.float32), 'template_pseudo_beta': template_beta_coordinates[None].astype(jnp.float32), 'template_pseudo_beta_mask': template_beta_mask[None].astype(jnp.float32), 'template_mask': jnp.ones((1,), dtype=jnp.float32), 'mask_template_interchain': jnp.asarray(True)}

def residue_entity_ids(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...], multi_chain_binders: tuple[tuple[str, ...], ...]=()) -> Array:
    shared_entity = {name: min(group) for group in multi_chain_binders for name in group}
    entity_names = sorted({shared_entity.get(name, name) for name in chain_names})
    return jnp.concatenate([jnp.full((length,), entity_names.index(shared_entity.get(name, name)), dtype=jnp.int32) for name, length in zip(chain_names, chain_lengths)])

def interface_asym_ids(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...]) -> Array:
    assembly_name = lambda chain_name: BINDER_CHAIN_PREFIX if is_binder_chain(chain_name) else chain_name
    assembly_names = sorted({assembly_name(name) for name in chain_names})
    if len(assembly_names) < 2:
        return residue_chain_ids(chain_lengths)
    return jnp.concatenate([jnp.full((length,), assembly_names.index(assembly_name(name)), dtype=jnp.int32) for name, length in zip(chain_names, chain_lengths)])

def cyclic_sequence_offsets(residue_index: Array, asym_id: Array, flags: Array, seq_mask: Array, offset_mode: str='direction') -> Array:
    offsets = residue_index[:, None] - residue_index[None, :]
    same_chain = asym_id[:, None] == asym_id[None, :]
    chain_lengths = (same_chain * seq_mask[None, :]).sum(-1)
    cyclic_mask = has_residue_flag(flags, ResidueFlags.CYCLIC) * seq_mask
    cyclic_pairs = same_chain * cyclic_mask[:, None] * cyclic_mask[None, :]
    cyclic_distance = jnp.minimum(jnp.abs(offsets), chain_lengths[:, None] - jnp.abs(offsets))
    if offset_mode in ('direction', 'neighbours'):
        cyclic_distance = jnp.where(cyclic_distance < jnp.abs(offsets), -cyclic_distance, cyclic_distance)
    if offset_mode == 'neighbours':
        cyclic_distance = jnp.where(jnp.abs(cyclic_distance) > 2, 32 * jnp.sign(cyclic_distance), cyclic_distance)
    return jnp.where(cyclic_pairs, cyclic_distance * jnp.sign(offsets), offsets).astype(jnp.int32)

def alphafold_input_features(sequence: Array, sequence_profile: Array, atoms: Array, atom_mask: Array, residue_index: Array, asym_id: Array, seq_mask: Array, flags: Array, dropout: Array, cyclic_offset_mode: str='direction', entity_id: Array | None=None, target_flexibility: float=0.0, bigbang_initialization: bool=False) -> dict[str, Array]:
    length = sequence.shape[0]
    aatype = sequence.argmax(-1).astype(jnp.int32)
    mask = seq_mask[:, None]
    entity_id = asym_id if entity_id is None else entity_id
    msa_feat = jnp.zeros((1, length, 49)).at[:, :, 0:20].set(sequence[None]).at[:, :, 25:45].set(sequence_profile[None])
    return {'aatype': aatype, 'residue_index': residue_index, 'offset': cyclic_sequence_offsets(residue_index, asym_id, flags, seq_mask, cyclic_offset_mode), 'asym_id': asym_id, 'entity_id': entity_id, 'sym_id': asym_id, 'seq_mask': seq_mask, 'msa_mask': jnp.ones((1, length)) * seq_mask[None], 'target_feat': sequence, 'msa_feat': msa_feat, 'extra_msa': jnp.zeros((1, length), dtype=jnp.int32), 'extra_msa_mask': jnp.zeros((1, length), dtype=jnp.float32), 'extra_has_deletion': jnp.zeros((1, length), dtype=jnp.float32), 'extra_deletion_value': jnp.zeros((1, length)), 'all_atom_positions': atoms.astype(jnp.float32), 'all_atom_mask': atom_mask.astype(jnp.float32), 'atom14_atom_exists': jnp.where(mask, jnp.asarray(residue_constants.restype_atom14_mask)[aatype], 0), 'atom37_atom_exists': jnp.where(mask, jnp.asarray(residue_constants.restype_atom37_mask)[aatype], 0), 'residx_atom14_to_atom37': jnp.where(mask, jnp.asarray(residue_constants.restype_atom14_to_atom37)[aatype], 0), 'residx_atom37_to_atom14': jnp.where(mask, jnp.asarray(residue_constants.restype_atom37_to_atom14)[aatype], 0), 'use_dropout': dropout, 'prev': {'prev_msa_first_row': jnp.zeros((length, 256)), 'prev_pair': jnp.zeros((length, length, 128)), 'prev_pos': jnp.zeros((length, 37, 3))}, **({'initial_atom_pos': atoms.astype(jnp.float32)} if bigbang_initialization else {}), **target_template_features(aatype, atoms, atom_mask, flags, residue_index, target_flexibility)}

def recycled_alphafold_outputs(alphafold_runner: af_model.RunModel, model_parameters: Array, key: Array, model_inputs: dict, num_recycle: int) -> dict:
    recycle_keys = jax.random.split(key, num_recycle + 1)
    previous_state = model_inputs['prev']
    for recycle_key in recycle_keys[:-1]:
        previous_state = jax.lax.stop_gradient(alphafold_runner.apply(model_parameters, recycle_key, {**model_inputs, 'prev': previous_state})['prev'])
        if 'initial_atom_pos' in model_inputs:
            model_inputs = {**model_inputs, 'initial_atom_pos': previous_state['prev_pos']}
    return alphafold_runner.apply(model_parameters, recycle_keys[-1], {**model_inputs, 'prev': previous_state})

def alphafold_prediction_metrics(alphafold_outputs: dict, seq_mask: Array, interface_asym_id: Array) -> dict[str, Array]:
    metrics: dict[str, Array] = {}
    if 'predicted_lddt' in alphafold_outputs:
        confidence_logits = alphafold_outputs['predicted_lddt']['logits']
        bin_width = 1.0 / confidence_logits.shape[-1]
        bin_centers = jnp.arange(0.5 * bin_width, 1.0, bin_width)
        metrics['plddt'] = (jax.nn.softmax(confidence_logits, axis=-1) * bin_centers[None, :]).sum(-1)
    if 'predicted_aligned_error' in alphafold_outputs:
        pae_head = alphafold_outputs['predicted_aligned_error']
        probabilities = jax.nn.softmax(pae_head['logits'], axis=-1)
        pae_bin_edges = pae_head['breaks']
        pae_bin_width = pae_bin_edges[1] - pae_bin_edges[0]
        bin_centers = jnp.append(pae_bin_edges + pae_bin_width / 2, pae_bin_edges[-1] + 1.5 * pae_bin_width)
        pae = (probabilities * bin_centers).sum(-1)
        metrics['pae'] = (pae + pae.T) / 2
        metrics['ptm'] = confidence.predicted_tm_score(pae_head['logits'], pae_head['breaks'], residue_weights=seq_mask, use_jnp=True)
        metrics['iptm'], metrics['iptm_per_residue'] = confidence.predicted_tm_score(pae_head['logits'], pae_head['breaks'], residue_weights=seq_mask, asym_id=interface_asym_id, use_jnp=True, return_per_alignment=True)
    if 'experimentally_resolved' in alphafold_outputs:
        metrics['experimentally_resolved_ca'] = jax.nn.sigmoid(alphafold_outputs['experimentally_resolved']['logits'][:, ATOM_INDEX['CA']])
    if 'distogram' in alphafold_outputs:
        metrics['distogram'] = alphafold_outputs['distogram']['logits']
    return metrics

def trim_prediction_padding(value: Array, residue_count: int) -> Array:
    if value.ndim == 0:
        return value
    if value.ndim == 1:
        return value[:residue_count]
    return value[:residue_count, :residue_count]

def protein_state_shapes(protein_states: ProteinStates) -> tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...]:
    complex_shapes = []
    for state_name in sorted(protein_states):
        chain_names = tuple(sorted(protein_states[state_name]))
        chain_lengths = tuple(len(protein_states[state_name][chain_name]) for chain_name in chain_names)
        complex_shapes.append((state_name, chain_names, chain_lengths))
    return tuple(complex_shapes)

MONOMER_CHAIN_GAP = 49

def monomer_chain_break_indices(chain_lengths: tuple[int, ...], residue_index: Array) -> Array:
    # Preserve internal receptor breaks; only replace jumps between top-level chains.
    steps = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.diff(residue_index)])
    chain_starts = jnp.cumsum(jnp.asarray(chain_lengths[:-1], dtype=jnp.int32))
    steps = steps.at[chain_starts].set(MONOMER_CHAIN_GAP + 1)
    return jnp.cumsum(steps)

def alphafold_model_family(model_name: str) -> tuple:
    if 'multimer' in model_name:
        return ('multimer',)
    return 'monomer', tuple(sorted(af_config.CONFIG_DIFFS.get(model_name, {}).items()))

SUBBATCH_RESIDUE_THRESHOLD = 384
LARGE_COMPLEX_SUBBATCH_SIZE = 4

def resolve_subbatch_size(residue_count: int, subbatch_size: int | None | str='auto') -> int | None:
    if subbatch_size != 'auto':
        return subbatch_size
    return LARGE_COMPLEX_SUBBATCH_SIZE if residue_count > SUBBATCH_RESIDUE_THRESHOLD else None

class AlphaFoldDesignModel(DifferentiableProteinPredictor):
    def __init__(self, presets: str | tuple[str, ...]='model_1_ptm', data_dir: str | None=None, key: Array | None=None, max_cache_size: int=8, models: tuple[str, ...] | None=None, num_recycle: int=1, cyclic_offset_mode: str='direction', subbatch_size: int | None | str='auto', length_bucket_size: int=DEFAULT_LENGTH_BUCKET, attention_backend: str='auto', use_cueq: bool=False, dropout: bool=True, multi_chain_binders: tuple[tuple[str, ...], ...]=(), target_pad_length: int=0, target_flexibility: float=0.0, bigbang_initialization: bool=False):
        self.cyclic_offset_mode = cyclic_offset_mode
        self.target_pad_length = target_pad_length
        self.multi_chain_binders = multi_chain_binders
        self.presets = (presets,) if isinstance(presets, str) else tuple(presets)
        self.models = self.presets if models is None else tuple(models)
        self.key = jax.random.PRNGKey(0) if key is None else key
        self.dropout = dropout
        #for the desperation autotuner only
        self.num_recycle = num_recycle
        self.target_flexibility = target_flexibility
        self.bigbang_initialization = bigbang_initialization
        self.subbatch_size = subbatch_size
        self.length_bucket_size = length_bucket_size
        self.attention_backend = accel.supported_attention_backend(attention_backend)
        self.use_cueq = use_cueq
        self.model_families: dict[str, tuple] = {}
        self.model_family_models: dict[tuple, str] = {}
        self.alphafold_runners: dict[tuple, af_model.RunModel] = {}
        self.model_parameters: dict[str, object] = {}
        for model_name in self.presets:
            model_family = alphafold_model_family(model_name)
            self.model_families[model_name] = model_family
            self.model_family_models.setdefault(model_family, model_name)
            self._alphafold_runner(model_family, resolve_subbatch_size(4, self.subbatch_size))
            model_parameters = af_data.get_model_haiku_params(model_name=model_name, data_dir=data_dir, fuse=True) if data_dir else None
            if model_parameters is None:
                raise ValueError(f'no AlphaFold parameters for {model_name!r} in data_dir {data_dir!r}; run "bindcraft fetch-weights" to download them')
            self.model_parameters[model_name] = model_parameters
        self.prediction_compile_cache = CompiledModelCache(max_cache_size)
        self.gradient_compile_cache = CompiledModelCache(max_cache_size)

    def _alphafold_runner(self, model_family: tuple, subbatch_size: int | None) -> af_model.RunModel:
        runner_key = model_family, subbatch_size, self.attention_backend, self.use_cueq
        alphafold_runner = self.alphafold_runners.get(runner_key)
        if alphafold_runner is None:
            model_name = self.model_family_models[model_family]
            use_multimer = 'multimer' in model_name
            model_config = copy.deepcopy(af_config.model_config(model_name))
            model_config.model.global_config.use_dgram = False
            model_config.model.global_config.use_remat = True
            model_config.model.global_config.bfloat16 = True
            model_config.model.global_config.subbatch_size = subbatch_size
            model_config.model.global_config.attention_backend = self.attention_backend
            model_config.model.global_config.use_cueq = self.use_cueq
            model_config.model.num_recycle = 0
            alphafold_runner = af_model.RunModel(model_config, params=None, use_multimer=use_multimer)
            self.alphafold_runners[runner_key] = alphafold_runner
        return alphafold_runner

    def _sample_design_model(self) -> str:
        self.key, model_random_key = jax.random.split(self.key)
        return self.models[int(jax.random.randint(model_random_key, (), 0, len(self.models)))]

    def _resolve_model_name(self, model: str | None) -> str:
        if model is None:
            return self._sample_design_model()
        if model not in self.model_families:
            raise KeyError(f'model {model!r} not in the AlphaFold model pool {self.presets}')
        return model

    def _compiled_complex_prediction(self, model: str, padded_length: int) -> Callable:
        model_family = self.model_families[model]
        subbatch_size = resolve_subbatch_size(padded_length, self.subbatch_size)
        cache_key = model_family, padded_length, subbatch_size, self.multi_chain_binders, self.num_recycle, self.target_flexibility, self.bigbang_initialization
        compiled_prediction = self.prediction_compile_cache.get(cache_key)
        if compiled_prediction is None:
            alphafold_runner = self._alphafold_runner(model_family, subbatch_size)
            def predict_complex_arrays(model_parameters: Array, key: Array, sequence: Array, atoms: Array, atom_mask: Array, residue_index: Array, asym_id: Array, entity_id: Array, interface_asym_id: Array, seq_mask: Array, flags: Array, dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array):
                sequence_features, sequence_profile = prepare_design_sequence_features(sequence, flags, softmax_weight, one_hot_weight, temperature, logit_scale)
                model_inputs = alphafold_input_features(sequence_features, sequence_profile, atoms, atom_mask, residue_index, asym_id, seq_mask, flags, dropout, self.cyclic_offset_mode, entity_id, self.target_flexibility, self.bigbang_initialization)
                alphafold_outputs = recycled_alphafold_outputs(alphafold_runner, model_parameters, key, model_inputs, self.num_recycle)
                predicted_atom_positions = alphafold_outputs['structure_module']['final_atom_positions'].astype(jnp.float16)
                predicted_atom_mask = alphafold_outputs['structure_module']['final_atom_mask'].astype(bool)
                return predicted_atom_positions, predicted_atom_mask, alphafold_prediction_metrics(alphafold_outputs, seq_mask, interface_asym_id)
            compiled_prediction = jax.jit(predict_complex_arrays)
            self.prediction_compile_cache.set(cache_key, compiled_prediction)
        return compiled_prediction

    def predict(self, protein_states: ProteinStates, model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=1.0, temperature: float=0.01, logit_scale: float=2.0) -> StructurePredictions:
        model = self._resolve_model_name(model)
        return {name: self._predict_complex(protein_complex, model, softmax_weight, one_hot_weight, temperature, logit_scale) for name, protein_complex in protein_states.items()}

    def _predict_complex(self, protein_complex: dict[str, Protein], model: str, softmax_weight: float, one_hot_weight: float, temperature: float, logit_scale: float) -> StructurePrediction:
        chain_names = tuple(sorted(protein_complex))
        true_lengths = tuple(len(protein_complex[name]) for name in chain_names)
        padded_complex = padded_prediction_complex(protein_complex, self.length_bucket_size, self.target_pad_length) if self.target_pad_length else protein_complex
        chain_lengths = tuple(len(padded_complex[name]) for name in chain_names)
        residue_count = sum(chain_lengths)
        padded_residue_count = padded_prediction_length(residue_count, self.length_bucket_size)
        padding_length = padded_residue_count - residue_count
        chain_arrays = concatenate_chain_arrays(chain_names, padded_complex, 'sequence', 'atoms', 'atom_mask', 'flags', 'residue_index')
        sequence, atoms, atom_mask, flags, residue_index = chain_arrays['sequence'], chain_arrays['atoms'], chain_arrays['atom_mask'], chain_arrays['flags'], chain_arrays['residue_index']
        asym_id = residue_chain_ids(chain_lengths)
        entity_id = residue_entity_ids(chain_names, chain_lengths, self.multi_chain_binders)
        interface_asym_id = interface_asym_ids(chain_names, chain_lengths)
        if self.model_families[model][0] == 'monomer' and len(chain_names) > 1:
            residue_index = monomer_chain_break_indices(chain_lengths, residue_index)
        seq_mask = real_residue_weights(flags)
        if padding_length:
            sequence = jnp.pad(sequence, [[0, padding_length], [0, 0]])
            atoms = jnp.pad(atoms, [[0, padding_length], [0, 0], [0, 0]])
            atom_mask = jnp.pad(atom_mask, [[0, padding_length], [0, 0]])
            flags = jnp.pad(flags, [0, padding_length])
            residue_index = jnp.pad(residue_index, [0, padding_length])
            asym_id = jnp.pad(asym_id, [0, padding_length], constant_values=len(chain_names))
            entity_id = jnp.pad(entity_id, [0, padding_length], constant_values=len(chain_names))
            interface_asym_id = jnp.pad(interface_asym_id, [0, padding_length], constant_values=len(chain_names))
            seq_mask = jnp.pad(seq_mask, [0, padding_length])
        positions, mask, metrics = self._compiled_complex_prediction(model, padded_residue_count)(self.model_parameters[model], self.key, sequence, atoms, atom_mask, residue_index, asym_id, entity_id, interface_asym_id, seq_mask, flags, jnp.asarray(self.dropout), jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight), jnp.asarray(temperature), jnp.asarray(logit_scale))
        positions, mask = positions[:residue_count], mask[:residue_count]
        metrics = {name: trim_prediction_padding(value, residue_count) for name, value in metrics.items()}
        positions = align_prediction_to_target_template(positions, mask, atoms[:residue_count], atom_mask[:residue_count], flags[:residue_count])
        predicted_chain_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, atoms=positions, atom_mask=mask)
        predicted_complex = {name: protein_complex[name].replace(**{field: value[:true_length] for field, value in predicted_chain_arrays[name].items()}) for name, true_length in zip(chain_names, true_lengths)}
        if chain_lengths != true_lengths:
            metrics = trim_padded_metrics(metrics, real_residue_positions(chain_lengths, true_lengths))
        return StructurePrediction(protein_complex=predicted_complex, metrics=metrics)

    def _compiled_sequence_gradients(self, model: str, complex_shapes: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...], reference_shapes: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...], losses: dict[str, DesignLoss]) -> Callable:
        model_family = self.model_families[model]
        loss_signature = tuple((name, losses[name].function, losses[name].required_states) for name in sorted(losses))
        state_subbatch_sizes = tuple((state_name, resolve_subbatch_size(sum(chain_lengths), self.subbatch_size)) for state_name, _, chain_lengths in complex_shapes)
        cache_key = model_family, complex_shapes, reference_shapes, loss_signature, state_subbatch_sizes, self.multi_chain_binders, self.num_recycle, self.target_flexibility, self.bigbang_initialization
        compiled_gradient = self.gradient_compile_cache.get(cache_key)
        if compiled_gradient is None:
            state_alphafold_runners = {state_name: self._alphafold_runner(model_family, subbatch_size) for state_name, subbatch_size in state_subbatch_sizes}
            def predict_complex_arrays(alphafold_runner: af_model.RunModel, model_parameters: Array, key: Array, chain_names: tuple[str, ...], chain_lengths: tuple[int, ...], sequence: Array, atoms: Array, atom_mask: Array, flags: Array, residue_index: Array, dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array):
                asym_id = residue_chain_ids(chain_lengths)
                entity_id = residue_entity_ids(chain_names, chain_lengths, self.multi_chain_binders)
                seq_mask = real_residue_weights(flags)
                sequence_features, sequence_profile = prepare_design_sequence_features(sequence, flags, softmax_weight, one_hot_weight, temperature, logit_scale)
                model_inputs = alphafold_input_features(sequence_features, sequence_profile, atoms, atom_mask, residue_index, asym_id, seq_mask, flags, dropout, self.cyclic_offset_mode, entity_id, self.target_flexibility, self.bigbang_initialization)
                alphafold_outputs = recycled_alphafold_outputs(alphafold_runner, model_parameters, key, model_inputs, self.num_recycle)
                predicted_atom_positions = alphafold_outputs['structure_module']['final_atom_positions'].astype(jnp.float16)
                predicted_atom_mask = alphafold_outputs['structure_module']['final_atom_mask'].astype(bool)
                metrics = alphafold_prediction_metrics(alphafold_outputs, seq_mask, interface_asym_ids(chain_names, chain_lengths))
                predicted_atom_positions = align_prediction_to_target_template(predicted_atom_positions, predicted_atom_mask, atoms, atom_mask, flags)
                input_chain_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, sequence=sequence, atoms=atoms, atom_mask=atom_mask, flags=flags, residue_index=residue_index)
                predicted_chain_arrays = split_residue_arrays_by_chain(chain_names, chain_lengths, atoms=predicted_atom_positions, atom_mask=predicted_atom_mask)
                input_complex = {name: Protein(**input_chain_arrays[name]) for name in chain_names}
                protein_complex = {name: input_complex[name].replace(**predicted_chain_arrays[name]) for name in chain_names}
                return input_complex, protein_complex, predicted_atom_positions, predicted_atom_mask, metrics
            def sequence_design_loss(model_parameters: Array, key: Array, sequences: dict[str, Array], state_templates: dict[str, dict[str, Array]], reference_templates: dict[str, dict[str, Array]], weights: dict[str, Array], frozen_interfaces: dict[str, dict[str, Array]], dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array):
                protein_states: ProteinStates = {}
                predictions: StructurePredictions = {}
                prediction_arrays: dict[str, tuple[Array, Array, dict]] = {}
                for state_name, state_chain_names, chain_lengths in complex_shapes:
                    template = state_templates[state_name]
                    original_complex, predicted_complex, predicted_atom_positions, predicted_atom_mask, metrics = predict_complex_arrays(state_alphafold_runners[state_name], model_parameters, key, state_chain_names, chain_lengths, jnp.concatenate([sequences[chain_name] for chain_name in state_chain_names], axis=0), template['atoms'], template['atom_mask'], template['flags'], template['residue_index'], dropout, softmax_weight, one_hot_weight, temperature, logit_scale)
                    protein_states[state_name] = original_complex
                    predictions[state_name] = StructurePrediction(protein_complex=predicted_complex, metrics=dict(metrics))
                    prediction_arrays[state_name] = predicted_atom_positions, predicted_atom_mask, metrics
                loss_predictions = dict(predictions)
                for state_name, state_chain_names, chain_lengths in reference_shapes:
                    template = reference_templates[state_name]
                    chain_arrays = split_residue_arrays_by_chain(state_chain_names, chain_lengths, sequence=jnp.concatenate([sequences[name] for name in state_chain_names], axis=0), atoms=template['atoms'], atom_mask=template['atom_mask'], flags=template['flags'], residue_index=template['residue_index'])
                    loss_predictions[state_name] = StructurePrediction(protein_complex={name: Protein(**chain_arrays[name]) for name in state_chain_names}, metrics={})
                weighted_losses = {name: weights[name] * entry.function(protein_states, loss_predictions, **frozen_interfaces.get(name, {})) for name, entry in losses.items()}
                total_loss = sum(weighted_losses.values()) if weighted_losses else jnp.asarray(0.0)
                for name, entry in losses.items():
                    for state_name in entry.required_states or prediction_arrays.keys():
                        metrics = prediction_arrays[state_name][2]
                        if name in metrics:
                            raise ValueError(f'loss {name!r} would overwrite an existing metric on {state_name!r}')
                        metrics[name] = weighted_losses[name]
                return total_loss, prediction_arrays
            compiled_gradient = jax.jit(jax.value_and_grad(sequence_design_loss, argnums=2, has_aux=True))
            self.gradient_compile_cache.set(cache_key, compiled_gradient)
        return compiled_gradient

    def sequence_gradients(self, protein_states: ProteinStates, losses: dict[str, DesignLoss], model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=0.0, temperature: float=1.0, logit_scale: float=2.0, reference_predictions: StructurePredictions | None=None, compile_only: bool=False) -> tuple[StructurePredictions, dict[str, Array]]:
        model = self._resolve_model_name(model)
        _, original_shared_chains = collect_shared_chains(protein_states)
        state_names = canonical_state_names(protein_states)
        chain_names = canonical_chain_names(protein_states, state_names)
        padded_states = pad_design_chains(renamed_protein_states(protein_states, state_names, chain_names), self.length_bucket_size, self.target_pad_length)
        metric_source_names = {renamed_loss_name(name, state_names): name for name in losses}
        losses = renamed_state_losses(losses, state_names)
        state_source_names = {canonical: name for name, canonical in state_names.items()}
        chain_source_names = {canonical: name for name, canonical in chain_names.items()}
        complex_shapes = protein_state_shapes(padded_states)
        shared_chain_names, shared_chains = collect_shared_chains(padded_states)
        sequences = {name: shared_chains[name].sequence for name in shared_chain_names}
        state_templates = {state_name: concatenate_chain_arrays(state_chain_names, padded_states[state_name], 'atoms', 'atom_mask', 'flags', 'residue_index') for state_name, state_chain_names, _ in complex_shapes}
        reference_states = {state_names.get(state, state): {chain_names.get(name, name): protein.padded_to(len(shared_chains[chain_names.get(name, name)])) for name, protein in prediction.protein_complex.items() if chain_names.get(name, name) in shared_chains} for state, prediction in (reference_predictions or {}).items()}
        reference_states = {state: protein_complex for state, protein_complex in reference_states.items() if protein_complex}
        reference_shapes = protein_state_shapes(reference_states)
        reference_templates = {state_name: concatenate_chain_arrays(state_chain_names, reference_states[state_name], 'atoms', 'atom_mask', 'flags', 'residue_index') for state_name, state_chain_names, _ in reference_shapes}
        weights = {name: jnp.asarray(entry.weight, dtype=jnp.float32) for name, entry in losses.items()}
        frozen_interfaces = frozen_interface_arguments(losses, shared_chains)
        compiled_sequence_gradients = self._compiled_sequence_gradients(model, complex_shapes, reference_shapes, losses)
        gradient_arguments = (self.model_parameters[model], self.key, sequences, state_templates, reference_templates, weights, frozen_interfaces, jnp.asarray(self.dropout), jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight), jnp.asarray(temperature), jnp.asarray(logit_scale))
        with one_worker_compiles((model, complex_shapes, reference_shapes, tuple(sorted(losses)))):
            compiled_sequence_gradients.lower(*gradient_arguments).compile()
        if compile_only:
            return {}, {}, jnp.asarray(0.0)
        (design_loss, prediction_arrays), shared_chain_gradients = compiled_sequence_gradients(*gradient_arguments)
        predictions: StructurePredictions = {}
        for canonical_state, canonical_chains, chain_lengths in complex_shapes:
            state_name = state_source_names[canonical_state]
            source_chains = tuple(chain_source_names[name] for name in canonical_chains)
            positions, mask, metrics = prediction_arrays[canonical_state]
            predicted_chain_arrays = split_residue_arrays_by_chain(canonical_chains, chain_lengths, atoms=positions, atom_mask=mask)
            true_lengths = tuple(len(protein_states[state_name][name]) for name in source_chains)
            predicted_complex = {name: protein_states[state_name][name].replace(**{field: value[:true_length] for field, value in predicted_chain_arrays[canonical_chain].items()}) for name, canonical_chain, true_length in zip(source_chains, canonical_chains, true_lengths)}
            metrics = {metric_source_names.get(name, name): value for name, value in metrics.items()}
            predictions[state_name] = StructurePrediction(protein_complex=predicted_complex, metrics=trim_padded_metrics(metrics, real_residue_positions(chain_lengths, true_lengths)) if chain_lengths != true_lengths else metrics)
        sequence_gradients = {chain_source_names[name]: shared_chain_gradients[name][:len(original_shared_chains[chain_source_names[name]])] for name in shared_chain_names}
        return predictions, sequence_gradients, design_loss
MULTIMER_POOL: tuple[str, ...] = tuple(f'model_{i}_multimer_v3' for i in range(1, 6))
MONOMER_POOL: tuple[str, ...] = ('model_1_ptm', 'model_2_ptm')
