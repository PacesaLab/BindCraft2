import hashlib
import numpy as np
from bindcraft.protein import ATOM_INDEX, Protein
from bindcraft.settings import DESIGN_STAGE_NAMES

EXCLUDED_SETTING_NAMES = (
    'binder_name', 'campaign_name', 'project_folder', 'binder_chain', 'target_chain', 'hash_design_names',
    'binder_scaffold',
    'parameter_sweep',
    'save_', 'sparse_output', 'archive_trajectories', 'relax_',
    'filters', 'multitarget_cumulative_filter', 'multitarget_filter_models',
    'mpnn_', 'induced_fit_mpnn_', 'multitarget_tied_redesign', 'redesign_interface', 'sequence_candidates',
    'kept_sequences', 'enough_passing_sequences', 'number_of_final_designs', 'trajectory_only', 'validation_',
    '*autotune',
    'attention_backend', 'subbatch_size', 'use_cueq', 'gpu_ids', 'auto_multi_gpu', 'design_workers',
    'workers_per_gpu', 'max_workers_per_gpu', 'worker_launch_stagger', 'max_trajectories', 'resume')
EXCLUDED_TARGET_SETTING_NAMES = ('target_path',)

def is_excluded_setting(name: str, excluded: tuple[str, ...]=EXCLUDED_SETTING_NAMES) -> bool:
    if name.startswith(('min_', 'max_')) and name.rsplit('_', 1)[-1] in DESIGN_STAGE_NAMES:
        return True
    return any(pattern[1:] in name if pattern.startswith('*') else name.startswith(pattern) if pattern.endswith('_') else name == pattern for pattern in excluded)

def hashed_target_requests(targets: list) -> list:
    return [{name: value for name, value in target.items() if name not in EXCLUDED_TARGET_SETTING_NAMES} if isinstance(target, dict) else target for target in targets]

def design_setting_values(settings: dict, excluded: tuple[str, ...]=EXCLUDED_SETTING_NAMES) -> dict:
    return {name: hashed_target_requests(value) if name == 'targets' and isinstance(value, list) else value for name, value in settings.items() if not is_excluded_setting(name, excluded)}

def canonical_text(value) -> str:
    if isinstance(value, bool):
        return 'b:1' if value else 'b:0'
    if isinstance(value, int):
        return f'i:{value}'
    if isinstance(value, float):
        return 'f:' + format(value, '.8g')
    if value is None:
        return 'n:'
    if isinstance(value, str):
        return f's:{value}'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(canonical_text(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{' + ','.join(f'{name}={canonical_text(value[name])}' for name in sorted(value)) + '}'
    raise TypeError(f'a setting of type {type(value).__name__} cannot be hashed; write it as a number, string, list or dict')

def hash_values(values: dict, length: int=16) -> str:
    return hashlib.sha256(canonical_text(values).encode()).hexdigest()[:length]

def target_conformation_fingerprint(target: Protein, precision: int=1, length: int=16) -> str:
    resolved_alpha_carbons = np.asarray(target.atom_mask)[:, ATOM_INDEX['CA']]
    coordinates = np.asarray(target.atoms, dtype=np.float32)[:, ATOM_INDEX['CA']][resolved_alpha_carbons]
    if len(coordinates) < 2:
        return hashlib.sha256(np.asarray(target.sequence, dtype=np.float32).argmax(-1).astype(np.int16).tobytes()).hexdigest()[:length]
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    return hashlib.sha256(np.sort(np.round(distances[np.triu_indices(len(coordinates), k=1)], precision)).astype(np.float32).tobytes()).hexdigest()[:length]

def design_hash(settings: dict, drawn: dict | None=None, targets: dict[str, Protein] | None=None, length: int=16) -> tuple[str, dict]:
    basis = {**design_setting_values(settings), **(drawn or {}), **{f'conformation.{name}': target_conformation_fingerprint(target) for name, target in (targets or {}).items()}}
    return hash_values(basis, length=length), basis

def design_name(label: str, binder_length: int, identity: str, hashed: bool=True, trajectory_number: int=0) -> str:
    return f'{label}_l{int(binder_length)}_{identity if hashed else trajectory_number}'
