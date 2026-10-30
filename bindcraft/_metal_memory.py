"""Conservative unified-memory estimates; no models or backends are loaded."""

from bindcraft._worker_memory import GIB, MIB, pinned_settings

MEMORY_MODEL_VERSION = 4
CACHE_SIZE = 16
MARGIN_PERCENT = 25
# Standard AF2 checkpoints contain at most 356 MiB of FP32 arrays; MPNN's
# 128-channel checkpoints contain 6.4 MiB. Upload/fusion temporaries belong
# to peak workspace, not a second permanently resident copy of every model.
AF_MODEL_BYTES = 384 * MIB
MPNN_MODEL_BYTES = 16 * MIB
RUNTIME_BYTES = GIB
# Compiled code/graph storage is per callable, not a retained activation arena.
# AF graphs grow with unrolled recycles; MPNN uses a scan for its decoder.
AF_GRAPH_BYTES_PER_RECYCLE = 128 * MIB
MPNN_GRAPH_BYTES = 64 * MIB


def execution_identity(settings: dict) -> dict:
    """Keep a worker generation on the same runtime and memory policy."""
    from bindcraft._metal_runtime import get_runtime_identity, runtime_identity_key
    runtime = get_runtime_identity()
    return {'schema_version': 2, 'runtime_identity': runtime,
            'runtime_key': runtime_identity_key(runtime), 'memory_model_version': MEMORY_MODEL_VERSION}


def _integer(value, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}')
    return value


def _model_count(value, default: int) -> int:
    if value is None:
        return default
    return _integer(len(value) if isinstance(value, (list, tuple)) else value, 'model count')


def _fasta_target(target: dict) -> bool:
    source = str(target.get('target_path', '')).strip()
    return source.lower().endswith(('.fa', '.fasta', '.faa')) or source.startswith('>')


def estimate_worker_memory(settings: dict, residue_count: int, *, shape_count: int = 1,
                           execution_profile: dict | None = None) -> dict:
    """Size a whole worker from the upstream padded-complex upper bound.

    Count resident parameters, bounded compiled graphs and live result arrays
    once, then add the largest serial phase workspace and a 25% margin. Graph
    storage scales with cache entries/recycles; activation and result buffers
    scale with residue count. Stock attention also gets a cubic allowance.
    Smaller chunks receive no speculative memory discount.
    These are conservative planning allowances, not measured per-target fits.
    Host reserve and coordinator overhead are charged separately by the pool.
    """
    result = {'schema_version': 2, 'status': 'unknown', 'reason': None, 'total_bytes': None,
              'accounting_kind': 'conservative_estimate', 'execution_identity': None}
    try:
        active = pinned_settings(settings, execution_profile) if execution_profile is not None else settings
        if not isinstance(active, dict):
            raise ValueError('settings must be a mapping')
        _integer(residue_count, 'padded residue count')
        _integer(shape_count, 'shape count')
        from bindcraft._metal_policy import resolve_policy
        policy = resolve_policy(residue_count, subbatch_size=active.get('subbatch_size', 'auto'),
                                attention_backend=active.get('attention_backend', 'auto'),
                                use_remat=active.get('use_remat', True))
        redesign = not active.get('trajectory_only')
        design_recycles = _integer(active.get('design_recycles', 1), 'design_recycles', 0)
        beta_recycles = _integer(active.get('betasheet_reopt_recycles', design_recycles), 'betasheet_reopt_recycles', 0)
        validation_recycles = _integer(active.get('validation_recycles', 3), 'validation_recycles', 0)
        candidates = _integer(active.get('sequence_candidates', 10), 'sequence_candidates') if redesign else 0
        multimer = active.get('validation_model') == 'multimer' or (
            active.get('validation_model') is None and (active.get('binder_scaffold') or active.get('copies', 1) > 1))
        # Match the held-out split: implicit multimer design uses the models
        # left after validation, not another independent five-model pool.
        held_out = _model_count(active.get('validation_models', 2), 0)
        design_models = _integer(_model_count(active.get('design_models'), 5 - held_out if multimer else 5),
                                 'design model count')
        validation_models = _model_count(active.get('validation_models'),
                                        (5 - design_models if multimer else 2) or 2) if redesign else 0
        resident = design_models + validation_models
        targets = active.get('targets') or []
        target_names = {'complex', *(str(target.get('name')) for target in targets if target.get('name'))}
        crops = max(1, _integer(active.get('idr_crop_count', 1), 'idr_crop_count', 0))
        states = max(1, sum(crops if _fasta_target(target) else 1 for target in targets))
        extra_states = set()
        for name, entry in (active.get('losses') or {}).items():
            if isinstance(entry, dict):
                for source in (entry, entry.get('params') or {}):
                    for field in ('prediction_state', 'reference_state'):
                        value = source.get(field)
                        if value and str(value) not in target_names:
                            extra_states.add(str(value))
            if name.split('.')[0] in ('induced_fit_interface', 'fold_switching'):
                extra_states.add('binder_alone')
        if active.get('weights_induced_fit_interface') or active.get('weights_fold_switching'):
            extra_states.add('binder_alone')
        states += len(extra_states) + len(active.get('binder_shapes') or ())
        # Variable scaffold edits and FASTA crops can trace many lengths even
        # inside one binder bucket. Charge their full bounded function caches.
        shapes = CACHE_SIZE if active.get('binder_scaffold') or any(_fasta_target(target) for target in targets) else shape_count
        variants = len({design_recycles, beta_recycles})
        desperate = execution_profile is None and redesign and active.get('desperation', True)
        if desperate:
            # Unmanaged callers can retain original/current/incoming validators.
            resident += 10
            variants = 2 * len({design_recycles, beta_recycles, 3})
            design_recycles = max(design_recycles, 3)
        design_entries = 2 * min(CACHE_SIZE, shapes * states * variants)
        validation_entries = min(CACHE_SIZE, 2 * shapes * states) * (2 if desperate else 1) if redesign else 0
        # Redraws can shrink the candidate batch, retaining a JIT specialization
        # for each batch size even when the chain-shape callable is reused.
        mpnn_entries = min(CACHE_SIZE, shapes * states) * candidates
        square = residue_count ** 2
        scaled = lambda amount: (amount * square + 256 ** 2 - 1) // 256 ** 2
        persistent = RUNTIME_BYTES + resident * AF_MODEL_BYTES + (MPNN_MODEL_BYTES if redesign else 0)
        retained = AF_GRAPH_BYTES_PER_RECYCLE * (
            design_entries * (max(design_recycles, beta_recycles) + 1) +
            validation_entries * (validation_recycles + 1)) + mpnn_entries * MPNN_GRAPH_BYTES
        # Earlier recycles are stop_gradient calls and execute serially. They
        # enlarge the unrolled graph above, not the live gradient activation set.
        workspace = (8 * GIB + scaled(6 * GIB)) * states
        if not policy['rematerialization']:
            workspace *= 4
        if policy['attention_backend'] == 'stock':
            workspace *= max(1, (residue_count + 255) // 256)
        global_chunk = min(residue_count, policy['global_subbatch_size'] or residue_count)
        workspace *= max(1, (global_chunk + 383) // 384)
        workspace += scaled(GIB // 2) * (bool(active.get('target_flexibility')) +
                                         bool(active.get('hard_target') or active.get('large_complex_initialization')))
        cold = scaled(4 * GIB)
        overlap = cold if active.get('compile_next_length') else 0
        # FP32 64-bin distogram + PAE, with 1 KiB/residue for coordinates,
        # sequence, masks and confidence tracks. Accepted ensembles drop the
        # distogram. Keep the design, in-flight model outputs, and both old
        # and incoming candidate summaries across the campaign-loop assignment.
        full_result = 65 * 4 * square + 1024 * residue_count
        summary_result = 4 * square + 1024 * residue_count
        downstream = states * full_result
        mpnn_workspace = relax_workspace = 0
        if redesign:
            downstream += (states + 1) * ((validation_models + 1) * full_result +
                                         (2 * candidates + 2) * summary_result)
            # Batched MPNN has 128 channels and 48 neighbors. Allow eight live
            # edge tensors and eight pair matrices per candidate. Tied decoding
            # concatenates target states into one longer complex.
            mpnn_length = residue_count * (states if active.get('multitarget_tied_redesign', True) else 1)
            mpnn_workspace = candidates * (8 * 128 * 4 * mpnn_length * min(48, mpnn_length) +
                                           8 * 4 * mpnn_length ** 2)
            if active.get('relax_accepted_designs'):
                # Relaxation uses all-atom pair matrices, after AF/MPNN finish.
                relax_workspace = 16 * 4 * (37 * residue_count) ** 2
        # First-use compilation finishes before execution; Metal campaigns also
        # disable background next-length compilation. Count their peak rather
        # than summing both. Models, cached executables and downstream results
        # can stay live across stages, so keep those allowances additive.
        transient = max(workspace, cold, mpnn_workspace, relax_workspace) + overlap
        subtotal = persistent + retained + transient + downstream
        margin = (subtotal * MARGIN_PERCENT + 99) // 100
        total = (subtotal + margin + MIB - 1) // MIB * MIB
        if total >= 1 << 63:
            raise ValueError('Estimated worker memory exceeds the signed-64-bit byte range')
        return {**result, 'status': 'known', 'total_bytes': total,
                'residue_count': residue_count, 'shape_count': shape_count,
                'persistent_bytes': persistent, 'retained_bytes': retained,
                'peak_workspace_bytes': workspace, 'cold_bytes': cold, 'overlap_bytes': overlap,
                'transient_bytes': transient,
                'mpnn_workspace_bytes': mpnn_workspace, 'relax_workspace_bytes': relax_workspace,
                'downstream_bytes': downstream, 'margin_bytes': margin, 'margin_percent': MARGIN_PERCENT,
                'execution_identity': execution_identity(active)}
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        return {**result, 'reason': str(error)}
