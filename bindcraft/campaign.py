import os
import threading
import time
import jax
from bindcraft.af2 import AlphaFoldDesignModel, MONOMER_POOL, MULTIMER_POOL, campaign_length_bucket, padded_prediction_length
from bindcraft.campaign_output import trajectory_output_path, CampaignProgress, DEFAULT_PROJECT_FOLDER, RANKING_METRIC, RANK_STAGE, REFOLD_STAGE, TRAJECTORY_STAGE, accepted_state_suffixes, append_accepted_design, append_campaign_metrics, archive_trajectory_folder, designed_span_stamp, discard_trajectory_structures, drawn_weight_stamp, model_score_stamp, rank_accepted_designs, reprediction_facts, structure_metadata, stage_folder, stage_table, target_ordered_row, timing_stamp, weighted_target_order, write_campaign_metadata, write_campaign_summary
from bindcraft.campaign_log import binder_optimization, campaign_budget_exhausted, campaign_closed, campaign_header, campaign_label, design_worker_index, speaks_for_the_campaign, trajectory_already_designed, trajectory_design_label, trajectory_header
from bindcraft.design_identity import design_hash, design_name
from bindcraft.parameter_sweep import arm_trajectory_budget, autotuned_settings, autotuned_stamp, parameter_sweep_arms, parameter_sweep_options, sweep_block_budgets, write_sweep_record
from bindcraft.MPNN_stage import redesign_and_validate_binders
from bindcraft.filters import INTERFACE_PDAE_METRICS, all_atom_clashes_metric, backbone_clashes_metric, binder_chain_sequences, design_sequence_report, design_stage_filters, evaluate_design_filters, residue_confidence_tracks
from bindcraft.design_workers import campaign_subbatch_size, dispatch_design_workers, running_as_design_worker
from bindcraft.desperation import desperate_settings
from bindcraft.loss import DISTOGRAM_DEPENDENT_LOSSES
from bindcraft.model_weights import model_weights
from bindcraft.preflight import cleaned_campaign_settings, preflight_campaign
from bindcraft.protein import BINDER_ALONE, StructurePrediction, relax_protein_complex, superposed_on_binder, write_structure
from bindcraft.proteinmpnn import ProteinMPNNSequenceModel
from bindcraft.settings import DEFAULT_SETTINGS, PRESET_TIERS, build_design_settings, design_seeds_from_given_coordinates, parse_setting_overrides, read_campaign_metadata, read_settings, requested_core_profiles, requested_preset_names, resolve_cyclic_offset_mode, resolve_validation_model, select_design_and_validation_models, target_state_names, validation_seeds_from_given_coordinates
from bindcraft.protein_preparation import receptor_chain_layouts, design_residue_count, frame_holding_target, initialize_design_trajectory, prepare_targets, states_holding_the_frame, target_frame_report, sampled_trajectory_values, validation_target_states
from bindcraft.trajectory import primary_target_state, run_trajectory
from bindcraft.trajectory_output import copy_trajectory_animation

DEFAULT_TRAJECTORY_ONLY_BUDGET = 100

def campaign_trajectory_budget(settings: dict) -> int | None:
    max_trajectories = settings.get('max_trajectories') or (DEFAULT_TRAJECTORY_ONLY_BUDGET if settings.get('trajectory_only') else None)
    arms = parameter_sweep_arms(settings)
    return arm_trajectory_budget(max_trajectories, len(arms)) if arms else max_trajectories

def scored_trajectory_metrics(design_settings, protein_states, predictions) -> dict[str, float]:
    #score the hallucinated pose on the same battery the refold stage runs, so the trajectory table is comparable rather than a dump of the optimiser's own loss terms
    if not predictions:
        return {}
    binder_chain = design_settings.designed_binder_chain
    target_prefix = design_settings.target_chain_prefix
    primary = next((name for name in target_state_names(design_settings) if name in predictions), primary_target_state(protein_states))
    stage_filters = design_stage_filters(design_settings, protein_states, 'final', campaign_filters=design_settings.filters)
    _result, metrics = evaluate_design_filters(stage_filters, protein_states, predictions)
    for name, interface_pdae in INTERFACE_PDAE_METRICS.items():
        value = interface_pdae(protein_states, predictions, prediction_state=primary, binder=binder_chain, target=f'{target_prefix}_{primary}')
        if value is not None:
            metrics[name] = value
    metrics['Binder_Sequence'] = binder_chain_sequences(predictions[primary].protein_complex, binder_chain)
    return metrics

def settings_provenance(settings: dict, setting_overrides: tuple[str, ...] | list[str]=()) -> dict[str, str]:
    #which preset file each tier drew from, and any overrides passed on top, recorded on every row of every table
    tiers = {'settings_core': requested_core_profiles(settings) or ('default',)}
    for tier in PRESET_TIERS:
        tiers[f'settings_{tier}'] = requested_preset_names(settings, tier)
    provenance = {name: ';'.join(names) for name, names in tiers.items() if names}
    if setting_overrides:
        provenance['settings_overrides'] = ' '.join(str(override) for override in setting_overrides)
    return provenance

def binder_protomer_length(protein_complex: dict, binder: str) -> int:
    return len(binder_chain_sequences(protein_complex, binder).split('/')[0])

def refuse_predictor_without_distogram(settings: dict, design_model) -> None:
    if getattr(design_model, 'provides_distogram', True):
        return
    needed = sorted(name for name in (settings.get('losses') or {}) if name.split('.')[0] in DISTOGRAM_DEPENDENT_LOSSES)
    if needed:
        raise ValueError(f"{type(design_model).__name__} returns no distogram, and these losses read one: {', '.join(needed)}. Use a predictor that provides a distogram, or drop those losses.")

def apply_desperation_settings(design_model, validation_model, settings: dict) -> None:
    design_model.num_recycle = int(settings.get('design_recycles', DEFAULT_SETTINGS['design_recycles']))
    for prediction_model in (design_model, validation_model):
        if prediction_model is not None:
            prediction_model.target_flexibility = float(settings.get('target_flexibility', DEFAULT_SETTINGS['target_flexibility']))
    if design_model is not None:
        design_model.bigbang_initialization = design_seeds_from_given_coordinates(settings)
    if validation_model is not None:
        validation_model.bigbang_initialization = validation_seeds_from_given_coordinates(settings)

def desperate_prediction_pools(design_model, validation_model, settings: dict, build_validation_model) -> tuple:
    selected_models = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)
    design_model.models = tuple(model for model in selected_models.design_models if model in design_model.model_families) or design_model.models
    if validation_model is None or all(model in validation_model.model_families for model in selected_models.validation_models):
        return validation_model, selected_models.validation_models
    print(f"desperation: validation moves to the {resolve_validation_model(settings)} pool {selected_models.validation_models}, held out of the {len(design_model.models)} models design was narrowed to {design_model.models}", flush=True)
    return build_validation_model(selected_models.validation_models), selected_models.validation_models

def relaxed_accepted_design(validated_binder, primary_target: str, design_name: str, settings: dict) -> dict | None:
    try:
        relaxed_complex = relax_protein_complex(validated_binder.predictions[primary_target].protein_complex, {name[len('relax_'):]: value for name, value in settings.items() if name.startswith('relax_') and name != 'relax_accepted_designs'})
    except Exception as relax_failure:
        print(f'{design_name}: relaxation failed, keeping only the design as predicted ({relax_failure})', flush=True)
        return None
    relaxed_predictions = {primary_target: StructurePrediction(protein_complex=relaxed_complex, metrics=validated_binder.predictions[primary_target].metrics)}
    for check, clash_metric in (('Backbone_Clashes', backbone_clashes_metric), ('All_Atom_Clashes', all_atom_clashes_metric)):
        if check in validated_binder.metrics:
            validated_binder.metrics[f'Relaxed_{check}'] = clash_metric({}, relaxed_predictions, prediction_state=primary_target)
    validated_binder.metrics['Relaxed'] = 1.0
    return relaxed_complex

def trajectory_length_bucket(design_settings, key: jax.Array, trajectory_number: int) -> int:
    drawn, _targets = sampled_trajectory_values(design_settings, jax.random.fold_in(key, trajectory_number))
    return padded_prediction_length(drawn['binder_length'], campaign_length_bucket(design_settings.settings))

def compile_next_length_bucket(design_settings, alphafold_model, key: jax.Array, trajectory_number: int) -> threading.Thread:
    def compile_gradient_graph() -> None:
        try:
            protein_states, _multi_chain_binders, losses = initialize_design_trajectory(design_settings, jax.random.split(jax.random.fold_in(key, trajectory_number))[0])
            alphafold_model.sequence_gradients(protein_states, losses, model=alphafold_model.models[0], compile_only=True)
        except Exception as compile_failure:
            print(f'compiling the next length bucket failed, leaving it to the trajectory that folds there ({compile_failure})', flush=True)
    compile_thread = threading.Thread(target=compile_gradient_graph, daemon=True)
    compile_thread.start()
    return compile_thread

def write_trajectory_design(design_settings, protein_states, predictions, trajectory_directory: str, receptor_chains: dict) -> None:
    """The fold the hallucination stage ended on, before ProteinMPNN redraws its sequence and before the monomer checks.

    save_design_frames keeps every sequence update, which is thousands of files nobody reads; this keeps
    the one structure the gradient actually built, so what redesign made of the binder can be read
    against what was handed to it. A multitarget trajectory writes the pose it reached on each target,
    and an induced-fit or fold-switching one writes the unbound fold beside them, since the pair is the
    switch."""
    primary_target = next((name for name in target_state_names(design_settings) if name in predictions), primary_target_state(protein_states))
    primary_complex = predictions[primary_target].protein_complex
    framed_states = states_holding_the_frame({name: state_complex for name, state_complex in protein_states.items() if name != BINDER_ALONE}, frame_holding_target(design_settings), design_settings.target_chain_prefix)
    state_suffixes = accepted_state_suffixes(design_settings.prepared_states, predictions)
    for state, prediction in predictions.items():
        suffix = '_monomer' if state == BINDER_ALONE else state_suffixes.get(state, '')
        #every state but the one holding the target frame is read against the binder it shares
        on_its_own_frame = state == primary_target or state in framed_states
        written_complex = prediction.protein_complex if on_its_own_frame else superposed_on_binder(prediction.protein_complex, primary_complex)
        write_structure(written_complex, trajectory_output_path(trajectory_directory, f'trajectory{suffix}.cif'), plddt=prediction.metrics.get('plddt'),
                        metadata=structure_metadata(campaign_label(design_settings.settings), prediction.metrics, design_settings, state=state, stage='trajectory'),
                        residue_metrics=residue_confidence_tracks(predictions, state), receptor_chains=receptor_chains)

def run_campaign_arm(settings: dict, project_folder: str, alphafold_model, validation_model, mpnn_model, key: jax.Array, max_trajectories: int | None, build_validation_model, autotune: bool=True, closing: bool=True, metadata: dict[str, str] | None=None) -> int:
    requested_designs = int(settings.get('number_of_final_designs', 1))
    design_settings = build_design_settings(settings)
    target_chain_prefix = design_settings.target_chain_prefix
    receptor_chains = receptor_chain_layouts(design_settings)
    trajectories_dir = stage_folder(project_folder, TRAJECTORY_STAGE)
    accepted_dir = stage_folder(project_folder, RANK_STAGE)
    relaxed_dir = os.path.join(accepted_dir, 'relaxed')
    trajectories_csv = stage_table(project_folder, TRAJECTORY_STAGE)
    candidates_csv = stage_table(project_folder, REFOLD_STAGE)
    campaign_progress = CampaignProgress(project_folder, requested_designs, max_trajectories)
    compiled_length_buckets: set[int] = set()
    compiling_next_length: threading.Thread | None = None
    #read once, before any trajectory narrows the settings down to the one sequence it redesigns
    campaign_binder_sequences = design_settings.binder.sequences
    worker_label = f" | worker {os.environ['BINDCRAFT_WORKER_ID']}" if 'BINDCRAFT_WORKER_ID' in os.environ else ''
    if speaks_for_the_campaign() and len(design_settings.prepared_states) > 1:
        print(target_frame_report(prepare_targets(design_settings), frame_holding_target(design_settings)), flush=True)
    if mpnn_model is None and speaks_for_the_campaign():
        print(f'trajectory-only: no ProteinMPNN redesign, no design will be accepted; stopping at max_trajectories={max_trajectories}', flush=True)
    while True:
        claimed = campaign_progress.claim_trajectory()
        if claimed is None:
            break
        trajectory_number, accepted_design_count = claimed
        tuned_settings = autotuned_settings(campaign_progress, settings, project_folder) if autotune else settings
        tuned_settings, desperation = desperate_settings(tuned_settings, project_folder)
        if desperation:
            print(desperation, flush=True)
        #one trajectory per sequence that was given, in name order, so every one is folded and redesigned exactly once
        if campaign_binder_sequences:
            parent_name = sorted(campaign_binder_sequences)[(trajectory_number - 1) % len(campaign_binder_sequences)]
            tuned_settings = {**tuned_settings, 'binder_sequences': {parent_name: campaign_binder_sequences[parent_name]}}
        design_settings = build_design_settings(tuned_settings) if tuned_settings != design_settings.settings else design_settings
        trajectory_random_key = jax.random.fold_in(key, trajectory_number)
        for prediction_model in (alphafold_model, validation_model, mpnn_model):
            if prediction_model is not None:
                prediction_model.key = trajectory_random_key
        validation_model, validation_models = desperate_prediction_pools(alphafold_model, validation_model, tuned_settings, build_validation_model)
        apply_desperation_settings(alphafold_model, validation_model, tuned_settings)
        autotuned = autotuned_stamp(settings, tuned_settings)
        drawn, trajectory_targets = sampled_trajectory_values(design_settings, trajectory_random_key)
        identity, _hashed_values = design_hash(tuned_settings, {**drawn, 'design_models': list(alphafold_model.models)}, trajectory_targets)
        name = design_name(trajectory_design_label(settings, tuple(state.objective for state in design_settings.prepared_states)), drawn['binder_length'], identity, settings.get('hash_design_names', True), trajectory_number)
        print(trajectory_header(trajectory_number, name, accepted_design_count, requested_designs, worker_label, autotuned), flush=True)
        if not campaign_progress.claim_recipe(identity):
            print(trajectory_already_designed(name), flush=True)
            continue
        trajectory_directory = os.path.join(trajectories_dir, name)
        trajectory_length = padded_prediction_length(drawn['binder_length'], campaign_length_bucket(settings))
        compiled_fresh = trajectory_length not in compiled_length_buckets
        compiled_length_buckets.add(trajectory_length)
        next_length_bucket = trajectory_length_bucket(design_settings, key, trajectory_number + 1) if tuned_settings.get('compile_next_length', True) else None
        if next_length_bucket is not None and next_length_bucket not in compiled_length_buckets and (compiling_next_length is None or not compiling_next_length.is_alive()):
            compiling_next_length = compile_next_length_bucket(design_settings, alphafold_model, key, trajectory_number + 1)
            compiled_length_buckets.add(next_length_bucket)
        design_started = time.time()
        protein_states, predictions, failed_stage = run_trajectory(design_settings, alphafold_model, trajectory_random_key, trajectory_directory=trajectory_directory, targets=trajectory_targets)
        target_order = weighted_target_order(design_settings.prepared_states)
        metrics = scored_trajectory_metrics(design_settings, protein_states, predictions) if failed_stage is None else {}
        if settings.get('save_design_trajectory') and predictions:
            write_trajectory_design(design_settings, protein_states, predictions, trajectory_directory, receptor_chains)
        append_campaign_metrics(trajectories_csv, target_ordered_row({'design': name, 'trajectory': trajectory_number, 'length': drawn['binder_length'], **({'binder_parent': drawn['binder_parent']} if 'binder_parent' in drawn else {}), 'hash': identity, 'terminated': failed_stage or '', 'autotuned': autotuned, **drawn_weight_stamp(drawn), **metrics, 'Timing': timing_stamp(worker=os.environ.get('BINDCRAFT_WORKER_ID', '0'), start=design_started, design=time.time() - design_started, compiled=int(compiled_fresh)), **(metadata or {})}, target_order))
        campaign_progress.record_trajectory_outcome(failed_stage)
        accepted_from_trajectory = 0
        if failed_stage is None and mpnn_model is not None:
            print(binder_optimization(settings.get('sequence_candidates', DEFAULT_SETTINGS['sequence_candidates'])), flush=True)
            primary_target = next((name for name in target_state_names(design_settings) if name in predictions), primary_target_state(protein_states))
            binder_chains = design_settings.binder_chains
            binder_chain = design_settings.designed_binder_chain
            binder_alone_complex = predictions[BINDER_ALONE].protein_complex if BINDER_ALONE in predictions else None
            framed_states = states_holding_the_frame({name: complex for name, complex in protein_states.items() if name != BINDER_ALONE}, frame_holding_target(design_settings), target_chain_prefix)
            validated_binders = redesign_and_validate_binders(predictions[primary_target].protein_complex, validation_model, mpnn_model, design_settings, validation_models, binder=binder_chain, target=f'{target_chain_prefix}_{primary_target}', prediction_state=primary_target, target_states=validation_target_states(design_settings, {name: protein_complex for name, protein_complex in protein_states.items() if name != BINDER_ALONE}, drawn['trajectory_seed']), predicted_states={name: prediction.protein_complex for name, prediction in predictions.items() if name != BINDER_ALONE}, multi_chain_binder=binder_chains if design_settings.binder.copies > 1 and design_settings.oligomer_tie == 'symmetric' else (), binder_alone_complex=binder_alone_complex, design_pae=predictions[primary_target].metrics.get('pae'), trajectory_seed=drawn['trajectory_seed'], design=name, design_hash=identity, candidates_csv=candidates_csv, project_folder=project_folder)
            for candidate_index, validated_binder in enumerate(validated_binders):
                accepted_design_count, accepted_from_trajectory = campaign_progress.record_accepted_design(), accepted_from_trajectory + 1
                candidate_name = f'{name}_seq{candidate_index}'
                relaxed_complex = relaxed_accepted_design(validated_binder, primary_target, candidate_name, settings) if settings.get('relax_accepted_designs') else None
                primary_complex = validated_binder.predictions[primary_target].protein_complex
                candidate_metadata = structure_metadata(campaign_label(settings), {**validated_binder.metrics, **model_score_stamp(validated_binder.model_metrics)}, design_settings, **{**designed_span_stamp(primary_complex, receptor_chains), **(metadata or {}), 'design': candidate_name, 'design_hash': identity, 'validation_models': ' '.join(validation_models), **reprediction_facts(mpnn_model, validation_model), **drawn_weight_stamp(drawn)})
                state_suffixes = accepted_state_suffixes(design_settings.prepared_states, validated_binder.predictions)
                write_structure(primary_complex, os.path.join(accepted_dir, f'{candidate_name}{state_suffixes[primary_target]}.cif'), plddt=validated_binder.predictions[primary_target].metrics.get('plddt'), metadata=candidate_metadata, residue_metrics=residue_confidence_tracks(validated_binder.predictions, primary_target, binder_chain, f'{target_chain_prefix}_{primary_target}'), receptor_chains=receptor_chains)
                #for multitargeting and detargeting
                for target_state in validated_binder.predictions:
                    if target_state in (primary_target, BINDER_ALONE):
                        continue
                    state_complex = validated_binder.predictions[target_state].protein_complex
                    write_structure(state_complex if target_state in framed_states else superposed_on_binder(state_complex, primary_complex), os.path.join(accepted_dir, f'{candidate_name}{state_suffixes[target_state]}.cif'), plddt=validated_binder.predictions[target_state].metrics.get('plddt'), metadata=candidate_metadata, residue_metrics=residue_confidence_tracks(validated_binder.predictions, target_state, binder_chain, f'{target_chain_prefix}_{target_state}'), receptor_chains=receptor_chains)
                #conformational switching case
                if binder_alone_complex is not None and settings.get('save_binder_monomers', True):
                    write_structure(superposed_on_binder(validated_binder.predictions[BINDER_ALONE].protein_complex, primary_complex), os.path.join(accepted_dir, f'{candidate_name}_monomer.cif'), plddt=validated_binder.predictions[BINDER_ALONE].metrics.get('plddt'), metadata=candidate_metadata, residue_metrics=residue_confidence_tracks(validated_binder.predictions, BINDER_ALONE, binder_chain), receptor_chains=receptor_chains)
                #relaxation case
                if relaxed_complex is not None:
                    write_structure(relaxed_complex, os.path.join(relaxed_dir, f'{candidate_name}{state_suffixes[primary_target]}.cif'), plddt=validated_binder.predictions[primary_target].metrics.get('plddt'), metadata=candidate_metadata, residue_metrics=residue_confidence_tracks(validated_binder.predictions, primary_target, binder_chain, f'{target_chain_prefix}_{primary_target}'), receptor_chains=receptor_chains)
                copy_trajectory_animation(trajectory_directory, os.path.join(accepted_dir, f'{candidate_name}.html'))
                append_accepted_design(project_folder, target_ordered_row({'design': candidate_name, 'length': binder_protomer_length(primary_complex, binder_chain), 'hash': identity, **validated_binder.metrics, **design_sequence_report(validated_binder.predictions, prediction_state=primary_target, binder=binder_chain, target=f'{target_chain_prefix}_{primary_target}', receptor_chains=receptor_chains), **(metadata or {})}, weighted_target_order(design_settings.prepared_states)))
                if accepted_design_count >= requested_designs:
                    break
        if not accepted_from_trajectory and not settings.get('save_failed_trajectories', True):
            discard_trajectory_structures(trajectory_directory)
        if settings.get('archive_trajectories'):
            archive_trajectory_folder(trajectory_directory)
    accepted_design_count, trajectory_count = campaign_progress.campaign_status()
    if closing and design_worker_index() is None and accepted_design_count < requested_designs and max_trajectories and trajectory_count >= max_trajectories:
        print('\n' + campaign_budget_exhausted(max_trajectories, trajectory_count, accepted_design_count, requested_designs), flush=True)
    write_campaign_summary(project_folder)
    rank_accepted_designs(project_folder)
    if design_worker_index() is None and closing:
        print(campaign_closed(accepted_design_count, trajectory_count, RANKING_METRIC), flush=True)
    return trajectory_count

def print_campaign_header(settings: dict, project_folder: str, design_settings) -> None:
    protein_states, _multi_chain_binders, losses = initialize_design_trajectory(design_settings, jax.random.PRNGKey(design_settings.seed))
    chain_residues = {state: {chain: len(protein) for chain, protein in protein_complex.items()} for state, protein_complex in protein_states.items()}
    campaign = campaign_label(settings) or os.path.basename(os.path.normpath(project_folder))
    for line in campaign_header(campaign, settings, chain_residues, {name: loss.weight for name, loss in losses.items()}, tuple(design_settings.prepared_states)):
        print(line, flush=True)

def run_campaign(settings: dict, project_folder: str, af2_weights: str | None=None, mpnn_weights: str | None=None, key: jax.Array | None=None, max_trajectories: int | None=None, metadata: dict[str, str] | None=None) -> int:
    if max_trajectories is None:
        max_trajectories = settings.get('max_trajectories')
    key = jax.random.PRNGKey(settings.get('campaign_seed') or 0) if key is None else key
    selected_models = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)
    if selected_models.validation_pool_exhausted:
        print(f'validation held out on the monomer pool {selected_models.validation_models}: the {len(selected_models.design_models)} design models leave nothing of the multimer pool for a held-out validation set', flush=True)
    subbatch_size = campaign_subbatch_size(settings, design_residue_count(settings))
    attention_backend = settings.get('attention_backend', 'auto')
    use_cueq = bool(settings.get('use_cueq', False))
    length_bucket_size = campaign_length_bucket(settings)
    design_settings = build_design_settings(settings)
    if speaks_for_the_campaign():
        print_campaign_header(settings, project_folder, design_settings)
    target_lengths = {len(protein) for protein in prepare_targets(design_settings, longest_crop=True).values()}
    target_pad_length = padded_prediction_length(max(target_lengths), length_bucket_size) if len(target_lengths) > 1 else 0  #for multitargeting
    multi_chain_binders = (design_settings.binder_chains,) if design_settings.binder.copies > 1 and design_settings.oligomer_tie == 'symmetric' else ()  #for oligomers
    alphafold_model = AlphaFoldDesignModel(presets=selected_models.design_models, data_dir=af2_weights, max_cache_size=16, num_recycle=settings.get('design_recycles', DEFAULT_SETTINGS['design_recycles']), models=selected_models.design_models, cyclic_offset_mode=resolve_cyclic_offset_mode(settings), subbatch_size=subbatch_size, attention_backend=attention_backend, use_cueq=use_cueq, length_bucket_size=length_bucket_size, multi_chain_binders=multi_chain_binders, target_pad_length=target_pad_length)
    refuse_predictor_without_distogram(settings, alphafold_model)
    mpnn_model = ProteinMPNNSequenceModel(data_dir=mpnn_weights, max_cache_size=16, model_name=settings.get('mpnn_model', 'v_48_020'), variant=settings.get('mpnn_variant', 'negative'), omitted_amino_acids=design_settings.binder.omitted_amino_acids, amino_acid_bias=design_settings.binder.amino_acid_bias, multi_chain_binders=multi_chain_binders, length_bucket_size=length_bucket_size, target_pad_length=target_pad_length, redesign_max_positions=settings.get('redesign_max_positions'), redesign_position_temperature=settings.get('redesign_position_temperature', 1.0)) if mpnn_weights and (not settings.get('trajectory_only')) else None
    build_validation_model = lambda validation_models: AlphaFoldDesignModel(presets=validation_models, data_dir=af2_weights, max_cache_size=16, num_recycle=settings.get('validation_recycles', 3), cyclic_offset_mode=resolve_cyclic_offset_mode(settings), subbatch_size=subbatch_size, attention_backend=attention_backend, use_cueq=use_cueq, length_bucket_size=length_bucket_size, dropout=False, multi_chain_binders=multi_chain_binders, target_pad_length=target_pad_length)
    validation_model = build_validation_model(selected_models.validation_models) if mpnn_model else None
    if mpnn_model is None and max_trajectories is None:
        max_trajectories = DEFAULT_TRAJECTORY_ONLY_BUDGET
    arms = parameter_sweep_arms(settings)
    if not arms:
        return run_campaign_arm(settings, project_folder, alphafold_model, validation_model, mpnn_model, key, max_trajectories, build_validation_model, autotune=settings.get('autotune', True), metadata=metadata)
    arm_trajectories = arm_trajectory_budget(max_trajectories, len(arms))
    reached = {}
    for budget in sweep_block_budgets(arm_trajectories, int(parameter_sweep_options(settings)['block_trajectories'])):
        for arm, overrides in arms:
            arm_settings = {**settings, 'number_of_final_designs': arm_trajectories * int(settings.get('kept_sequences', DEFAULT_SETTINGS['kept_sequences'])), **overrides}
            print(f"\n=== sweep arm {arm} | {overrides or 'campaign settings'} | through trajectory {budget} of {arm_trajectories} ===", flush=True)
            reached[arm] = run_campaign_arm(arm_settings, os.path.join(project_folder, arm), alphafold_model, validation_model, mpnn_model, key, budget, build_validation_model, autotune=False, closing=False, metadata=metadata)
        if speaks_for_the_campaign():
            print(f'sweep record: {write_sweep_record(project_folder, design_settings, arms)}', flush=True)
    return sum(reached.values())

def launch_campaign(settings_path: str, setting_overrides: list[str] | tuple[str, ...]=(), metadata_path: str | None=None, af2_weights: str | None=None, mpnn_weights: str | None=None) -> int:
    settings = cleaned_campaign_settings(read_settings(settings_path, parse_setting_overrides(setting_overrides)))
    metadata = {**settings_provenance(settings, tuple(setting_overrides)), **(read_campaign_metadata(metadata_path) or {})}
    project_folder = settings.get('project_folder', DEFAULT_PROJECT_FOLDER)
    if af2_weights is None or mpnn_weights is None:
        resolved_af2_weights, resolved_mpnn_weights = model_weights()
        af2_weights = resolved_af2_weights if af2_weights is None else af2_weights
        mpnn_weights = resolved_mpnn_weights if mpnn_weights is None else mpnn_weights
    loaded_checkpoints = preflight_campaign(settings, project_folder, af2_weights, mpnn_weights)
    if not running_as_design_worker():
        write_campaign_metadata(project_folder, settings, settings_path, loaded_checkpoints, metadata)
    worker_status = dispatch_design_workers(settings, os.path.join(project_folder, 'workers'), design_residue_count(settings), worker_arguments=('--metadata', str(metadata_path)) if metadata_path else (), trajectory_budget=campaign_trajectory_budget(settings))
    if worker_status is not None:
        if parameter_sweep_arms(settings):
            print(f'sweep record: {write_sweep_record(project_folder, build_design_settings(settings), parameter_sweep_arms(settings))}', flush=True)
        return worker_status
    run_campaign(settings, project_folder, af2_weights=af2_weights, mpnn_weights=mpnn_weights, metadata=metadata)
    return 0
