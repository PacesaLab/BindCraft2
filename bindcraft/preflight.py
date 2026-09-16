import os
import jax
from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL
from bindcraft.campaign_output import DESIGN_STAGES
from bindcraft.design_workers import running_as_design_worker
from bindcraft.loss import build_losses, induced_fit_hinge_names
from bindcraft.protein import ResidueFlags, listed_offending_residues, partly_trimmed_residues, scaffold_edit_flags
from bindcraft.protein_preparation import prepare_binder_chains, prepare_targets
from bindcraft.model_weights import alphafold_parameter_file, mpnn_variant_directory
from bindcraft.settings import DESIGN_STAGE_NAMES, DEFAULT_VALIDATION_MODEL_COUNT, BinderDesignSettings, PredictionModelSelection, build_design_settings, conflicting_campaign_features, detarget_state_names, is_fasta, merged_gradient_sequence_updates, named_prediction_models, select_design_and_validation_models, validates_on_multimer

CAMPAIGN_OUTPUT_NAMES = (*DESIGN_STAGES, '.campaign_state.json', 'trajectories.csv', 'candidates.csv', 'accepted.csv', 'trajectories', 'accepted')

class CampaignPreflightError(ValueError):
    pass

def campaign_redesigns_sequences(settings: dict, mpnn_weights: str | None) -> bool:
    return bool(mpnn_weights) and (not settings.get('trajectory_only'))

def campaign_prediction_models(settings: dict) -> PredictionModelSelection:
    return select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)

def crossed_over_models(settings: dict) -> tuple[str, ...]:
    design_models = named_prediction_models(settings.get('design_models'))
    return tuple(model for model in named_prediction_models(settings.get('validation_models')) if model in design_models)

def prediction_model_problems(settings: dict) -> list[str]:
    crossed_over = crossed_over_models(settings)
    if not crossed_over:
        return []
    return [f'design_models and validation_models both name {", ".join(crossed_over)}: a design is never scored by a model that shaped it, so validation is held out of design. Drop {"it" if len(crossed_over) == 1 else "them"} from one side, or give one side a count and the split is made for you.']

def campaign_parameter_files(settings: dict, af2_weights: str | None, mpnn_weights: str | None) -> dict[str, str | None]:
    selected_models = campaign_prediction_models(settings)
    required_models = dict.fromkeys(selected_models.design_models + (selected_models.validation_models if campaign_redesigns_sequences(settings, mpnn_weights) else ()))
    return {model_name: alphafold_parameter_file(af2_weights, model_name) if af2_weights else None for model_name in required_models}

def campaign_redesign_checkpoint(settings: dict, mpnn_weights: str | None) -> str | None:
    return os.path.join(mpnn_variant_directory(mpnn_weights, settings.get('mpnn_variant', 'negative')), f"{settings.get('mpnn_model', 'v_48_020')}.npz") if campaign_redesigns_sequences(settings, mpnn_weights) else None

def campaign_checkpoints(settings: dict, af2_weights: str | None, mpnn_weights: str | None) -> tuple[str, ...]:
    resolved_paths = (*campaign_parameter_files(settings, af2_weights, mpnn_weights).values(), campaign_redesign_checkpoint(settings, mpnn_weights))
    return tuple(path for path in resolved_paths if path and os.path.isfile(path))

def nearest_existing_output_ancestor(project_folder: str) -> str:
    folder = os.path.abspath(project_folder)
    while not os.path.isdir(folder) and os.path.dirname(folder) != folder:
        folder = os.path.dirname(folder)
    return folder

def campaign_output_in_folder(project_folder: str) -> list[str]:
    if not os.path.isdir(project_folder):
        return []
    campaign_folders = [project_folder] + sorted(entry.path for entry in os.scandir(project_folder) if entry.is_dir())
    return sorted({os.path.relpath(os.path.join(folder, name), project_folder) for folder in campaign_folders for name in CAMPAIGN_OUTPUT_NAMES if os.path.exists(os.path.join(folder, name))})

COMMA_SEPARATED_SETTINGS = frozenset({'chains', 'coldspots', 'gpu_ids', 'hotspots', 'mutate_positions', 'scaffold_edits'})

def cleaned_setting(name: str, value):
    if isinstance(value, str):
        if '\n' in value.strip():
            return value
        return ','.join(''.join(span.split()) for span in value.split(',') if span.strip()) if name in COMMA_SEPARATED_SETTINGS else value.strip()
    if isinstance(value, dict):
        return {key.strip() if isinstance(key, str) else key: cleaned_setting(key, entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [cleaned_setting(name, entry) for entry in value]
    return value

def cleaned_campaign_settings(settings: dict) -> dict:
    return {name: cleaned_setting(name, value) for name, value in settings.items()}

def structure_design_inputs(design_settings: BinderDesignSettings) -> list[tuple[str, str]]:
    #inline structure or FASTA target case
    design_inputs = [(f'target {target.name!r}', target.path) for target in design_settings.targets] + ([('binder_scaffold', design_settings.binder.scaffold)] if design_settings.binder.scaffold else [])
    return [(label, path) for label, path in design_inputs if '\n' not in path]

def partly_trimmed_input_residues(design_settings: BinderDesignSettings) -> str:
    trimmed = [(label, partly_trimmed_residues(path)) for label, path in structure_design_inputs(design_settings) if not is_fasta(path) and os.path.isfile(path)]
    return ' | '.join(f'{label} dropped {listed_offending_residues(list(residues))}' for label, residues in trimmed if residues)

PARATOPE_DEPENDENT_LOSSES = ('binder_coldspot', 'binder_intra_hotspot', 'binder_intra_coldspot')

def inert_paratope_losses(settings: dict) -> tuple[str, ...]:
    weighted = tuple(name for name in PARATOPE_DEPENDENT_LOSSES if settings.get(f'weights_{name}') or name in (settings.get('losses') or {}))
    if not weighted:
        return ()
    edit_flags = scaffold_edit_flags(str(settings.get('mutate_positions') or ''))
    paratope = [flags for flags in edit_flags if flags & ResidueFlags.CONTACT]
    framework = [flags for flags in edit_flags if not flags & ResidueFlags.CONTACT]
    whole_binder = {name for name in weighted if (((settings.get('losses') or {}).get(name) or {}).get('params') or {}).get('designed_only') is False}
    return tuple(name for name in weighted if not (paratope and (framework or name in whole_binder)))

def paratope_declaration_problems(design_settings: BinderDesignSettings) -> list[str]:
    inert = inert_paratope_losses(design_settings.settings)
    return [f"{', '.join(inert)} read a paratope this campaign does not declare, so they would carry a weight and steer nothing. mutate_positions has to name both kinds of position: an edit written plain is paratope, one written with * is framework. Drop the loss or declare the split."] if inert else []

def target_rotation_problems(design_settings: BinderDesignSettings) -> list[str]:
    target_names = [state.name for state in design_settings.prepared_states]
    starved_stages = [f'{design_stage}_steps {stage_rounds}' for design_stage, stage_rounds in merged_gradient_sequence_updates(design_settings).items() if 0 < stage_rounds < len(target_names)]
    return [f"a design stage has fewer rounds than the {len(target_names)} targets its filter scores ({', '.join(target_names)}), so the rotation cannot reach them all: {', '.join(starved_stages)}"] if starved_stages else []

def induced_fit_problems(design_settings: BinderDesignSettings) -> list[str]:
    target_names = [state.name for state in design_settings.prepared_states]
    hinge_names = induced_fit_hinge_names(build_losses(design_settings.settings))
    if not hinge_names:
        return []
    problems = []
    if len(target_names) > 1:
        problems.append(f"induced-fit design ({', '.join(hinge_names)}) freezes one bound structure for its binder-alone block, so it designs against exactly one target; this campaign names {len(target_names)} ({', '.join(target_names)})")
    if int(design_settings.settings.get('induced_fit_monomer_steps', 30)) < 1:
        problems.append(f"induced-fit design ({', '.join(hinge_names)}) with no binder-alone block never predicts the unbound fold its hinge measures against, so every stage would run without the hinge")
    return problems

def feature_conflict_problems(design_settings: BinderDesignSettings) -> list[str]:
    return [f'{first} and {second} cannot both hold: {reason}' for first, second, reason in conflicting_campaign_features(design_settings.settings)]

FEATURE_VALIDATORS = (paratope_declaration_problems, target_rotation_problems, induced_fit_problems, feature_conflict_problems)

def feature_problems(design_settings: BinderDesignSettings) -> list[str]:
    return [problem for validate in FEATURE_VALIDATORS for problem in validate(design_settings)]

def checkpoint_problems(settings: dict, selected_models, af2_weights: str | None, mpnn_weights: str | None) -> list[str]:
    problems = []
    redesigns_sequences = campaign_redesigns_sequences(settings, mpnn_weights)
    if redesigns_sequences and validates_on_multimer(settings) and selected_models.validation_pool_exhausted:
        held_out = len(MULTIMER_POOL) - len(selected_models.design_models)
        problems.append(f'validation_model "multimer" needs multimer models left over to validate on: {len(selected_models.design_models)} design models leave {held_out} of the {len(MULTIMER_POOL)}-model pool, and validation asks for {settings.get("validation_models", DEFAULT_VALIDATION_MODEL_COUNT)}. Drop "design_models" and the split is made for you, or ask for fewer validation_models, or set validation_model to "monomer". A scaffolded binder and an oligomer ask for the multimer pool on their own, because the monomer one cannot resolve a framework paratope or read an assembly as one molecule.')
    if not af2_weights:
        problems.append('AlphaFold parameters are not on this machine: run "bindcraft fetch-weights" to download them')
    elif af2_weights and (not os.path.isdir(af2_weights)):
        problems.append(f'AlphaFold parameter directory does not exist: {af2_weights}')
    elif af2_weights:
        unavailable_models = [model_name for model_name, parameter_file in campaign_parameter_files(settings, af2_weights, mpnn_weights).items() if parameter_file is None]
        if unavailable_models:
            problems.append(f"AlphaFold parameters missing from {af2_weights}: {', '.join((f'params_{model_name}.npz' for model_name in unavailable_models))}")
    if settings.get('trajectory_only'):
        if not settings.get('max_trajectories'):
            problems.append('trajectory_only needs max_trajectories: no design is ever accepted, so number_of_final_designs cannot stop the campaign')
    elif not mpnn_weights:
        problems.append('ProteinMPNN weights are not configured: they ship with the package, so BINDCRAFT_MPNN_WEIGHTS names a directory this installation does not hold. Unset it, or set "trajectory_only": true to run gradient trajectories without sequence redesign, which accepts no design')
    else:
        try:
            redesign_checkpoint = campaign_redesign_checkpoint(settings, mpnn_weights)
        except ValueError as error:
            problems.append(str(error))
        else:
            if not os.path.isfile(redesign_checkpoint):
                problems.append(f'no ProteinMPNN checkpoint at {redesign_checkpoint}')
    return problems

def design_input_problems(design_settings: BinderDesignSettings) -> list[str]:
    missing_inputs = [f'{label}: no file at {path}' for label, path in structure_design_inputs(design_settings) if not os.path.isfile(path)]
    if missing_inputs:
        return missing_inputs
    try:
        prepare_targets(design_settings)
        prepare_binder_chains(design_settings, jax.random.PRNGKey(design_settings.seed))
    except (ValueError, KeyError, OSError) as error:
        trimmed_inputs = partly_trimmed_input_residues(design_settings)
        return [f'design inputs did not parse: {error}' + (f'; an incomplete trim left {trimmed_inputs}' if trimmed_inputs else '')]
    return []

def undecided_avoidance(design_settings: BinderDesignSettings) -> str:
    """The off-targets a campaign steers away from and then accepts designs against without asking whether they got away."""
    settings = design_settings.settings
    if settings.get('max_detarget_interface_residues_final') is not None or any(settings.get(f'max_detarget_iptm_{stage}') is not None for stage in DESIGN_STAGE_NAMES):
        return ''
    avoided = detarget_state_names(design_settings)
    return f"{', '.join(avoided)} avoided by weight alone: no ceiling decides it, so a design holding the off-target is accepted" if avoided else ''

def output_folder_problems(settings: dict, project_folder: str) -> list[str]:
    problems = []
    output_ancestor = nearest_existing_output_ancestor(project_folder)
    if not os.access(output_ancestor, os.W_OK):
        problems.append(f'output folder {project_folder} cannot be written: no write permission on {output_ancestor}')
    if not running_as_design_worker() and (not settings.get('resume')):
        campaign_output = campaign_output_in_folder(project_folder)
        if campaign_output:
            problems.append(f"""{project_folder} already holds campaign output ({', '.join(campaign_output)}); use a fresh folder or set "resume": true""")
    return problems

def preflight_campaign(settings: dict, project_folder: str, af2_weights: str | None=None, mpnn_weights: str | None=None) -> tuple[str, ...]:
    settings.update(cleaned_campaign_settings(settings))
    crossover = prediction_model_problems(settings)
    if crossover:
        raise CampaignPreflightError('\n'.join(crossover))
    try:
        design_settings = build_design_settings(settings)
        selected_models = campaign_prediction_models(settings)
    except (ValueError, KeyError, OSError) as error:
        raise CampaignPreflightError(f'settings did not resolve: {error}')
    problems = checkpoint_problems(settings, selected_models, af2_weights, mpnn_weights) + design_input_problems(design_settings) + output_folder_problems(settings, project_folder) + feature_problems(design_settings)
    if problems:
        raise CampaignPreflightError('\n'.join(problems))
    redesigns_sequences = campaign_redesigns_sequences(settings, mpnn_weights)
    redesign_plan = f"{settings.get('mpnn_variant', 'negative')}/{settings.get('mpnn_model', 'v_48_020')}" if redesigns_sequences else 'trajectory-only'
    if running_as_design_worker():
        return campaign_checkpoints(settings, af2_weights, mpnn_weights)
    trimmed_inputs = partly_trimmed_input_residues(design_settings)
    if trimmed_inputs:
        print(f'campaign inputs: {trimmed_inputs}', flush=True)
    unchecked_avoidance = undecided_avoidance(design_settings)
    if unchecked_avoidance:
        print(f'campaign negative selection: {unchecked_avoidance}', flush=True)
    print(f"campaign preflight: design {','.join(selected_models.design_models)} | validation {(','.join(selected_models.validation_models) if redesigns_sequences else 'validation off')} | redesign {redesign_plan} | targets {','.join((target.name for target in design_settings.targets))} | output {project_folder}", flush=True)
    return campaign_checkpoints(settings, af2_weights, mpnn_weights)
