import json
import copy
import os
import difflib
import inspect
import math
import random
from dataclasses import dataclass, field
from typing import Callable, NamedTuple
from bindcraft.filters import DesignFilter, REGISTERED_FILTER_METRICS, build_filters
from bindcraft.loss import BINDER_ALONE_LOSS_NAMES, PROTOMER_SCOPED_LOSSES, REGISTERED_LOSSES
from bindcraft.model_weights import DEFAULT_MPNN_MODEL, DEFAULT_MPNN_VARIANT
from bindcraft.protein import AMINO_ACIDS, ResidueFlags, scaffold_edit_flags, structure_chain_names, target_chain_name
from pathlib import Path

FASTA_SUFFIXES = '.fasta', '.fa', '.faa'
DEFAULT_IDR_CROP_LENGTHS = 10, 40
DEFAULT_VALIDATION_CROP_FLANK = 5
DEFAULT_VALIDATION_MODEL_COUNT = 2
MULTIMER_MODEL_POOL = tuple(f'model_{index}_multimer_v3' for index in range(1, 6))
DEFAULT_INDUCED_FIT_INTERFACE_RMSD = 5.0
DEFAULT_INDUCED_FIT_TM_TARGET = 0.6
DEFAULT_DISORDERED_TARGET_PLDDT = 0.6
DEFAULT_DETARGET_INTERFACE_RESIDUES = 3
#an off-target sits between the binding rounds, as the multitargeting work it comes from does, so the repulsion it carries is applied all through a stage
DEFAULT_DETARGET_CHECK_INTERVAL = 1

def is_fasta(path) -> bool:
    return Path(path or '').suffix.lower() in FASTA_SUFFIXES

def has_fasta_target(settings: dict) -> bool:
    return any(is_fasta(target.get('target_path')) for target in settings.get('targets', []))
LOSS_PARAMETERS = {'interface_contact_distance': ('interface_contacts', 'cutoff'), 'non_contact_distance': ('non_contact', 'cutoff'), 'termini_distance_threshold': ('termini_distance', 'threshold_distance'), 'disulfide_distance': ('disulfide', 'distance'), 'disulfide_sigma': ('disulfide', 'sigma'), 'disulfide_sequence_separation': ('disulfide', 'sequence_separation'), 'disulfide_temperature': ('disulfide', 'temperature'), 'induced_fit_delta': ('induced_fit_interface', 'interface_rmsd_target'), 'induced_fit_interface_cutoff': ('induced_fit_interface', 'cutoff'), 'induced_fit_tm_target': ('induced_fit_global', 'tm_target'), 'humanization_species': ('humanization', 'species'), 'humanization_coupling_weight': ('humanization', 'coupling_weight'), 'humanization_hydro_weight': ('humanization', 'hydrophobicity_weight'), 'humanization_mhc2_weight': ('humanization', 'mhc_class_ii_weight'), 'exposed_loops_measure': ('exposed_loops', 'measure'), 'exposed_loops_distinguish_sheets': ('exposed_loops', 'distinguish_sheets')}
DOMAIN_PARAMETERS = 'n_domains', 'min_domain_size', 'max_domain_size', 'max_domains', 'domain_rg_weight', 'domain_sep_weight', 'domain_contact_cutoff', 'domain_pae_margin', 'domain_linker_gap', 'domain_linker_sharpness', 'domain_linker_helix_weight'
DESIGN_STAGE_NAMES = 'screen', 'refine', 'anneal', 'harden', 'mutate', 'final'
CAMPAIGN_SETTING_NAMES = frozenset({'aa_bias', 'archive_trajectories', 'attention_backend', 'auto_multi_gpu', 'autotune', 'autotune_loss_weights', 'betasheet_reopt_extra_anneal_steps', 'betasheet_reopt_extra_refine_steps', 'betasheet_reopt_recycles', 'betasheet_reopt_trigger', 'binder_chain', 'binder_lengths', 'binder_name', 'binder_scaffold', 'binder_sequences', 'binder_shapes', 'campaign_name', 'campaign_seed', 'compile_next_length', 'copies', 'crop_fasta_sequence', 'cyclic_offset_mode', 'cyclize_peptide', 'design_dropout', 'design_models', 'design_recycles', 'design_workers', 'desperation', 'desperation_trajectories', 'detarget_check_interval', 'domain_linker_fix_cut', 'enough_passing_sequences', 'filters', 'forced_targeting', 'forced_targeting_shell', 'gpu_ids', 'hash_design_names', 'idr_crop_count', 'induced_fit_monomer_adaptive', 'induced_fit_monomer_chunk', 'induced_fit_monomer_plddt', 'induced_fit_monomer_steps', 'induced_fit_mpnn_designed_share', 'induced_fit_mpnn_shell', 'induced_fit_mpnn_threshold', 'induced_fit_steps', 'initial_guess', 'kept_sequences', 'length_bucket_size', 'losses', 'max_binder_chain_breaks_final', 'max_binder_free_cysteines_final', 'max_coldspot_contact_final', 'max_cyclic_closure_distance_final', 'max_detarget_interface_residues_final', 'max_detarget_iptm', 'max_detarget_rounds', 'max_exposed_loop_fraction_final', 'max_helix_fraction_final', 'max_induced_fit_tm_final', 'max_interdomain_contact_final', 'max_mhc_anchor_score_final', 'max_off_epitope_contact_final', 'max_off_paratope_contact_final', 'max_oligomer_symmetry_rmsd_final', 'max_protease_site_score_final', 'max_scaffold_framework_rmsd_final', 'max_surface_hydrophobicity_final', 'max_termini_distance_final', 'max_terminus_exposure_final', 'max_trajectories', 'max_workers_per_gpu', 'min_binder_disulfides_final', 'min_domain_separation_ratio_final', 'min_epitope_residues_contacted_final', 'min_framework_packing_final', 'min_hotspot_contact_final', 'min_induced_fit_interface_rmsd_final', 'min_interface_buried_area_final', 'min_receptor_chains_contacted_final', 'min_scaffold_sequence_retained_final', 'min_target_crop_length_final', 'min_target_plddt_final', 'min_termini_away_cosine_final', 'mpnn_fix_linker', 'mpnn_model', 'mpnn_variant', 'multitarget_best_round', 'multitarget_cumulative_filter', 'multitarget_filter_models', 'multitarget_merged_gradient_budget', 'multitarget_merged_gradients', 'multitarget_rounds_per_target', 'multitarget_steps', 'multitarget_swap_patience', 'multitarget_swap_threshold', 'multitarget_tied_redesign', 'multitarget_warmup_patience', 'mutate_positions', 'number_of_final_designs', 'oligomer_tie', 'parameter_sweep', 'project_folder', 'redesign_interface', 'redesign_max_positions', 'redesign_position_temperature', 'relax_accepted_designs', 'relax_learning_rate', 'relax_min_sep', 'relax_overlap_tol', 'relax_restraint_backbone', 'relax_restraint_sidechain', 'relax_steps', 'relax_weight_bond', 'relax_weight_clash', 'resume', 'save_binder_monomers', 'save_design_animations', 'save_design_frames', 'save_design_trajectory', 'save_design_sequences', 'save_failed_refolds', 'save_failed_trajectories', 'save_loss_plots', 'sequence_candidates', 'sparse_output', 'subbatch_size', 'target_chain', 'targets', 'trajectory_only', 'use_cueq', 'validation_crop_flank', 'validation_model', 'validation_models', 'validation_recycles', 'worker_launch_stagger', 'workers_per_gpu'})
FINAL_CONFIDENCE_FILTERS = {'min_monomer_plddt_final': 'Unbound_Binder_pLDDT', 'min_ptm_final': 'pTM', 'min_iptm_final': 'i_pTM', 'max_ipae_final': 'i_pAE'}
TARGET_SETTING_NAMES = frozenset({'name', 'target_path', 'chains', 'hotspots', 'coldspots', 'weight', 'objective'})
METRIC_ENTRY_NAMES = frozenset({'params', 'prediction_state'})
FILTER_ENTRY_NAMES = METRIC_ENTRY_NAMES | {'threshold', 'higher', 'mandatory'}
CORE_DEFAULTS = json.loads((Path(__file__).parent.parent / 'settings' / 'core' / 'default.json').read_text())
DEFAULT_LOSSES = CORE_DEFAULTS['losses']
DEFAULT_FILTERS = CORE_DEFAULTS['filters']
DEFAULT_SETTINGS = CORE_DEFAULTS

@dataclass(frozen=True)
class PreparedTargetState:
    name: str
    source_target: str
    path: str | None
    objective: str
    weight: float
    target_chain: str
    chains: str = ''
    hotspots: str = ''
    coldspots: str = ''
    crop_index: int = 0
    crop_length_bounds: tuple[int, int] | None = None

def resolve_crop_length_bounds(settings: dict) -> tuple[int, int] | None:
    crop_lengths = settings.get('crop_fasta_sequence', DEFAULT_IDR_CROP_LENGTHS)
    if crop_lengths is None or crop_lengths is False:
        return None
    crop_length_bounds = (crop_lengths, crop_lengths) if isinstance(crop_lengths, int) and (not isinstance(crop_lengths, bool)) else crop_lengths
    if not isinstance(crop_length_bounds, (tuple, list)) or len(crop_length_bounds) != 2 or min(crop_length_bounds) < 1:
        raise ValueError('crop_fasta_sequence must be a positive length, two-value range, or false')
    return (min(crop_length_bounds), max(crop_length_bounds))

def resolve_validation_crop_flank(settings: dict) -> int:
    flank = settings.get('validation_crop_flank', DEFAULT_VALIDATION_CROP_FLANK)
    if not isinstance(flank, int) or isinstance(flank, bool) or flank < 0:
        raise ValueError('validation_crop_flank must be a residue count of zero or more')
    return flank

def resolve_amino_acid_bias(settings: dict) -> dict[str, float]:
    amino_acid_weights = {str(amino_acid): float(weight) for amino_acid, weight in (settings.get('aa_bias') or {}).items()}
    unknown = ''.join(sorted(set(amino_acid_weights) - set(AMINO_ACIDS)))
    if unknown:
        raise ValueError(f'aa_bias names {unknown!r}, which is not an amino acid; write each biased residue as an uppercase one-letter code and its propensity, as in {{"C": 0, "W": 0.4}}')
    return {amino_acid: math.log(weight) for amino_acid, weight in amino_acid_weights.items() if weight > 0}

def resolve_omitted_amino_acids(settings: dict) -> str:
    excluded_by_propensity = {str(amino_acid) for amino_acid, weight in (settings.get('aa_bias') or {}).items() if float(weight) <= 0}
    return ''.join(amino_acid for amino_acid in AMINO_ACIDS if amino_acid in excluded_by_propensity)

CYCLIC_OFFSET_MODES = 'distance', 'direction', 'neighbours'

def resolve_cyclic_offset_mode(settings: dict) -> str:
    named_mode = settings.get('cyclic_offset_mode', 'direction')
    if named_mode not in CYCLIC_OFFSET_MODES:
        raise ValueError(f'unknown cyclic_offset_mode {named_mode!r}; use one of {", ".join(CYCLIC_OFFSET_MODES)}')
    return named_mode

def resolve_binder_chains(copies: int, scaffold: str | None=None) -> tuple[str, ...]:
    scaffold_chain_count = len(structure_chain_names(scaffold)) if scaffold else 1
    if scaffold_chain_count > 1 and int(copies) > 1:
        raise ValueError(f'binder_scaffold holds {scaffold_chain_count} chains and copies is {copies}; a multi-chain scaffold is one binder spanning its chains, so it is designed as one copy')
    chain_count = max(int(copies), scaffold_chain_count)
    return ('binder',) if chain_count <= 1 else tuple(f'binder_{index}' for index in range(chain_count))

OLIGOMER_TIES = frozenset({'symmetric', 'none'})

def resolve_oligomer_tie(settings: dict) -> str:
    oligomer_tie = str(settings.get('oligomer_tie', 'symmetric'))
    if oligomer_tie not in OLIGOMER_TIES:
        raise ValueError(f'unknown oligomer_tie {oligomer_tie!r}; use one of {", ".join(sorted(OLIGOMER_TIES))}')
    return oligomer_tie

def resolve_designed_binder_chain(settings: dict, binder_chains: tuple[str, ...]=()) -> str:
    return settings.get('binder_chain') or (binder_chains or resolve_binder_chains(settings.get('copies', 1), settings.get('binder_scaffold')))[0]

def resolve_prepared_states(settings: dict) -> tuple[PreparedTargetState, ...]:
    target_chain_prefix = settings.get('target_chain', 'target')
    prepared_states = []
    for target in settings.get('targets', []):
        path = target.get('target_path')
        crop_count = max(1, int(settings.get('idr_crop_count', 1))) if is_fasta(path) else 1
        #only a disordered target is cropped
        crop_length_bounds = resolve_crop_length_bounds(settings) if is_fasta(path) else None
        weight = float(target.get('weight', 1.0))
        objective = 'detarget' if target.get('objective') == 'detarget' or weight < 0 else 'target'
        weight = -abs(weight) if objective == 'detarget' else weight
        for crop_index in range(crop_count):
            name = target['name'] if crop_count == 1 else f"{target['name']}_epitope_{crop_index + 1}"
            prepared_states.append(PreparedTargetState(name, target['name'], path, objective, weight, target_chain_name(target_chain_prefix, name), target.get('chains', ''), target.get('hotspots', ''), target.get('coldspots', ''), 0 if crop_count == 1 else crop_index + 1, crop_length_bounds))
    return tuple(prepared_states)

def record_modality_filter(settings: dict, name: str, unrejecting_threshold: float, higher: bool=False, threshold_setting: str='', **params) -> None:
    if isinstance(settings.get('filters'), dict):
        configured_threshold = settings.get(threshold_setting)
        threshold = float(unrejecting_threshold if configured_threshold is None else configured_threshold)
        settings['filters'].setdefault(name, {'threshold': threshold, 'higher': higher, 'mandatory': configured_threshold is not None, **({'params': params} if params else {})})

class ModalityCheck(NamedTuple):
    metric: str
    unrejecting_threshold: float
    higher: bool = False
    threshold_setting: str = ''
    parameters: Callable[[dict], dict] | None = None

class ConfigurationRequest(NamedTuple):
    overrides: dict
    loss_settings: dict

class CampaignFeature(NamedTuple):
    name: str
    requested: Callable[[dict], bool]
    shorthand: str | Callable[[dict], str] = ''
    switch: str = ''
    configure: Callable[[dict, ConfigurationRequest], None] | None = None
    checks: tuple[ModalityCheck, ...] = ()
    note: Callable[[dict], str] | None = None
    conflicts: tuple[tuple[str, str], ...] = ()

def initial_guess_note(settings: dict) -> str:
    if design_seeds_from_given_coordinates(settings):
        return 'big bang: the gradient stages start from the coordinates on hand, which for a binder folded from nothing is the origin'
    return 'initial guess: the re-prediction starts from the pose the trajectory folded, and the gradient stages start where they always do'

def design_seeds_from_given_coordinates(settings: dict) -> bool:
    return bool(settings.get('bigbang_initialization'))

def validation_seeds_from_given_coordinates(settings: dict) -> bool:
    return bool(settings.get('initial_guess')) or bool(settings.get('bigbang_initialization'))

def binder_scaffold_name(settings: dict) -> str:
    return os.path.splitext(os.path.basename(str(settings.get('binder_scaffold') or '')))[0]

def scaffold_check_parameters(settings: dict) -> dict:
    return {'scaffold': settings['binder_scaffold'], 'scaffold_edits': settings.get('mutate_positions', '')}

def epitope_check_parameters(settings: dict) -> dict:
    return {'epitope_cutoff': settings['forced_targeting_shell']} if 'forced_targeting_shell' in settings else {}

def domain_check_parameters(settings: dict) -> dict:
    return {'n_domains': settings.get('n_domains', 2), 'min_domain_size': settings.get('min_domain_size', 50), 'domain_contact_cutoff': settings.get('domain_contact_cutoff', 8.0)}

def humanization_check_parameters(settings: dict) -> dict:
    return settings.get('losses', {}).get('humanization', {}).get('params', {})

def terminus_exposure_check_parameters(settings: dict) -> dict:
    return {key: value for key, value in settings.get('losses', {}).get('exposed_termini', {}).get('params', {}).items() if key == 'terminus_length'}

def configure_multidomain(settings: dict, request: ConfigurationRequest) -> None:
    settings['weights_ptm_loss'] = 0.0

def configure_oligomer(settings: dict, request: ConfigurationRequest) -> None:
    settings.setdefault('validation_model', 'multimer')
    for name in PROTOMER_SCOPED_LOSSES:
        settings['losses'].setdefault(name, {}).setdefault('params', {}).setdefault('per_protomer', True)

def configure_coldspots(settings: dict, request: ConfigurationRequest) -> None:
    if 'weights_coldspot_repel' not in request.overrides and 'coldspot_repel' not in request.loss_settings:
        settings['weights_coldspot_repel'] = 1.0
    settings.setdefault('max_coldspot_contact_final', 0.05)

def configure_negative_selection(settings: dict, request: ConfigurationRequest) -> None:
    settings.setdefault('weights_non_contact', 1.0)
    #a repulsion weight steers away from an off-target without ever saying the design got away, and interface confidence does not say it either: a peptide can read 0.27 interface pTM with its whole face on the off-target
    settings.setdefault('max_detarget_interface_residues_final', DEFAULT_DETARGET_INTERFACE_RESIDUES)

def configure_fold_conditioning(settings: dict, request: ConfigurationRequest) -> None:
    if isinstance(settings['filters'], dict):
        settings['filters'].setdefault('Binder_RMSD', {'threshold': None, 'higher': False})
    #a scaffold naming its own framework rows is held off those rows, which is how a VHH keeps the face its long CDR3 packs onto designable; one naming none is held off the whole binder outside its paratope
    edit_flags = scaffold_edit_flags(str(settings.get('mutate_positions') or ''))
    paratope_rows = [flags for flags in edit_flags if flags & ResidueFlags.CONTACT]
    framework_rows = [flags for flags in edit_flags if not flags & ResidueFlags.CONTACT]
    if paratope_rows:
        if not framework_rows and 'binder_coldspot' not in request.loss_settings:
            settings['losses'].setdefault('binder_coldspot', {}).setdefault('params', {})['designed_only'] = False
        if 'weights_binder_coldspot' not in request.overrides:
            settings['weights_binder_coldspot'] = 1.0
        settings.setdefault('max_off_paratope_contact_final', 0.2)

def configure_focused_epitope(settings: dict, request: ConfigurationRequest) -> None:
    settings.setdefault('weights_target_rmsd', 1.0)

def configure_disordered_target(settings: dict, request: ConfigurationRequest) -> None:
    for name, weight in {'interface_pae': 0.3, 'target_plddt': 1.0, 'target_helicity': -0.3}.items():
        if f'weights_{name}' not in request.overrides:
            settings[f'weights_{name}'] = weight
    if 'interface_contact_distance' not in request.overrides and 'interface_contacts' not in request.loss_settings:
        settings['losses']['interface_contacts']['params']['cutoff'] = 10.0
    if isinstance(settings['filters'], dict):
        settings['filters'].setdefault('Target_pLDDT', {'threshold': settings.get('min_target_plddt_final', DEFAULT_DISORDERED_TARGET_PLDDT), 'higher': True})

def configure_induced_fit(settings: dict, request: ConfigurationRequest) -> None:
    if isinstance(settings['filters'], dict):
        settings['filters'].setdefault('Induced_Fit_Interface_RMSD', {'threshold': settings.get('min_induced_fit_interface_rmsd_final', DEFAULT_INDUCED_FIT_INTERFACE_RMSD), 'higher': True})

def configure_fold_switching(settings: dict, request: ConfigurationRequest) -> None:
    if isinstance(settings['filters'], dict):
        settings['filters'].setdefault('Induced_Fit_TM', {'threshold': settings.get('max_induced_fit_tm_final', settings.get('induced_fit_tm_target', DEFAULT_INDUCED_FIT_TM_TARGET)), 'higher': False})

def terminus_away_feature(terminus: str) -> 'CampaignFeature':
    return CampaignFeature(f'{terminus} terminus away from the target', lambda settings, terminus=terminus: bool(settings.get(f'weights_{terminus}_terminus_away')),
                           checks=(ModalityCheck(f'{terminus.upper()}_Terminus_Away_Cosine', -1.0, True, f'min_{terminus}_terminus_away_cosine_final'),))

def mutation_check_parameters(settings: dict) -> dict:
    return {'parent_sequences': tuple(resolve_binder_sequences(settings).values())}

def configure_given_binder_sequences(settings: dict, request: ConfigurationRequest) -> None:
    #a sequence that was given fixes its own length, and no stage can insert or delete a residue, so the lengths are the given ones whatever a preset or the campaign asked for
    settings['binder_lengths'] = tuple(sorted({len(sequence) for sequence in resolve_binder_sequences(settings).values()}))
    #a scaffold, unlike a length, cannot be reconciled: it decides what the binder starts as, and so does the sequence
    if settings.get('binder_scaffold'):
        raise NotImplementedError('binder_sequences and binder_scaffold both decide what the binder starts as: seeding a scaffold framework with a sequence that is given is not implemented. Drop one of them, or the scaffold modality that supplies it.')
    #one trajectory per sequence that was given, so every one is folded and designed from exactly once
    parent_count = len(resolve_binder_sequences(settings))
    settings.setdefault('max_trajectories', parent_count)
    #the accepted count is what every trajectory could contribute at most, so it stops nothing: the trajectory budget above is the limit that matters here
    kept_per_trajectory = min(int(settings.get('kept_sequences', 1) or 1), int(settings.get('sequence_candidates', 1) or 1))
    settings.setdefault('number_of_final_designs', parent_count * kept_per_trajectory)

def given_binder_sequences_note(settings: dict) -> str:
    parents = resolve_binder_sequences(settings)
    designed = 'redesigned by ProteinMPNN alone' if settings.get('mpnn_redesign') else 'carried into the gradient stages as their starting sequence'
    return f"given binder sequence: {len(parents)} of {'/'.join(str(len(sequence)) for sequence in parents.values())} residue(s), one trajectory each, {designed}"

def configure_mpnn_redesign(settings: dict, request: ConfigurationRequest) -> None:
    if not resolve_binder_sequences(settings):
        raise ValueError('mpnn_redesign redesigns the sequences written under binder_sequences, and none were given')

def mpnn_redesign_note(settings: dict) -> str:
    parents = resolve_binder_sequences(settings)
    cap = settings.get('redesign_max_positions')
    limit = 'no cap on substitutions' if cap is None else ('evaluation only, no substitutions' if not int(cap) else f'at most {int(cap)} substitution(s)')
    return f"MPNN redesign: {len(parents)} given sequence(s) of {'/'.join(str(len(sequence)) for sequence in parents.values())} residue(s), {limit}"

def designs_peptide(settings: dict) -> bool:
    lengths = campaign_binder_lengths(settings.get('binder_lengths'))
    return bool(lengths) and max(lengths) < 36 and not settings.get('cyclize_peptide')

CAMPAIGN_FEATURES = (
    CampaignFeature('multi-chain binder', lambda settings: int(settings.get('copies', 1) or 1) > 1, shorthand='oligomer', switch='copies', configure=configure_oligomer,
                    checks=(ModalityCheck('Oligomer_Symmetry_RMSD', float('inf'), threshold_setting='max_oligomer_symmetry_rmsd_final'),),
                    conflicts=(('multidomain binder', 'the domain split does not engage across oligomer copies'),)),
    CampaignFeature('initial guess', lambda settings: bool(settings.get('initial_guess')) or bool(settings.get('bigbang_initialization')),
                    note=initial_guess_note),
    CampaignFeature('fold conditioning', lambda settings: bool(settings.get('binder_scaffold')), shorthand=lambda settings: binder_scaffold_name(settings).lower(), switch='binder_scaffold', configure=configure_fold_conditioning,
                    checks=(ModalityCheck('Scaffold_Sequence_Retained_Fraction', 0.0, True, 'min_scaffold_sequence_retained_final', parameters=scaffold_check_parameters),
                            ModalityCheck('Scaffold_Framework_RMSD', float('inf'), threshold_setting='max_scaffold_framework_rmsd_final', parameters=scaffold_check_parameters),
                            ModalityCheck('Framework_Packing_Fraction', 0.0, True, 'min_framework_packing_final'),
                            ModalityCheck('Off_Paratope_Contact_Fraction', 1.0, threshold_setting='max_off_paratope_contact_final')),
                    conflicts=(('cyclic peptide', 'a framework fixes the fold, so it cannot also be closed into a macrocycle'),
                               ('multi-chain binder', 'a scaffold defines one binder chain, so it cannot also be copied into an oligomer'),
                               ('fold switching', 'a framework fixes the fold, so it cannot also be told to switch fold'),
                               ('mixed topology', 'a framework fixes the fold, so it cannot also be told to change its secondary structure'))),
    CampaignFeature('given binder sequence', lambda settings: bool(settings.get('binder_sequences')), shorthand='seeded', switch='binder_sequences',
                    configure=configure_given_binder_sequences, note=given_binder_sequences_note,
                    checks=(ModalityCheck('Binder_Mutations', 0.0, True, parameters=mutation_check_parameters),)),  #recorded, never rejecting: how far a design moved is worth reading whether or not it is required
    CampaignFeature('MPNN redesign', lambda settings: bool(settings.get('mpnn_redesign')), shorthand='redesign', switch='mpnn_redesign', configure=configure_mpnn_redesign,
                    note=mpnn_redesign_note),
    CampaignFeature('forced targeting', lambda settings: bool(settings.get('forced_targeting')), shorthand='forced', switch='forced_targeting', configure=configure_focused_epitope,
                    note=lambda settings: f"forced targeting: everything outside {settings.get('forced_targeting_shell', 'the default')} Angstrom of the hotspots rebuilt as lysine"),
    CampaignFeature('multidomain binder', lambda settings: bool(settings.get('weights_multidomain')), configure=configure_multidomain,
                    note=lambda settings: f"multidomain, {settings.get('n_domains', 2)} domains of at least {settings.get('min_domain_size', 50)} residues",
                    checks=(ModalityCheck('Interdomain_Contact_Fraction', 1.0, threshold_setting='max_interdomain_contact_final', parameters=domain_check_parameters),
                            ModalityCheck('Domain_Separation_Ratio', 0.0, True, 'min_domain_separation_ratio_final', parameters=domain_check_parameters),
                            ModalityCheck('Binder_Chain_Breaks', float('inf'), threshold_setting='max_binder_chain_breaks_final'))),
    CampaignFeature('cyclic peptide', lambda settings: bool(settings.get('cyclize_peptide')), shorthand='cyclic', switch='cyclize_peptide',
                    checks=(ModalityCheck('Cyclic_Closure_Distance', float('inf'), threshold_setting='max_cyclic_closure_distance_final'),),
                    note=lambda settings: 'cyclic peptide closure'),
    CampaignFeature('peptide', designs_peptide, shorthand='peptide'),
    CampaignFeature('disordered target crops', lambda settings: bool(settings.get('idr_crop_count')), shorthand='idr'),
    CampaignFeature('disulfide staple', lambda settings: bool(settings.get('weights_disulfide')) or 'disulfide' in settings.get('losses', {}),
                    checks=(ModalityCheck('Binder_Disulfides', 0.0, True, 'min_binder_disulfides_final'),)),
    CampaignFeature('coldspot', lambda settings: any(state.coldspots for state in resolve_prepared_states(settings)), configure=configure_coldspots,
                    checks=(ModalityCheck('Coldspot_Contact_Fraction', 100.0, threshold_setting='max_coldspot_contact_final'),)),
    CampaignFeature('hotspot targeting', lambda settings: any(state.hotspots for state in resolve_prepared_states(settings)),
                    checks=(ModalityCheck('Hotspot_Contact_Fraction', 0.0, True, 'min_hotspot_contact_final'),
                            ModalityCheck('Off_Epitope_Contact_Fraction', 1.0, threshold_setting='max_off_epitope_contact_final', parameters=epitope_check_parameters),
                            ModalityCheck('Epitope_Residues_Contacted', 0.0, True, 'min_epitope_residues_contacted_final', parameters=epitope_check_parameters))),
    CampaignFeature('negative selection', lambda settings: any(state.objective == 'detarget' for state in resolve_prepared_states(settings)), configure=configure_negative_selection,
                    checks=(ModalityCheck('Interface_Residues_detarget', float('inf'), threshold_setting='max_detarget_interface_residues_final'),)),
    CampaignFeature('termini distance', lambda settings: bool(settings.get('weights_termini_distance')),
                    checks=(ModalityCheck('Termini_Distance', float('inf'), threshold_setting='max_termini_distance_final'),)),
    CampaignFeature('termini orientation', lambda settings: bool(settings.get('weights_termini_angle')),
                    checks=(ModalityCheck('Termini_Away_Cosine', -1.0, True, 'min_termini_away_cosine_final'),)),
    terminus_away_feature('n'),
    terminus_away_feature('c'),
    CampaignFeature('disordered target', has_fasta_target, configure=configure_disordered_target,
                    checks=(ModalityCheck('Target_Crop_Length', 0.0, True, 'min_target_crop_length_final'),),
                    conflicts=(('forced targeting', 'forced targeting rebuilds around a resolved backbone, which a sequence target does not carry'),
                               ('coldspot', 'coldspots are residue numbers a sequence target does not carry'))),
    CampaignFeature('humanization', lambda settings: bool(settings.get('weights_humanization')),
                    checks=(ModalityCheck('MHC_Anchor_Score', float('inf'), threshold_setting='max_mhc_anchor_score_final', parameters=humanization_check_parameters),)),
    CampaignFeature('protease resistance', lambda settings: bool(settings.get('weights_protease_sites')),
                    checks=(ModalityCheck('Protease_Site_Score', float('inf'), threshold_setting='max_protease_site_score_final'),)),
    CampaignFeature('buried loops', lambda settings: bool(settings.get('weights_exposed_loops')),
                    checks=(ModalityCheck('Exposed_Loop_Fraction', float('inf'), threshold_setting='max_exposed_loop_fraction_final'),)),
    CampaignFeature('buried termini', lambda settings: bool(settings.get('weights_exposed_termini')),
                    checks=(ModalityCheck('Terminus_Exposure', float('inf'), threshold_setting='max_terminus_exposure_final', parameters=terminus_exposure_check_parameters),)),
    CampaignFeature('mixed topology', lambda settings: bool(settings.get('weights_non_helical')) or settings.get('max_helix_fraction_final') is not None,
                    checks=(ModalityCheck('Binder_Helix_Fraction', 1.0, threshold_setting='max_helix_fraction_final'),)),
    CampaignFeature('induced fit', lambda settings: bool(settings.get('weights_induced_fit_interface')), configure=configure_induced_fit,
                    conflicts=(('negative selection', 'induced fit freezes one bound structure to compare a free state against, so it designs against a single target'),)),
    CampaignFeature('fold switching', lambda settings: bool(settings.get('weights_induced_fit_global')) or bool(settings.get('weights_fold_switching')), configure=configure_fold_switching),
    CampaignFeature('conformational switching', lambda settings: bool(settings.get('binder_shapes')),
                    note=lambda settings: 'two conformations compared: ' + ' against '.join(('/'.join(group) for group in settings['binder_shapes']))),
)

EVERY_CAMPAIGN_CHECKS = (ModalityCheck('Receptor_Chains_Contacted', 0.0, True, 'min_receptor_chains_contacted_final'),
                         ModalityCheck('Target_pLDDT', 0.0, True, 'min_target_plddt_final'))

def requested_campaign_features(settings: dict) -> tuple[CampaignFeature, ...]:
    return tuple(feature for feature in CAMPAIGN_FEATURES if feature.requested(settings))

def conflicting_campaign_features(settings: dict) -> tuple[tuple[str, str, str], ...]:
    requested = {feature.name for feature in requested_campaign_features(settings)}
    conflicts = {(*sorted((feature.name, other)), reason) for feature in CAMPAIGN_FEATURES if feature.name in requested for other, reason in feature.conflicts if other in requested}
    return tuple(sorted(conflicts))

def feature_shorthand(feature: CampaignFeature, settings: dict) -> str:
    return feature.shorthand(settings) if callable(feature.shorthand) else feature.shorthand

def install_modality_check(settings: dict, check: ModalityCheck) -> None:
    record_modality_filter(settings, check.metric, check.unrejecting_threshold, check.higher, check.threshold_setting, **(check.parameters(settings) if check.parameters else {}))

def add_design_property_records(settings: dict) -> None:
    install_modality_check(settings, ModalityCheck('Interface_BuriedArea', 0.0, True, 'min_interface_buried_area_final'))
    install_modality_check(settings, ModalityCheck('Interface_BuriedArea_Fraction', 0.0, True, 'min_interface_buried_area_fraction_final'))
    install_modality_check(settings, ModalityCheck('Surface_Hydrophobicity', float('inf'), threshold_setting='max_surface_hydrophobicity_final'))
    install_modality_check(settings, ModalityCheck('Interface_Hydrophobicity', float('inf'), threshold_setting='max_interface_hydrophobicity_final'))
    install_modality_check(settings, ModalityCheck('Target_RMSD', float('inf'), threshold_setting='max_target_rmsd_final'))
    install_modality_check(settings, ModalityCheck('SS_pLDDT', 0.0, True, 'min_ss_plddt_final'))
    install_modality_check(settings, ModalityCheck('All_Atom_Clashes', float('inf'), threshold_setting='max_all_atom_clashes_final'))
    install_modality_check(settings, ModalityCheck('Binder_Free_Cysteines', float('inf'), threshold_setting='max_binder_free_cysteines_final'))
    for name in ('Binder_Helix_Fraction', 'Binder_BetaSheet_Fraction', 'Binder_Loop_Fraction', 'Interface_Helix_Fraction', 'Interface_BetaSheet_Fraction', 'Interface_Loop_Fraction', 'Binder_Disulfides', 'Binder_Mass_kDa', 'Binder_pI', 'Binder_Net_Charge', 'Binder_Extinction', 'Binder_Cysteines'):
        install_modality_check(settings, ModalityCheck(name, float('inf')))

def record_conformational_change(settings: dict) -> None:
    if isinstance(settings['filters'], dict) and any((name in settings['filters'] for name in ('Induced_Fit_TM', 'Induced_Fit_Interface_RMSD'))):
        settings['filters'].setdefault('Induced_Fit_RMSD', {'threshold': float('inf'), 'higher': False})

def configure_campaign_features(settings: dict, request: ConfigurationRequest) -> None:
    for feature in requested_campaign_features(settings):
        if feature.configure is not None:
            feature.configure(settings, request)
    record_conformational_change(settings)
    for feature in requested_campaign_features(settings):
        for check in feature.checks:
            install_modality_check(settings, check)
    for check in EVERY_CAMPAIGN_CHECKS:
        install_modality_check(settings, check)
    add_design_property_records(settings)

CAMPAIGN_PRESETS = Path(__file__).parent.parent / 'settings'
PRESET_TIERS = 'modality', 'property', 'target'
BINDER_FORMATS = 'binder', 'large_binder', 'peptide', 'cyclic_peptide', 'homo_oligomer', 'multidomain', 'VHH', 'scFv', 'Fab', 'ARP'

def shipped_preset_names(tier: str, presets: Path=CAMPAIGN_PRESETS) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in (presets / tier).glob('*.json')))

def design_property_names() -> tuple[str, ...]:
    return shipped_preset_names('property')

def read_preset(tier: str, name: str, presets: Path=CAMPAIGN_PRESETS) -> dict:
    path = presets / tier / f'{name}.json'
    if not path.is_file():
        raise ValueError(f'unknown {tier} {name!r}; this package ships {", ".join(shipped_preset_names(tier, presets))}')
    preset = json.loads(path.read_text())
    preset.pop('description', None)
    if preset.get('binder_scaffold'):
        preset['binder_scaffold'] = str((presets.parent / preset['binder_scaffold']).resolve())
    for target in preset.get('targets', []):
        target['target_path'] = str((path.parent / target['target_path']).resolve())
    return preset

def requested_preset_names(request: dict, tier: str, presets: Path=CAMPAIGN_PRESETS) -> tuple[str, ...]:
    if tier == 'target':
        named = request.get('target') or ()
        return (named,) if isinstance(named, str) else tuple(named)
    if tier == 'modality':
        named = request.get('modality') or ()
        named = (named,) if isinstance(named, str) else tuple(named)
        return named if not named or any(name in BINDER_FORMATS for name in named) else ('binder', *named)
    return tuple(name for name in shipped_preset_names(tier, presets) if request.get(name))

def preset_tier_layers(request: dict, tier: str, presets: Path=CAMPAIGN_PRESETS) -> list[dict]:
    layers = [read_preset(tier, name, presets) for name in requested_preset_names(request, tier, presets)]
    accumulated = [target for layer in layers for target in layer.pop('targets', [])]
    return layers + [{'targets': accumulated}] if accumulated else layers

CORE_RESERVED_NAMES = frozenset({'default', 'reference'})

def requested_core_profiles(request: dict) -> tuple[str, ...]:
    named = request.get('core') or ()
    named = (named,) if isinstance(named, str) else tuple(named)
    return tuple(name for name in named if name not in CORE_RESERVED_NAMES)

def core_profile_layers(request: dict, presets: Path=CAMPAIGN_PRESETS) -> list[dict]:
    return [read_preset('core', name, presets) for name in requested_core_profiles(request)]

def campaign_over_presets(request: dict, presets: Path=CAMPAIGN_PRESETS) -> dict:
    layers = core_profile_layers(request, presets)
    for tier in PRESET_TIERS:
        layers += preset_tier_layers(request, tier, presets)
    settings = {}
    for layer in layers + [request]:
        settings = overridden_settings(settings, layer)
    return settings

def known_campaign_settings() -> frozenset[str]:
    return frozenset({'core', 'modality', 'paratope_conformations', 'target'} |set(design_property_names()) | CAMPAIGN_SETTING_NAMES | set(DEFAULT_SETTINGS) | set(FINAL_CONFIDENCE_FILTERS) | set(LOSS_PARAMETERS) | set(DOMAIN_PARAMETERS) | {f'weights_{name}' for name in REGISTERED_LOSSES} | {f'{stage}_steps' for stage in DESIGN_STAGE_NAMES if stage != 'final'} | {f'{filter}_{stage}' for filter in ('min_plddt', 'min_iptm', 'max_detarget_iptm') for stage in DESIGN_STAGE_NAMES} | {f'min_{terminus}_terminus_away_cosine_final' for terminus in ('n', 'c')})

def metric_parameter_names(metric_function) -> frozenset[str]:
    return frozenset(inspect.signature(metric_function).parameters) - {'protein_states', 'predictions'}

def unrecognized_setting(name: str, accepted, prefix: str='') -> str:
    nearest = difflib.get_close_matches(name, sorted(accepted), n=1, cutoff=0.6)
    return f'{prefix}{name}' + (f' (did you mean {nearest[0]}?)' if nearest else '')

FRACTION_THRESHOLD_SETTINGS = frozenset({'betasheet_reopt_trigger', 'max_coldspot_contact_final', 'max_exposed_loop_fraction_final', 'max_helix_fraction_final', 'max_interdomain_contact_final', 'max_off_epitope_contact_final', 'max_off_paratope_contact_final', 'max_surface_hydrophobicity_final', 'min_framework_packing_final', 'min_hotspot_contact_final', 'min_scaffold_sequence_retained_final'})

def written_as_percentage(value) -> bool:
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(float(value)) and float(value) > 1

def reject_percentage_thresholds(overrides: dict) -> None:
    written = [(name, overrides[name]) for name in sorted(FRACTION_THRESHOLD_SETTINGS) if written_as_percentage(overrides.get(name))]
    configured = overrides.get('filters')
    for name, entry in configured.items() if isinstance(configured, dict) else ():
        if name.endswith('_Fraction') and isinstance(entry, dict) and written_as_percentage(entry.get('threshold')):
            written.append((f'filters.{name}.threshold', entry['threshold']))
    if written:
        raise ValueError('these checks read a fraction of 0 to 1 rather than a percentage: ' + ', '.join(f'{name} {float(value):g} (write {float(value) / 100:g})' for name, value in written))

def reject_unrecognized_settings(overrides: dict) -> None:
    accepted = known_campaign_settings()
    rejected = [unrecognized_setting(name, accepted) for name in overrides if name not in accepted]
    for target in overrides.get('targets') or ():
        rejected += [unrecognized_setting(name, TARGET_SETTING_NAMES, 'targets[].') for name in target if name not in TARGET_SETTING_NAMES]
    for block, registry, entry_names in (('losses', REGISTERED_LOSSES, METRIC_ENTRY_NAMES), ('filters', REGISTERED_FILTER_METRICS, FILTER_ENTRY_NAMES)):
        configured = overrides.get(block)
        for name, entry in configured.items() if isinstance(configured, dict) else ():
            registered_name = name
            if registered_name not in registry:
                rejected.append(unrecognized_setting(name, set(registry), f'{block}.'))
            elif isinstance(entry, dict):
                accepted_parameters = metric_parameter_names(registry[registered_name])
                rejected += [unrecognized_setting(key, entry_names, f'{block}.{name}.') for key in entry if key not in entry_names]
                rejected += [unrecognized_setting(key, accepted_parameters, f'{block}.{name}.params.') for key in entry.get('params') or () if key not in accepted_parameters]
    if rejected:
        raise ValueError('unrecognized campaign settings: ' + ', '.join(rejected))

def load_settings(overrides: dict | None=None) -> dict:
    overrides = campaign_over_presets(copy.deepcopy(overrides or {}))
    reject_unrecognized_settings(overrides)
    reject_percentage_thresholds(overrides)
    configured_losses = overrides.pop('losses', {})
    loss_settings = {}
    for name, value in configured_losses.items():
        if isinstance(value, dict):
            loss_settings[name] = value
        else:
            overrides[f'weights_{name}'] = value
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings.update({key: value for key, value in overrides.items() if key != 'filters'})
    settings['losses'] = copy.deepcopy(DEFAULT_LOSSES)
    for name, entry in loss_settings.items():
        settings['losses'][name] = {**settings['losses'].get(name, {}), **entry}
    for parameter, (name, argument) in {**LOSS_PARAMETERS, **{name: ('multidomain', name) for name in DOMAIN_PARAMETERS}}.items():
        if parameter in settings:
            settings['losses'].setdefault(name, {}).setdefault('params', {}).setdefault(argument, settings[parameter])
    if isinstance(overrides.get('filters'), dict):
        for name, entry in overrides['filters'].items():
            settings['filters'][name] = {**settings['filters'].get(name, {}), **entry}
    elif 'filters' in overrides:
        settings['filters'] = overrides['filters']
    if settings.get('sparse_output'):
        for name, value in {'save_failed_refolds': False, 'save_failed_trajectories': False, 'save_binder_monomers': False, 'save_design_animations': False, 'save_design_frames': False, 'save_design_sequences': False, 'save_design_trajectory': False, 'save_loss_plots': False}.items():
            if name not in overrides:
                settings[name] = value
    configure_campaign_features(settings, ConfigurationRequest(overrides, loss_settings))
    for setting_name, filter_name in FINAL_CONFIDENCE_FILTERS.items():
        if setting_name in overrides and isinstance(settings['filters'], dict) and (filter_name not in overrides.get('filters', {})):
            settings['filters'][filter_name]['threshold'] = float(overrides[setting_name])
    return settings

def resolved_campaign_facts(settings: dict) -> dict:
    from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL, campaign_length_bucket
    selected_models = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)
    return {'campaign_seed': int(settings.get('campaign_seed') or 0),
            'design_models': list(selected_models.design_models),
            'validation_models': list(selected_models.validation_models),
            'validation_model': resolve_validation_model(settings),
            'design_stage_steps': design_stage_rounds(settings),
            'binder_chains': list(resolve_binder_chains(int(settings.get('copies', 1)), settings.get('binder_scaffold'))),
            'oligomer_tie': resolve_oligomer_tie(settings),
            'cyclic_offset_mode': resolve_cyclic_offset_mode(settings),
            'length_bucket_size': campaign_length_bucket(settings)}

@dataclass
class TargetSettings:
    name: str
    path: str
    chains: str = ''
    hotspots: str = ''
    weight: float = 1.0
    coldspots: str = ''

def campaign_binder_lengths(binder_lengths) -> tuple[int, ...] | None:
    if binder_lengths is None:
        return None
    binder_lengths = tuple(int(length) for length in binder_lengths)
    return tuple(range(binder_lengths[0], binder_lengths[1] + 1)) if len(binder_lengths) == 2 else binder_lengths

def resolve_binder_lengths(binder_lengths) -> tuple[int, ...] | None:
    share = os.environ.get('BINDCRAFT_BINDER_LENGTHS')
    if share:
        return tuple(int(length) for length in share.split(','))
    return campaign_binder_lengths(binder_lengths)

def resolve_binder_sequences(settings: dict) -> dict[str, str]:
    binder_sequences = settings.get('binder_sequences') or {}
    if not isinstance(binder_sequences, dict):
        raise ValueError('binder_sequences is written {"name": "SEQUENCE"}, naming every binder the redesign starts from')
    resolved = {}
    for name, sequence in binder_sequences.items():
        residues = ''.join(str(sequence).split()).upper()
        unknown = ''.join(sorted(set(residues) - set(AMINO_ACIDS)))
        if not residues or unknown:
            raise ValueError(f'binder_sequences[{name!r}] must be a non-empty one-letter amino acid sequence; got {unknown or "nothing"}')
        resolved[name] = residues
    return resolved

@dataclass
class BinderSettings:
    lengths: tuple[int, ...] | None = None
    copies: int = 1
    scaffold: str | None = None
    scaffold_edits: str = ''
    sequences: dict[str, str] = field(default_factory=dict)
    amino_acid_bias: dict[str, float] = field(default_factory=dict)
    omitted_amino_acids: str = ''

@dataclass
class PredictionModelSelection:
    design_models: tuple[str, ...]
    validation_models: tuple[str, ...]
    validation_pool_exhausted: bool = False

def select_prediction_models(requested_models: int | tuple[str, ...], available_models: tuple[str, ...], campaign_seed: int=0) -> tuple[str, ...]:
    if isinstance(requested_models, int):
        if not 1 <= requested_models <= len(available_models):
            raise ValueError(f'Requested {requested_models} models from a pool of {len(available_models)}')
        return tuple(sorted(random.Random(campaign_seed).sample(available_models, requested_models)))
    unavailable_models = tuple(model for model in requested_models if model not in available_models)
    if unavailable_models:
        raise ValueError(f'Requested models {unavailable_models} outside the pool {available_models}')
    return tuple(requested_models)

def resolve_model_pool_selection(requested_models, pool: tuple[str, ...]):
    if requested_models is None or isinstance(requested_models, int):
        return requested_models
    chosen = []
    for model in requested_models:
        if isinstance(model, int) or (isinstance(model, str) and model.lstrip('-').isdigit()):
            index = int(model)
            if not 0 <= index < len(pool):
                raise ValueError(f'model index {index} is outside the {len(pool)}-model pool {pool}')
            chosen.append(pool[index])
        else:
            chosen.append(model)
    return tuple(dict.fromkeys(chosen))

def named_prediction_models(requested_models: int | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    return () if requested_models is None or isinstance(requested_models, int) else tuple(dict.fromkeys(requested_models))

def requested_validation_model_count(settings: dict) -> int:
    requested_models = settings.get('validation_models', DEFAULT_VALIDATION_MODEL_COUNT)
    return int(requested_models) if isinstance(requested_models, int) else len(named_prediction_models(requested_models))

def default_design_model_count(settings: dict, multimer_pool: tuple[str, ...]) -> int:
    if validates_on_multimer(settings):
        return len(multimer_pool) - requested_validation_model_count(settings)
    return len(multimer_pool)

def select_design_and_validation_models(settings: dict, multimer_pool: tuple[str, ...], monomer_pool: tuple[str, ...]) -> PredictionModelSelection:
    campaign_seed = int(settings.get('campaign_seed') or 0)
    requested_validation_models = resolve_model_pool_selection(settings.get('validation_models'), validation_model_pool(settings, multimer_pool, monomer_pool))
    named_validation_models = named_prediction_models(requested_validation_models)
    requested_design_models = resolve_model_pool_selection(settings.get('design_models', default_design_model_count(settings, multimer_pool)), multimer_pool)
    design_pool = tuple(model for model in multimer_pool if model not in named_validation_models)
    if named_validation_models and isinstance(requested_design_models, int) and requested_design_models > len(design_pool):
        raise ValueError(f'{requested_design_models} design models asked for, but validation named {len(named_validation_models)} of the {len(multimer_pool)}-model pool and design cannot reuse them, which leaves {len(design_pool)} to design on')
    design_models = select_prediction_models(requested_design_models, design_pool if isinstance(requested_design_models, int) else multimer_pool, campaign_seed)
    held_out_pool = tuple(model for model in validation_model_pool(settings, multimer_pool, monomer_pool) if model not in design_models)
    overlapping_models = tuple(model for model in named_validation_models if model in design_models)
    if overlapping_models:
        raise ValueError(f'Requested validation models {overlapping_models} are also design models; validation must be held out from design')
    if requested_validation_models is None:
        requested_validation_models = len(held_out_pool) or DEFAULT_VALIDATION_MODEL_COUNT
    validation_pool_exhausted = isinstance(requested_validation_models, int) and requested_validation_models > len(held_out_pool)
    return PredictionModelSelection(design_models, select_prediction_models(requested_validation_models, monomer_pool if validation_pool_exhausted else held_out_pool, campaign_seed), validation_pool_exhausted)

VALIDATION_MODEL_POOLS = ('monomer', 'multimer')

def resolve_validation_model(settings: dict) -> str:
    requested = settings.get('validation_model')
    if requested is None:
        return 'multimer' if settings.get('binder_scaffold') or int(settings.get('copies', 1)) > 1 else 'monomer'
    if requested not in VALIDATION_MODEL_POOLS:
        raise ValueError(f'unknown validation_model {requested!r}; use one of {", ".join(VALIDATION_MODEL_POOLS)}')
    return requested

def validates_on_multimer(settings: dict) -> bool:
    return resolve_validation_model(settings) == 'multimer'

def validation_model_pool(settings: dict, multimer_pool: tuple[str, ...], monomer_pool: tuple[str, ...]) -> tuple[str, ...]:
    return multimer_pool if validates_on_multimer(settings) else monomer_pool

DESIGN_STAGE_DEFAULT_ROUNDS = {'screen': 50, 'refine': 25, 'anneal': 45, 'harden': 5, 'mutate': 15}
MERGED_GRADIENT_STAGE = 'anneal'

def rounds_per_binding_target(settings: dict) -> int:
    return 2 if len([target for target in settings.get('targets', []) if target.get('objective', 'target') != 'detarget']) > 1 else 1

def design_stage_rounds(settings: dict) -> dict[str, int]:
    budget = rounds_per_binding_target(settings)
    return {design_stage: int(settings.get(f'{design_stage}_steps', default_rounds * budget)) for design_stage, default_rounds in DESIGN_STAGE_DEFAULT_ROUNDS.items()}

def gradient_stage_rounds(settings: dict) -> dict[str, int]:
    return {design_stage: rounds for design_stage, rounds in design_stage_rounds(settings).items() if design_stage != 'mutate'}

def design_model_count(settings: dict) -> int:
    requested_design_models = settings.get('design_models') or default_design_model_count(settings, MULTIMER_MODEL_POOL)
    return int(requested_design_models) if isinstance(requested_design_models, int) else len(named_prediction_models(requested_design_models))

def target_objective(target: TargetSettings) -> str:
    return 'detarget' if target.weight < 0 else 'target'

@dataclass
class BinderDesignSettings:
    targets: list[TargetSettings]
    binder: BinderSettings
    settings: dict
    filters: dict[str, DesignFilter] = field(default_factory=dict)
    seed: int = 0
    binder_shapes: tuple[tuple[str, ...], ...] = ()
    prepared_states: tuple[PreparedTargetState, ...] = ()
    binder_chains: tuple[str, ...] = ()
    designed_binder_chain: str = ''
    target_chain_prefix: str = 'target'
    oligomer_tie: str = ''

    def __post_init__(self) -> None:
        self.target_chain_prefix = self.settings.get('target_chain', self.target_chain_prefix)
        self.binder_chains = self.binder_chains or resolve_binder_chains(self.binder.copies, self.binder.scaffold)
        self.designed_binder_chain = self.designed_binder_chain or resolve_designed_binder_chain(self.settings, self.binder_chains)
        self.oligomer_tie = self.oligomer_tie or resolve_oligomer_tie(self.settings)
        self.prepared_states = self.prepared_states or tuple(PreparedTargetState(target.name, target.name, target.path, target_objective(target), target.weight, target_chain_name(self.target_chain_prefix, target.name), target.chains, target.hotspots, target.coldspots, 0, resolve_crop_length_bounds(self.settings) if is_fasta(target.path) else None) for target in self.targets)

def build_design_settings(settings: dict) -> BinderDesignSettings:
    prepared_states = resolve_prepared_states(settings)
    for state in prepared_states:
        if state.path is None:
            raise ValueError(f'Target {state.source_target!r} needs target_path or structure')
    targets = [TargetSettings(state.name, state.path, state.chains, state.hotspots, state.weight, state.coldspots) for state in prepared_states]
    binder_lengths = resolve_binder_lengths(settings.get('binder_lengths'))
    binder = BinderSettings(lengths=binder_lengths, copies=settings.get('copies', 1), scaffold=settings.get('binder_scaffold'), scaffold_edits=settings.get('mutate_positions', ''), sequences=resolve_binder_sequences(settings), amino_acid_bias=resolve_amino_acid_bias(settings), omitted_amino_acids=resolve_omitted_amino_acids(settings))
    filter_settings = settings.get('filters')
    filters = build_filters(filter_settings, resolve_designed_binder_chain(settings)) if isinstance(filter_settings, dict) else {}
    return BinderDesignSettings(targets, binder, settings, filters, settings.get('campaign_seed') or 0, tuple(tuple(group) for group in settings.get('binder_shapes', [])), prepared_states)

def target_state_names(design_settings: BinderDesignSettings) -> tuple[str, ...]:
    return tuple(state.name for state in design_settings.prepared_states if state.objective != 'detarget')

def detarget_state_names(design_settings: BinderDesignSettings) -> tuple[str, ...]:
    return tuple(state.name for state in design_settings.prepared_states if state.objective == 'detarget')

def merged_gradient_targets(design_settings: BinderDesignSettings, design_stage: str=MERGED_GRADIENT_STAGE) -> int:
    prepared_state_count = len(design_settings.prepared_states)
    return prepared_state_count if design_stage == MERGED_GRADIENT_STAGE and prepared_state_count > 1 and design_settings.settings.get('multitarget_merged_gradients', True) else 1

def merged_gradient_sequence_updates(design_settings: BinderDesignSettings) -> dict[str, int]:
    stage_rounds = gradient_stage_rounds(design_settings.settings)
    rotating_stages = design_settings.settings.get('multitarget_rounds_per_target') or ()
    if len(design_settings.prepared_states) > 1:
        stage_rounds = {design_stage: rounds * len(design_settings.prepared_states) if design_stage in rotating_stages else rounds for design_stage, rounds in stage_rounds.items()}
    rounds_per_sequence_update = merged_gradient_targets(design_settings) if design_settings.settings.get('multitarget_merged_gradient_budget', 'sequence_updates') == 'model_calls' else 1
    if rounds_per_sequence_update < 2:
        return stage_rounds
    return {design_stage: max(1, rounds // rounds_per_sequence_update) if rounds and design_stage == MERGED_GRADIENT_STAGE else rounds for design_stage, rounds in stage_rounds.items()}

def parse_setting_overrides(assignments) -> dict:
    overrides: dict = {}
    for assignment in assignments:
        name, separator, value = assignment.partition('=')
        if not separator or not name:
            raise ValueError(f'a setting override is written KEY=VALUE, as in binder_lengths=[60,80] or filters.i_pTM.threshold=0.8; got {assignment!r}')
        block = overrides
        *blocks, setting = name.split('.')
        for block_name in blocks:
            block = block.setdefault(block_name, {})
        try:
            block[setting] = json.loads(value)
        except json.JSONDecodeError:
            block[setting] = value
    return overrides

def overridden_settings(request: dict, overrides: dict) -> dict:
    merged = dict(request)
    for name, value in overrides.items():
        merged[name] = overridden_settings(merged[name], value) if isinstance(value, dict) and isinstance(merged.get(name), dict) else value
    return merged

def read_campaign_metadata(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    fields = json.loads(Path(path).resolve().read_text())
    if not isinstance(fields, dict):
        raise ValueError(f'metadata: {path} holds a JSON {type(fields).__name__}, where metadata is an object of fields to stamp, such as {{"author": "your name"}}')
    return {f'meta_{name}': value if isinstance(value, str) else json.dumps(value) for name, value in fields.items()}

def read_settings(path: str | Path, overrides: dict | None=None) -> dict:
    settings_path = Path(path).resolve()
    request = json.loads(settings_path.read_text())
    for target in request.get('targets', []):
        target['target_path'] = str((settings_path.parent / target['target_path']).resolve())
    if request.get('binder_scaffold'):
        request['binder_scaffold'] = str((settings_path.parent / request['binder_scaffold']).resolve())
    return load_settings(overridden_settings(request, overrides or {}))
