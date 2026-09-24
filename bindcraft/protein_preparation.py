import functools
import random
import re
import jax
import jax.numpy as jnp
from dataclasses import replace
from jax import Array
from bindcraft.protein import AMINO_ACIDS, Protein, ProteinStates, ResidueFlags, has_residue_flag, parse_residue_length_choices, selected_chain_names, structure_chain_labels, structure_chain_names, structure_source_description, superposed_on_reference_target, target_chain_name, target_fit_quality
from bindcraft.af2 import campaign_length_bucket, padded_prediction_length
from bindcraft.design_identity import target_conformation_fingerprint
from bindcraft.loss import DesignLoss, build_design_losses, induced_fit_binder_alone_losses, induced_fit_hinge_names, sampled_loss_weights
from bindcraft.sequence_optimization import OMITTED_AMINO_ACID_LOGIT
from bindcraft.settings import BinderDesignSettings, build_design_settings, resolve_validation_crop_flank, DEFAULT_IDR_CROP_LENGTHS, is_fasta

RECEPTOR_CHAIN_BREAK_GAP = 49

def chain_qualified_residue_span(span: str, receptor_chain_names: list[str]) -> str:
    span = span.strip()
    return span if span[0].isalpha() else f'{receptor_chain_names[0]}{span}'

def target_binding_site_flags(receptor_chain_names: list[str], hotspots: str, coldspots: str='') -> str:
    return ','.join(f'{chain_qualified_residue_span(span, receptor_chain_names)}+{residue_flag}' for residue_flag, spans in (('HOTSPOT', hotspots), ('COLDSPOT', coldspots)) for span in spans.split(',') if span.strip())

@functools.cache
def receptor_chain_layout(path: str, chains: str) -> tuple[tuple[str, int, int], ...]:
    """Each receptor chain of a target, with the residues it holds and the number its own structure starts it at.

    merge_receptor_chains fuses them into the one chain the losses and the filters address, and this is
    what an output needs to write them apart again under the numbering the reader asked its hotspots in."""
    receptor_chain_names = selected_chain_names(chains, structure_chain_labels(path)) or structure_chain_names(path)
    if len(receptor_chain_names) < 2:
        return ()
    receptor_chains = Protein.from_structure(path, chains=','.join(receptor_chain_names))
    return tuple((name, len(receptor_chains[name]), int(receptor_chains[name].residue_index[0])) for name in receptor_chain_names if name in receptor_chains)

def receptor_chain_layouts(design_settings: BinderDesignSettings) -> dict[str, tuple[tuple[str, int, int], ...]]:
    """Every fused target chain of a campaign, keyed by the chain name its complexes carry it under."""
    return {state.target_chain: layout for state in design_settings.prepared_states
            if state.path and (not is_fasta(state.path)) and (layout := receptor_chain_layout(state.path, state.chains))}

def merge_receptor_chains(receptor_chains: list[Protein]) -> Protein:
    residue_index = [receptor_chains[0].residue_index]
    for receptor_chain in receptor_chains[1:]:
        residue_index.append(receptor_chain.residue_index - receptor_chain.residue_index[0] + residue_index[-1][-1] + RECEPTOR_CHAIN_BREAK_GAP)
    return receptor_chains[0].replace(residue_index=jnp.concatenate(residue_index), **{name: jnp.concatenate([getattr(receptor_chain, name) for receptor_chain in receptor_chains]) for name in ('sequence', 'atoms', 'atom_mask', 'flags')})

def frame_holding_target(design_settings: BinderDesignSettings) -> str:
    return next((state.name for state in design_settings.prepared_states if state.objective != 'detarget'), design_settings.prepared_states[0].name if design_settings.prepared_states else '')

def targets_in_one_frame(targets: dict[str, Protein], frame_holder: str) -> dict[str, Protein]:
    if frame_holder not in targets:
        return targets
    moved = {name: None if name == frame_holder else superposed_on_reference_target(target, targets[frame_holder]) for name, target in targets.items()}
    return {name: moved[name] or target for name, target in targets.items()}

def target_frame_report(targets: dict[str, Protein], frame_holder: str) -> str:
    said = []
    for name, target in targets.items():
        if name == frame_holder:
            continue
        quality = target_fit_quality(target, targets[frame_holder])
        held = 'no coordinates to fit' if quality is None else f'holds {quality[0]:.2f} of the shorter at {quality[1]:.2f} identity'
        said.append(f'{name} {"on " + frame_holder if superposed_on_reference_target(target, targets[frame_holder]) is not None else "on its binder"} ({held})')
    return f'campaign frames: {frame_holder} holds its own | ' + ' | '.join(said)

def states_holding_the_frame(protein_states: ProteinStates, frame_holder: str, target_chain_prefix: str) -> set[str]:
    holder = protein_states.get(frame_holder, {}).get(target_chain_name(target_chain_prefix, frame_holder))
    if holder is None:
        return set(protein_states)
    return {frame_holder} | {name for name, protein_complex in protein_states.items() if name != frame_holder and (target := protein_complex.get(target_chain_name(target_chain_prefix, name))) is not None and superposed_on_reference_target(target, holder) is not None}

def flanked_epitope_window(epitope_start: int, epitope_length: int, flank: int, target_sequence_length: int) -> tuple[int, int]:
    flanked_start = max(0, epitope_start - flank)
    return flanked_start, min(target_sequence_length, epitope_start + epitope_length + flank) - flanked_start

def prepare_targets(design_settings: BinderDesignSettings, seed: int | None=None, longest_crop: bool=False, validation_crop: bool=False) -> dict[str, Protein]:
    targets, sampled_target_epitopes = {}, {}
    crop_flank = resolve_validation_crop_flank(design_settings.settings)
    random_generator = random.Random(design_settings.seed if seed is None else seed)
    for target in design_settings.targets:
        if is_fasta(target.path):
            target_protein = Protein.from_fasta(target.path, target.chains or 'A', target_binding_site_flags([target.chains or 'A'], target.hotspots, target.coldspots))
            crop_lengths = design_settings.settings.get('crop_fasta_sequence', DEFAULT_IDR_CROP_LENGTHS)
            if crop_lengths is not None and crop_lengths is not False:
                crop_length_bounds = (crop_lengths, crop_lengths) if isinstance(crop_lengths, int) and (not isinstance(crop_lengths, bool)) else crop_lengths
                if not isinstance(crop_length_bounds, (tuple, list)) or len(crop_length_bounds) != 2 or min(crop_length_bounds) < 1:
                    raise ValueError('crop_fasta_sequence must be a positive length, two-value range, or false')
                target_sequence_length = len(target_protein)
                epitope_lengths = sorted({min(length, target_sequence_length) for length in range(min(crop_length_bounds), max(crop_length_bounds) + 1)})
                epitope_windows = [(start, length) for length in epitope_lengths for start in range(target_sequence_length - length + 1)]
                if longest_crop:
                    epitope_start, epitope_length = 0, min(target_sequence_length, max(epitope_lengths) + 2 * crop_flank)
                else:
                    sampled_epitopes = sampled_target_epitopes.setdefault(target.path, set())
                    unsampled_epitopes = [window for window in epitope_windows if window not in sampled_epitopes]
                    if not unsampled_epitopes:
                        raise ValueError(f'crop_fasta_sequence yields only {len(epitope_windows)} distinct crop(s) of a {target_sequence_length}-residue target; idr_crop_count is larger')
                    epitope_start, epitope_length = random_generator.choice(unsampled_epitopes)
                    sampled_epitopes.add((epitope_start, epitope_length))
                    if validation_crop:
                        epitope_start, epitope_length = flanked_epitope_window(epitope_start, epitope_length, crop_flank, target_sequence_length)
                target_protein = target_protein.replace(**{name: getattr(target_protein, name)[epitope_start:epitope_start + epitope_length] for name in ('sequence', 'atoms', 'atom_mask', 'flags', 'residue_index')})
                if not longest_crop and not validation_crop:
                    print(f'target={target.name} crop={epitope_start + 1}-{epitope_start + epitope_length}/{target_sequence_length}', flush=True)
            targets[target.name] = target_protein
        else:
            receptor_chain_names = selected_chain_names(target.chains, structure_chain_labels(target.path)) or structure_chain_names(target.path)
            receptor_chains = Protein.from_structure(target.path, chains=','.join(receptor_chain_names), flags=target_binding_site_flags(receptor_chain_names, target.hotspots, target.coldspots))
            missing_chains = [chain_name for chain_name in receptor_chain_names if chain_name not in receptor_chains]
            if missing_chains:
                raise ValueError(f'Target {target.name!r} has no chain {missing_chains} in {structure_source_description(target.path)}; the structure has {structure_chain_names(target.path)}. Write several chains as "A,B".')
            targets[target.name] = merge_receptor_chains([receptor_chains[chain_name] for chain_name in receptor_chain_names])
            if target.coldspots and not validation_crop:
                print(f'target={target.name} coldspots={target.coldspots} residues={int(has_residue_flag(targets[target.name].flags, ResidueFlags.COLDSPOT).sum())} hotspots={target.hotspots or "none"}', flush=True)
    return targets if len(targets) < 2 else targets_in_one_frame(targets, frame_holding_target(design_settings))

def crops_a_disordered_target(design_settings: BinderDesignSettings) -> bool:
    return any(state.crop_length_bounds is not None for state in design_settings.prepared_states)

def validation_target_states(design_settings: BinderDesignSettings, target_states: ProteinStates, trajectory_seed: int) -> ProteinStates:
    if not crops_a_disordered_target(design_settings) or not resolve_validation_crop_flank(design_settings.settings):
        return target_states
    flanked_targets = prepare_targets(design_settings, trajectory_seed, validation_crop=True)
    target_chains = {state.name: state.target_chain for state in design_settings.prepared_states}
    return {name: {chain: flanked_targets[name] if chain == target_chains.get(name) and name in flanked_targets else protein for chain, protein in protein_complex.items()} for name, protein_complex in target_states.items()}

def sampled_binder_length(binder_lengths: tuple[int, ...] | None, length_random_key: Array) -> int:
    return 0 if binder_lengths is None else int(jax.random.choice(length_random_key, jnp.asarray(binder_lengths)))

def prepare_binder_chains(design_settings: BinderDesignSettings, key: Array) -> dict[str, Protein]:
    #one key for every chain, for multi-chain binders
    length_random_key, binder_random_key = jax.random.split(key)
    binder_lengths = design_settings.binder.lengths
    binder_length = sampled_binder_length(binder_lengths, length_random_key)
    if design_settings.binder.scaffold:
        scaffold_chains = structure_chain_names(design_settings.binder.scaffold)
        scaffold = Protein.from_structure(design_settings.binder.scaffold, chains=','.join(scaffold_chains))
        seeded_scaffold_chains = [scaffold_chains[index % len(scaffold_chains)] for index in range(len(design_settings.binder_chains))]
        binder = {chain_name: scaffold[scaffold_chain].with_scaffold(scaffold_chain, design_settings.binder.scaffold_edits, binder_random_key) for chain_name, scaffold_chain in zip(design_settings.binder_chains, seeded_scaffold_chains)}
    else:
        if binder_length < 1:
            raise ValueError('Binder length must be positive')
        binder = {chain_name: Protein.empty(binder_length, binder_random_key) for chain_name in design_settings.binder_chains}
    for chain_name, protein in binder.items():
        #an omitted residue is written into the logits because omitted_amino_acid_mask reads them back out of it;
        #aa_bias is not, it is a standing prior the design model applies every round rather than a head start the
        #first rounds spend, so one propensity means one thing however many rounds follow it
        omitted_sequence = protein.sequence
        for amino_acid in design_settings.binder.omitted_amino_acids:
            omitted_sequence = omitted_sequence.at[:, AMINO_ACIDS.index(amino_acid)].set(OMITTED_AMINO_ACID_LOGIT)
        #designed residues only, to hold a fold conditioning scaffold's framework
        sequence = jnp.where(has_residue_flag(protein.flags, ResidueFlags.DESIGN)[:, None], omitted_sequence, protein.sequence)
        binder[chain_name] = protein.replace(sequence=sequence, flags=protein.flags | int(ResidueFlags.CYCLIC) if design_settings.settings.get('cyclize_peptide') else protein.flags)
    return binder

def design_residue_count(settings: dict) -> int:
    design_settings = build_design_settings(settings)
    bucket_size = campaign_length_bucket(settings)
    #longest_crop: for disordered targets only
    targets = prepare_targets(design_settings, longest_crop=True)
    if design_settings.binder.scaffold:
        longest_edits = re.sub(r'\(([^)]+)\)', lambda match: f'({max(parse_residue_length_choices(match.group(1)))})', design_settings.binder.scaffold_edits)
        design_settings = replace(design_settings, binder=replace(design_settings.binder, scaffold_edits=longest_edits))
        binder_length = sum(padded_prediction_length(len(protein), bucket_size) for protein in prepare_binder_chains(design_settings, jax.random.PRNGKey(design_settings.seed)).values())
    else:
        binder_length = padded_prediction_length(max(design_settings.binder.lengths), bucket_size) * design_settings.binder.copies
    return padded_prediction_length(max((len(target) for target in targets.values()), default=0) + binder_length, bucket_size)

def sampled_trajectory_values(design_settings: BinderDesignSettings, key: Array) -> tuple[dict[str, int | float | str], dict[str, Protein]]:
    binder_initialization_key, conformation_random_key = jax.random.split(jax.random.split(key)[0])
    trajectory_seed = int(jax.random.randint(conformation_random_key, (), 0, 2 ** 30))
    binder_chains = prepare_binder_chains(design_settings, binder_initialization_key)
    drawn = {'binder_length': sum(len(protein) for protein in binder_chains.values()), 'trajectory_seed': trajectory_seed, **sampled_loss_weights(design_settings.settings, trajectory_seed)}
    if design_settings.binder.scaffold:
        drawn['conformation.binder_scaffold'] = ''.join(target_conformation_fingerprint(protein) for protein in binder_chains.values())
    return (drawn, prepare_targets(design_settings, trajectory_seed))

def initialize_design_trajectory(design_settings: BinderDesignSettings, key: Array, targets: dict[str, Protein] | None=None) -> tuple[ProteinStates, tuple[tuple[str, ...], ...], dict[str, DesignLoss]]:
    binder_initialization_key, conformation_random_key = jax.random.split(key)
    trajectory_seed = int(jax.random.randint(conformation_random_key, (), 0, 2 ** 30))
    targets = prepare_targets(design_settings, trajectory_seed) if targets is None else targets
    binder = prepare_binder_chains(design_settings, binder_initialization_key)
    multi_chain_binders = (design_settings.binder_chains,) if design_settings.binder.copies > 1 else ()  #for oligomers
    target_chain = design_settings.settings.get('target_chain', 'target')
    protein_states: ProteinStates = {name: {**binder, target_chain_name(target_chain, name): protein} for name, protein in targets.items()} or {'seed': dict(binder)}
    #bucket length, for multi-domain domain splits
    bucket_length = padded_prediction_length(len(next(iter(binder.values()))), campaign_length_bucket(design_settings.settings))
    losses = build_design_losses(design_settings.settings, {target.name: target.weight for target in design_settings.targets}, bucket_length, trajectory_seed)
    if induced_fit_hinge_names(losses):
        if len(targets) != 1:
            raise ValueError('Induced-fit design requires exactly one target')
        losses.update(induced_fit_binder_alone_losses(losses))
    return protein_states, multi_chain_binders, losses
