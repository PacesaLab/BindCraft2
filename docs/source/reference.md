# BC2 settings reference

[Installation and running](installation.md) · [Outputs and Measurements](outputs.md) · [Examples](examples.md)

[Inputs](#input-tiers-and-overrides) · [Every setting](#every-setting-at-its-default) · [Targets and scaffolds](#define-the-target-and-binder) · [Stages](#design-stages-and-acceptance) · [Models](#models-and-sequence-redesign) · [Biological options](#biological-options) · [Losses](#losses) · [Filters](#filters) · [Autotuning](#autotuning-and-parameter-sweeps) · [Files and resources](#output-and-execution-settings)

Start with a preset and change only the controls your experiment needs. Tables give the exact JSON names, defaults and reasons to change them. **Defaults can change with the selected modality or property**; `campaign_metadata.json` records what a run actually used. “Unset” means no explicit choice, not zero.

## Input tiers and overrides

```bash
bindcraft design examples/pdl1.json --modality VHH --humanize --set 'project_folder=results/pdl1_vhh'
bindcraft design --help
```

| Applied in order | What you supply | Purpose |
| --- | --- | --- |
| Core | Nothing | The baseline every campaign starts from, always loaded, in `settings/core/default.json` in your BindCraft2 repo. |
| Core profile | `"core": "benchmark"`, or `--core benchmark` | An opt-in profile beside the baseline, from the `settings/core/` directory in your BindCraft2 repo. |
| Modality | `"modality": "binder"` or a list such as `["VHH", "induced_fit"]` | Binder format and conformational objective; see "Design modalities" in your BindCraft2 repo's top-level README.md. |
| Properties | Top-level booleans such as `"humanize": true` | Optional biological properties and starting conformations. |
| Target | `"target": "hPDL1"` or `"target": ["hPDL1", "mPDL1"]` | Shipped structures and binding-site selections from the `settings/target/` directory in your BindCraft2 repo. |
| Campaign | Your other JSON entries | Requested designs, output location and explicit adjustments. |

Each layer overrides the ones above it in the table. A target therefore wins over the modality and property defaults it is combined with, the campaign file wins over every preset, and `settings/core/default.json` in your BindCraft2 repo is the floor under all of them. Command-line choices override matching entries in the campaign file. Nested objects merge by key; lists replace earlier lists. Several named target presets accumulate their target entries, but an explicit `targets` list replaces them. Modalities apply in the order named; properties apply in alphabetical name order.

`"core": "benchmark"`, or `--core benchmark`, applies a profile from the `settings/core/` directory in your BindCraft2 repo under every preset, so a modality or target still refines it. `benchmark.json` sets `campaign_seed` to 0, `autotune` to false and `desperation` to false, which is the profile to use when one change is being compared against another. `default` and `reference` are not profiles to name.

`--modality VHH` replaces the file's modality. Repeat it or use commas to combine modalities. Property flags use hyphens (`--termini-accessible`); JSON uses underscores (`"termini_accessible": true`). `--set 'filters.i_pTM.threshold=0.8'` changes a nested value. Explicit `--set` assignments take precedence over the shorthand flags. Names are case-sensitive.

`bindcraft design --list-targets`, `--list-modalities`, `--list-properties` and `--list-core` list the shipped preset names, and `--list-settings` lists the setting names `--set` accepts, one a line. Use `"modality": "binder"` for the standard preset; naming only `induced_fit` or `fold_switch` adds that preset automatically. Setting a property to `false` stops selecting it, but does not erase weights or filters written explicitly elsewhere in your campaign. For example, `aa_bias.C: 0` still excludes cysteine when `disulfide_staple` is selected.

### Every setting at its default

`settings/core/reference.json` in your BindCraft2 repo is the catalogue of all 235 settings BC2 reads, each written at its default. `null` there means off or unset, not zero. **This file is never loaded**; it is documentation only. Change a default for every campaign in `settings/core/default.json` in your BindCraft2 repo, and change one campaign in its own JSON or with `--set`.

### Paths

| Path | Where BC2 looks |
| --- | --- |
| `target_path`, `binder_scaffold` written in campaign JSON | From that JSON file's directory. |
| Structure in a shipped target or scaffold preset | The supplied location; no user path is needed. |
| A path given through `--set`, or a metadata filename | From the directory where the command runs. |
| `project_folder` | From the directory where the command runs, unless an absolute path is given. Running from `BC2` with `"project_folder": "results/pdl1"` writes to `BC2/results/pdl1/`. |

### Fractions, counts and distances

Use `0.5` for 50%, `0.05` for 5% and `1.0` for 100%. This applies to thresholds for hotspot coverage, coldspot contact, framework retention and secondary-structure fractions. Fraction thresholds above 1 are rejected. Counts remain integers, distances use Å, and interface areas use Å². **`i_pAE` is normalized PAE, whereas the PAE matrix uses Å**; see [measurement scales](outputs.md#confidence-and-error).

### Resolved settings and reproducibility

Use a new `project_folder` for a different experiment. An existing campaign continues by default: run it again and repeat its original flags or save those choices in its JSON; set `"resume": false` to refuse a non-empty folder instead. `campaign_seed` makes trajectory and model draws reproducible within the same setup; it does not promise identical numerical results across environments.

```bash
bindcraft design examples/pdl1.json
bindcraft design examples/pdl1.json --metadata examples/metadata.json
```

Metadata is a separate JSON object for author, project and other descriptive fields. It is recorded with `meta_` prefixes and does not alter design settings. Keep `campaign_metadata.json` with results; it records settings, derived choices, source revision and checkpoint hashes.

## Define the target and binder

| Setting | Default / accepted input | Why change it |
| --- | --- | --- |
| `target` | Shipped name or list | Reuse a prepared target without repeating its structure and selections. |
| `targets` | List of target objects | Supply your own proteins, receptor assemblies or off-targets. Each row below prefixed `targets[]` goes inside one object. |
| `targets[].name` | Required, distinct name | Identify a target in output tables and structures. |
| `targets[].target_path` | PDB, mmCIF or FASTA | Supply a structure or sequence target. |
| `targets[].chains` | Input chain selection | `A,B` selects two chains; use actual identifiers, including multi-character names. |
| `targets[].hotspots` | Unset | Desired binding residues, e.g. `A54,A56,B12-16`. Unprefixed numbers address the first selected chain. |
| `targets[].coldspots` | Unset | Residues that should remain free of the binder. Uses the same numbering as hotspots. |
| `targets[].objective` | `target` | Use `detarget` to discourage binding to a supplied off-target. |
| `targets[].weight` | 1 | Relative importance of each target; a negative value also selects detargeting. |
| `binder_lengths` | Preset or scaffold | `[80,80]` fixes 80 residues; `[60,100]` allows the inclusive range; `[60,80,100]` allows only those choices. Length is per copy for an oligomer. |
| `binder_scaffold` | Unset | Supply an existing binder fold; overrides de novo length selection. |
| `mutate_positions` | Preset or unset | Select scaffold residues to redesign, resize or mark as binding/non-binding. |
| `aa_bias` | Preset; `binder` excludes C | Amino-acid propensities: 1 neutral, 2 favoured, 0.4 disfavoured, 0 excluded. Applies to design and redesign. |
| `copies` | 1; oligomer preset 2 | Number of binder copies in an assembly. |
| `oligomer_tie` | `symmetric` | Tie copy identities and sequence redesign. `none` leaves copies independent; it does not enforce an identical-sequence oligomer. |
| `crop_fasta_sequence` | `[10,40]` | Length of a sampled sequence-target window; `false` uses the full sequence. |
| `idr_crop_count` | 1 | Number of sequence windows treated as separate target states. |
| `validation_crop_flank` | 5 | Restore up to this many target residues on each side of a sampled window during validation, to avoid designs dependent on artificial cut ends. `0` keeps the design window alone. |
| `target_chain` | `target` | Change the internal prefix used to address prepared target chains in custom objectives. Input chain selection still uses `targets[].chains`. |
| `binder_chain` | First binder chain | Choose which prepared binder chain custom objectives address. Usually automatic. |

Chains in one target object must share a structure and coordinate frame. Separate target objects need not be aligned. The complexes are written in a common viewing frame where possible; alignment does not establish biological compatibility with a full receptor, membrane or glycan.

### Scaffold editing

Use input chain letters and residue numbers: `A26-32` redesigns a span; `A52-57(5-7)` permits that span to contain 5–7 residues. `A33*` marks a non-binding framework position; `A33+` marks a binding position without making it freely designable. A bare chain such as `A` selects that chain. More explicit residue flags use `selection+FLAG`, for example `A33+CONTACT`; see the flag table below. Commas combine selections.

| Flag | Meaning |
| --- | --- |
| `DESIGN` | Sequence may change. |
| `CONTACT` | Marked binding/paratope region. |
| `HOTSPOT` / `COLDSPOT` | Desired / avoided target contact. |
| `TEMPLATE` | Coordinates used as a structural template. |
| `CYCLIC` | Residues belonging to a cyclic chain. |
| `PAD` | Padding, excluded from physical measurements; leave this to BC2. |

Named scaffold presets already provide matching edits. When substituting another scaffold, replace both `binder_scaffold` and `mutate_positions`. A resized loop changes total length; a scaffold's length is not taken from `binder_lengths`. The `scFv` preset models two variable-domain chains and supplies no connecting linker.

## Design stages and acceptance

| Stage setting | Default updates | Purpose |
| --- | ---: | --- |
| `screen_steps` | 50 | Explore sequences and docking. |
| `refine_steps` | 25 | Concentrate amino-acid probabilities. |
| `anneal_steps` | 45 | Progress toward discrete sequences. |
| `harden_steps` | 5 | Optimise a discrete sequence. |
| `mutate_steps` | 15 | Try substitutions that improve the result. |

Stage checks run after `screen`, `refine`, `anneal`, `harden`, `mutate` and `final`. ProteinMPNN then redesigns surviving binders and candidates undergo validation. Completing a trajectory does not mean a candidate was accepted.

| Setting | Default | Why change it |
| --- | --- | --- |
| `number_of_final_designs` | 1; quickstart 10 | How many accepted sequences you want. |
| `max_trajectories` | Unset | Optional attempt limit. Leave unset to keep working toward the requested count. |
| `campaign_seed` | 0 | Choose a new set of random trajectories or reproduce the same random setting draws. |
| `reproducible_gradients` | False | Set to True to make trajectories fully reproducible on same machine, slows long runs by 20%. |
| `resume` | true | Continue the same experiment in its existing folder; `false` refuses a non-empty folder instead. |
| `trajectory_only` | false | Explore gradient designs without ProteinMPNN acceptance; specify an attempt limit. |
| `min_plddt_<stage>` | 0.6 screen/refine/mutate; 0.65 anneal/harden; 0.7 final | Require binder confidence at each stage. |
| `min_iptm_<stage>` | Unset screen/refine; 0.5 anneal/harden/mutate; 0.7 final | Require interface confidence for binding targets. |
| `max_detarget_iptm_<stage>` | Unset | Reject attempts that retain too much confidence on an off-target. |
| `max_detarget_interface_residues_final` | 3 with an off-target | Reject an accepted design that still holds an off-target, counted in binder residues touching it. Interface confidence alone does not decide this: a peptide can read 0.27 `i_pTM` with its whole face on the off-target. |
| `betasheet_reopt_trigger` | 0.15 | Sheet fraction at screen that activates extra optimisation below. |
| `betasheet_reopt_extra_refine_steps`, `betasheet_reopt_extra_anneal_steps` | 0, 0 | Give sheet-rich designs extra updates. |
| `betasheet_reopt_recycles` | Unset | Increase recycling for those designs. |

Replace `<stage>` with any of the six stage names above. Presets can change stage lengths and thresholds. Interface pTM has no default floor during screen/refine because those stages optimise mixed amino-acid probabilities. Increasing updates costs time; lowering a filter changes what counts as acceptable.

## Models and sequence redesign

| Setting | Default | Why change it |
| --- | --- | --- |
| `validation_model` | `monomer`; `multimer` for peptides, scaffolds and oligomers | Select the validation model family appropriate to the format. |
| `design_models` | 5, or 3 with multimer validation | A count or exact model-name list; more models broaden optimisation. |
| `validation_models` | Models remaining in the validation pool | A count or exact list; more models provide a broader confidence check. Design and validation models must not overlap. |
| `design_recycles`, `validation_recycles` | 1, 3 | More recycling can improve convergence at extra cost. |
| `design_dropout` | true | Enable stochasticity during gradient design. Validation disables it. |
| `sequence_candidates` | 10 | Maximum distinct ProteinMPNN sequences tried per trajectory. |
| `enough_passing_sequences` | 3 | Stop drawing after this many candidates pass; raise to evaluate more alternatives. |
| `kept_sequences` | 1 | Retain the best passing candidates by `i_pDAE`. |
| `redesign_interface` | false | Allow ProteinMPNN to change interface residues instead of holding the designed interface fixed. |
| `mpnn_model` | `v_48_020` | Select the checkpoint filename stem in the chosen weight family. |
| `mpnn_variant` | `negative` | Surface-charge preference: `neutral`, `negative` or `positive`. |
| `mpnn_fix_linker` | true | Preserve linker residues identified by a multidomain design during redesign. |
| `domain_linker_fix_cut` | 0.5 | Membership threshold used to decide which linker residues are held. |

The supplied AlphaFold names are `model_1_multimer_v3` through `model_5_multimer_v3`, and `model_1_ptm`, `model_2_ptm`. Candidate confidence is averaged over validation models; structural measurements use the first model's coordinates. A candidate that cannot recover its thresholds may stop validation early. See [candidate records](outputs.md#reading-the-tables).

## Biological options

### Property objectives and acceptance filters

| Boolean / CLI flag | When to use it |
| --- | --- |
| `humanize` / `--humanize` | Favour human-like sequence features and reduce an MHC-II anchor-score proxy. |
| `protease_stable` / `--protease-stable` | Penalise cleavage propensity and exposed loops/termini. |
| `disulfide_staple` / `--disulfide-staple` | Allow cysteine and require a geometrically compatible cysteine pair. |
| `mixed_topology` / `--mixed-topology` | Encourage non-helical structure; by default accept at most 50% helix and at least 20% beta-sheet. |
| `termini_together` / `--termini-together` | Bring the chain ends together; default final distance ceiling 10 Å. |
| `termini_accessible` / `--termini-accessible` | Orient both ends away from the target. |
| `forced_targeting` / `--forced-targeting` | Focus contact on a declared structured epitope; default hotspot coverage floor 0.5. |

These booleans default to false. They select both objectives and associated thresholds, which explicit settings can override. Sequence and geometric proxies are not measurements of immunogenicity, serum half-life or binding affinity.

### Targeting options

`forced_targeting` requires a structured target and `hotspots`. During early design it replaces exposed residues outside the protected hotspot shell with lysine, then restores the original target before hardening and validation. `forced_targeting_shell` (10 Å) sets the protected shell. Ordinary hotspots do not require this property; `coldspots` automatically add repulsion and a default contact ceiling of 0.05.

Detargeting only checks the off-targets supplied. Use `targets[].objective: "detarget"`; a named target such as `hPD1` already carries that choice. The rotation gate and the acceptance ceiling are separate controls:

| Setting | Default | Why change it |
| --- | --- | --- |
| `multitarget_steps` | 1 | Updates per target slot in a rotation. |
| `multitarget_swap_threshold`, `multitarget_swap_patience` | 0.5, 20 | Interface-confidence goal and maximum wait before leaving a binding target. |
| `multitarget_warmup_patience` | Unset | Use a separate patience limit during the first visit to a target. |
| `max_detarget_iptm` | 0.4 | Stop a detarget visit once its interface confidence falls this low. This is not an acceptance filter. |
| `detarget_check_interval`, `max_detarget_rounds` | 10, 10 | How often to revisit off-targets, and the maximum updates spent on one check. |
| `multitarget_best_round` | false | Pass onward the last round that cleared every state's stage checks, rather than simply the last round. |
| `multitarget_cumulative_filter` | true for multiple states | Use each target's best round when evaluating the stage. |
| `multitarget_filter_models` | 1 | Models used for the final multitarget stage prediction. |
| `multitarget_rounds_per_target` | Unset | List stages, e.g. `["screen","refine"]`, whose update budget should apply to each target. |
| `multitarget_merged_gradients` | true | During anneal, combine gradients from the prepared targets before updating the shared sequence. |
| `multitarget_merged_gradient_budget` | `sequence_updates` | Keep anneal sequence updates even though each costs several predictions; `model_calls` divides the updates to conserve prediction calls. |
| `multitarget_tied_redesign` | true | Redesign against all targets together. False rotates candidates between target structures. |

### Conformational design

`induced_fit` requests movement of the binding surface; `fold_switch` requests a whole-fold difference between free and bound structures. Both presets require one target. `binder_shapes` instead specifies groups of states sharing a conformation, e.g. `[["hPDL1"],["binder_alone"]]`, used by `weights_fold_switching`. Structural differences do not establish switching kinetics or thermodynamic preference.

| Setting | Default | Why change it |
| --- | --- | --- |
| `binder_shapes` | Unset | Define explicit conformation groups. |
| `induced_fit_delta`, `induced_fit_interface_cutoff` | Objective 3 Å; preset 5 Å, 8 Å | Desired interface RMSD and distance defining the interface. |
| `induced_fit_tm_target` | 0.6 | Desired ceiling on similarity between whole folds. |
| `induced_fit_monomer_steps` | 30 | Maximum unbound-binder optimisation steps in each induced-fit block. |
| `induced_fit_monomer_chunk` | 5 | Steps between confidence checks in that block. |
| `induced_fit_monomer_plddt` | 0.7 | Confidence goal for adaptive stopping. |
| `induced_fit_monomer_adaptive` | true | Stop the block early when its confidence goal is met. |
| `induced_fit_steps` | 1 | Updates in an induced-fit slot mixed with target rotation. |
| `induced_fit_mpnn_threshold`, `induced_fit_mpnn_shell` | 2 Å, 1 | Identify moving residues and neighbouring positions to hold during redesign. |
| `induced_fit_mpnn_designed_share` | 0.25 | Minimum fraction still available for redesign when much of the binder moves. |
| `paratope_conformations` | Unset; VHH uses both | List `extended`, `folded_back` or both to sample exposed versus framework-packed paratope loops. |

### Starting conformations

| Setting | Default | Why change it |
| --- | --- | --- |
| `initial_guess` / `--initial-guess` | false | Start the re-prediction of each redesigned candidate from the pose the trajectory folded. The gradient stages are untouched, and a binder folded from nothing reaches it the same way a scaffold does, because ProteinMPNN decodes onto the predicted backbone. Measured on 17 matched candidates it raised binder pLDDT on every one and left interface pTM and pAE flat to slightly worse, so it is off by default. Rungs 1, 3 and 5 to 7 of the [desperation ladder](#the-desperation-ladder) turn it on. |
| `bigbang` / `--bigbang` | false | Property flag that sets `bigbang_initialization`. |
| `bigbang_initialization` | false | Also start the gradient stages from the coordinates on hand, so the target begins folded in its own frame while a binder folded from nothing still springs from the origin. Measured to cost a campaign: 20 of 20 trajectories died at screen at a binder pLDDT of 0.56 to 0.59 where flexibility alone put 5 of 6 past screen at 0.81, and seeding a VHH trajectory, which does have coordinates to start from, lost pLDDT and interface pTM on 6 of 6. Not on the desperation ladder, and not changed by the autotuner. |
| `target_flexibility` | 0 | Fraction of templated target residues whose sequence and sidechain information are withheld, while retaining their backbone. Increase only when target-side flexibility is part of the experiment. |

### Peptide and domain controls

| Setting | Default | Why change it |
| --- | --- | --- |
| `peptide` | false | Judge a peptide in the bound complex without the default free-fold confidence gate; selected by peptide presets. |
| `cyclize_peptide` | false | Enable head-to-tail cyclic residue offsets; implies peptide behaviour. |
| `cyclic_offset_mode` | `direction` | `distance` wraps separation; `direction` also preserves ring direction; `neighbours` distinguishes only nearby and distant pairs beyond two residues. |
| `n_domains`, `min_domain_size`, `max_domain_size`, `max_domains` | 2, 50, 180, 2 | Requested domain count and permitted domain sizes. |
| `domain_rg_weight`, `domain_sep_weight` | Objective 0.3, 0; multidomain preset 0.5, 1 | Encourage compact individual domains and separation between them. |
| `domain_contact_cutoff`, `domain_pae_margin` | 8 Å, 0.15; preset margin 0.2 | Define interdomain contact and error margins. |
| `domain_linker_gap`, `domain_linker_sharpness` | 0.1, 0.03 | Control linker membership inferred from domain confidence. |
| `domain_linker_helix_weight` | 0; multidomain preset 0.5 | Encourage helical character in the linker. |

### Other objective controls

These are top-level shortcuts for the corresponding [loss parameters](#loss-parameters). They tune the objective, independently of its final acceptance threshold.

| Setting | Default | Why change it |
| --- | --- | --- |
| `interface_contact_distance`, `non_contact_distance` | 20 Å configured, 14 Å | Distogram contact distances for attraction and off-target avoidance. |
| `termini_distance_threshold` | 7 Å | Distance below which the termini objective stops pulling. |
| `disulfide_distance`, `disulfide_sigma` | 3.8 Å, 1.5 | Preferred Cβ separation and smooth width for cysteine pairing. |
| `disulfide_sequence_separation`, `disulfide_temperature` | 3, 0.1 | Minimum sequence separation and softness of pairing choices. Each cysteine is paired with one partner, so a crowd of cysteines cannot satisfy itself. |
| `humanization_species` | `human` | Sequence-preference panel. |
| `humanization_mhc2_weight`, `humanization_coupling_weight`, `humanization_hydro_weight` | 1, 0.5, 0 | Relative contributions of MHC-II anchors, sequence coupling and hydrophobicity. |
| `exposed_loops_measure` | `plddt` | Loop proxy: `plddt`, `geometry` or `both`. |
| `exposed_loops_distinguish_sheets` | true | Treat extended strand geometry separately from loops. |

## Losses

A loss steers optimisation; a filter decides acceptance. Set a scalar or sampled weight using `weights_<name>` or `"losses": {"<name>": value}`. `null` or 0 switches it off. A two-value weight array samples a continuous range each trajectory; three or more entries sample discrete choices. Zero is excluded from sampled weights. Drawn values are recorded with accepted structures.

```json
{
  "weights_binder_helicity": [-0.5, 0.2],
  "weights_interface_contacts": 1.0,
  "losses": {
    "compactness": null,
    "interface_contacts": {"params": {"cutoff": 20.0, "contacts_per_residue": 2}}
  }
}
```

An object under `losses.<name>` accepts `prediction_state` and `params`; set its weight separately with `weights_<name>`. Ordinary positive weights minimise the named loss. **Helicity is a signed exception in practice:** a negative weight rewards helix; a positive weight discourages it. Weights have different numerical scales and should not be interpreted as percentages of importance.

The table lists every objective. Default weights describe the standard binder before biological options change them; “off” means enable explicitly or through a preset.

| Weight setting | Default | What it steers |
| --- | --- | --- |
| `weights_plddt_loss` | 0.1 | Confidence of the binder fold. |
| `weights_target_plddt` | off | Confidence of marked target binding residues, or the whole target when none are marked. |
| `weights_experimentally_resolved` | off | AlphaFold’s estimate of residue resolvability; a prediction, not an experimental observation. |
| `weights_sequence_entropy` | off | Keep amino-acid probabilities diverse during early optimisation. |
| `weights_humanization` | off | Sequence preferences and MHC anchor-score proxy. |
| `weights_protease_sites` | off | Cleavage-site propensity. |
| `weights_exposed_loops` | off | Exposed-loop susceptibility using a soft confidence/geometry proxy. |
| `weights_exposed_termini` | off | Exposure of the ends of each binder chain. |
| `weights_binder_pae` | 0.4 | Confidence in relative positions within the binder. |
| `weights_interface_pae` | 0.1 | Confidence in the binder–target pose. |
| `weights_compactness` | 0.5 | Binder radius of gyration relative to a globular protein of its length. |
| `weights_iptm_loss` | 0.05 | Interface confidence for binding targets. |
| `weights_ptm_loss` | off | Confidence in the entire complex as one structure. |
| `weights_target_rmsd` | off | Keep the target close to the supplied coordinates. |
| `weights_target_rigidity` | off | Keep a receptor assembly’s chains in their supplied arrangement. |
| `weights_distogram_cce` | off | Agreement between predicted distances and the structural template. |
| `weights_com_distance` | off | Bring target and binder centres together, relative to binder size. |
| `weights_binder_coldspot` | off | Keep target contact off the marked non-binding framework. |
| `weights_binder_intra_coldspot` | off | Keep paratope loops off the binder’s non-binding face. |
| `weights_binder_intra_hotspot` | off | Pack paratope loops against that face. |
| `weights_coldspot_repel` | off | Keep the binder away from target coldspots. |
| `weights_interface_contacts` | 1.0 | Promote binder–target contacts. |
| `weights_non_contact` | off | Discourage contacts with off-targets. |
| `weights_binder_contacts` | 1.0 | Promote contacts within the binder. |
| `weights_binder_helicity` | -0.3 | Binder helicity; negative favours helix. |
| `weights_target_helicity` | off | Target helicity; negative favours helix. |
| `weights_non_helical` | off | Discourage helical binder structure. |
| `weights_termini_distance` | off | Bring the binder’s chain ends together. |
| `weights_termini_angle` | off | Direct both ends away from the target. |
| `weights_n_terminus_away` | off | Direct the N terminus away from the target. |
| `weights_c_terminus_away` | off | Direct the C terminus away from the target. |
| `weights_disulfide` | off | Charge for every cysteine left without a partner. The loss reads in free cysteines, so a weight of 1.0 prices one unpaired cysteine at 1.0. |
| `weights_induced_fit_global` | off | Whole-fold difference between free and bound structures. |
| `weights_induced_fit_interface` | off | Movement of the binding surface relative to the binder core. |
| `weights_fold_switching` | off | Difference between explicit conformation groups. |
| `weights_collective_softness` | off | Favour a shared collective hinge motion over local floppiness. |
| `weights_multidomain` | off | Separate, compact domains connected by a linker. |

### Loss parameters

Use `losses.<name>.params` to change the parameters below. Defaults shown here are the effective standard-binder entries where configured, otherwise the objective's own defaults. Presets can override them. A parameter does not enable an objective whose weight is off.

Shared selectors are `prediction_state` (which prepared target or `binder_alone` to score), `chain` (one chain), and `binder`/`target` (their prepared roles). Leave these automatic for ordinary campaigns; use them to address a particular conformation or partner. A `reference_state` names the conformation to compare against. `complex` is resolved to the appropriate bound state. Objectives normally addressed to `binder_alone` can be applied to the binder in a bound state by the design workflow.

| Objective | Additional parameters and defaults |
| --- | --- |
| `humanization` | `species='human'`, `coupling_weight=0.5`, `hydrophobicity_weight=0.0`, `mhc_class_ii_weight=1.0`, `temperature=0.1` |
| `exposed_loops` | `measure='plddt'`, `plddt_threshold=0.85`, `plddt_temperature=0.05`, `window=3`, `distinguish_sheets=True` |
| `exposed_termini` | `terminus_length=3` |
| `binder_pae` | `domain_ids=()`, `per_protomer=False` |
| `compactness` | `eps=1e-08`, `per_protomer=False` |
| `com_distance` | `radius_ratio=0.9`, `eps=1e-08` |
| `binder_coldspot` | `cutoff=8.0`, `contacts_per_residue=1`, `designed_only=True` |
| `binder_intra_coldspot` | `cutoff=8.0`, `contacts_per_residue=1`, `sequence_separation=20` |
| `binder_intra_hotspot` | `cutoff=8.0`, `contacts_per_residue=1`, `sequence_separation=20` |
| `coldspot_repel` | `cutoff=8.0`, `contacts_per_residue=1` |
| `interface_contacts` | `cutoff=20.0`, `contacts_per_residue=2`, `contact_residue_count=float('inf')` |
| `non_contact` | `cutoff=14.0`, `contacts_per_residue=2` |
| `binder_contacts` | `cutoff=14.0`, `contacts_per_residue=2`, `contact_residue_count=float('inf')`, `sequence_separation=9`, `per_protomer=False` |
| `binder_helicity` | `cutoff=6.0` |
| `target_helicity` | `cutoff=6.0` |
| `non_helical` | `cutoff=6.0` |
| `termini_distance` | `threshold_distance=7.0` |
| `disulfide` | `distance=3.8`, `sigma=1.5`, `sequence_separation=3`, `temperature=0.1` |
| `induced_fit_global` | `tm_target=0.6` |
| `induced_fit_interface` | `interface_rmsd_target=3.0`, `cutoff=8.0`, `interface_mask=None` |
| `fold_switching` | `binder_shapes=()`, `tm_target=0.6` |
| `collective_softness` | `contact_decay=8.0`, `damping=0.01`, `power_iterations=12`, `eps=1e-08` |
| `multidomain` | `domain_ids=()`, `n_domains=2`, `min_domain_size=50`, `max_domain_size=180`, `max_domains=2`, `seed=0`, `domain_rg_weight=0.3`, `domain_sep_weight=0.0`, `domain_contact_cutoff=8.0`, `domain_pae_margin=0.15`, `domain_linker_gap=0.1`, `domain_linker_sharpness=0.03`, `domain_linker_helix_weight=0.0` |

`cutoff` is a distance in Å. `contacts_per_residue` is the desired count per selected residue; `contact_residue_count` limits how many residues contribute (`inf`: all). `sequence_separation` excludes nearby sequence neighbours. `per_protomer` scores each oligomer copy separately; BC2 enables it for oligomer compactness, contacts and PAE. `designed_only` restricts the binder region under consideration. `domain_ids` assigns residues to domains; leave empty for automatic assignment. `binder_shapes` names conformation groups.

For loop scoring, `plddt_threshold` and `plddt_temperature` control confidence-based loop propensity; `window` smooths neighbouring residues. `terminus_length` is the number of end residues scored. `radius_ratio` controls how tightly the binder surrounds the target. `sigma` and `temperature` smooth geometric or sequence choices. `interface_rmsd_target` and `tm_target` are desired conformational differences; `interface_mask` is an optional explicit residue mask, normally inferred automatically. In `collective_softness`, `contact_decay` sets the interaction range, `damping` regularises motion and `power_iterations` controls its numerical estimate. `eps` only stabilises division. Domain and humanization parameters have the meanings given in [biological options](#biological-options).

## Filters

A filter entry accepts `threshold`, `higher`, `mandatory`, `prediction_state` and `params`. `higher: true` requires a value at or above the threshold; false requires at or below it. `threshold: null` disables the check. `mandatory: false` allows a missing measurement; **it does not disable a threshold when a value is measured**. To record a measurement without excluding values, use a nonrestrictive finite threshold and `mandatory: false`.

```json
{
  "filters": {
    "i_pTM": {"threshold": 0.7, "higher": true},
    "Binder_BetaSheet_Fraction": {"threshold": 0.2, "higher": true},
    "Binder_RMSD": {"threshold": null}
  }
}
```

The standard `binder` preset uses free-binder pLDDT 0.7; shared settings use final interface pTM 0.6. Other default candidate checks are pTM 0.55, normalized interface PAE at most 0.35, no backbone clashes, and at least 7 interface residues. Biological presets adjust or add checks. A custom `filters.<metric>` entry takes precedence over a top-level convenience threshold for that metric.

The shortcuts below turn the corresponding acceptance requirement into a floor (`min_`) or ceiling (`max_`). Unless a preset sets one, most biological measurements are recorded with no restrictive threshold. See [all measurement definitions and units](outputs.md#measurements), including measurements configured only through `filters`.

| Threshold setting | Measurement / reason to use it |
| --- | --- |
| `min_monomer_plddt_final` | `Unbound_Binder_pLDDT` — require a confident free binder fold. |
| `min_ptm_final` | `pTM` — require overall complex confidence. |
| `max_ipae_final` | `i_pAE` — limit normalized interface error. |
| `min_iptm_final` | `i_pTM` — require interface confidence. |
| `min_target_plddt_final` | `Target_pLDDT` — require a confident target structure. |
| `max_binder_chain_breaks_final` | `Binder_Chain_Breaks` — limit discontinuities in the backbone. |
| `max_binder_free_cysteines_final` | `Binder_Free_Cysteines` — limit estimated unpaired cysteines. |
| `min_binder_disulfides_final` | `Binder_Disulfides` — require distance-compatible cysteine pairs, counted as a pairing. |
| `max_coldspot_contact_final` | `Coldspot_Contact_Fraction` — keep selected target residues free. |
| `max_cyclic_closure_distance_final` | `Cyclic_Closure_Distance` — require close cyclic chain ends. |
| `max_exposed_loop_fraction_final` | `Exposed_Loop_Fraction` — limit exposure among loop residues. |
| `max_helix_fraction_final` | `Binder_Helix_Fraction` — limit helical content. |
| `max_induced_fit_tm_final` | `Induced_Fit_TM` — require a whole-fold change. |
| `max_interdomain_contact_final` | `Interdomain_Contact_Fraction` — keep domains from collapsing together. |
| `max_mhc_anchor_score_final` | `MHC_Anchor_Score` — limit the humanization proxy. |
| `max_off_epitope_contact_final` | `Off_Epitope_Contact_Fraction` — focus contact within the protected epitope. |
| `max_off_paratope_contact_final` | `Off_Paratope_Contact_Fraction` — keep contact within the designated paratope. |
| `max_oligomer_symmetry_rmsd_final` | `Oligomer_Symmetry_RMSD` — require approximate cyclic symmetry. |
| `max_protease_site_score_final` | `Protease_Site_Score` — limit predicted cleavage propensity. |
| `max_scaffold_framework_rmsd_final` | `Scaffold_Framework_RMSD` — retain the starting framework geometry. |
| `max_surface_hydrophobicity_final` | `Surface_Hydrophobicity` — limit exposed hydrophobic residues. |
| `max_termini_distance_final` | `Termini_Distance` — bring chain ends together. |
| `max_terminus_exposure_final` | `Terminus_Exposure` — limit solvent exposure of terminal residues. |
| `min_domain_separation_ratio_final` | `Domain_Separation_Ratio` — require spatially separate domains. |
| `min_epitope_residues_contacted_final` | `Epitope_Residues_Contacted` — require contact across an epitope. |
| `min_framework_packing_final` | `Framework_Packing_Fraction` — require loops to cover the non-binding face. |
| `min_hotspot_contact_final` | `Hotspot_Contact_Fraction` — require coverage of named hotspots. |
| `min_induced_fit_interface_rmsd_final` | `Induced_Fit_Interface_RMSD` — require movement of the binding surface. |
| `min_interface_buried_area_final` | `Interface_BuriedArea` — require a minimum binder-side buried area. |
| `min_receptor_chains_contacted_final` | `Receptor_Chains_Contacted` — require engagement of several receptor chains. |
| `min_scaffold_sequence_retained_final` | `Scaffold_Sequence_Retained_Fraction` — retain held scaffold sequence. |
| `min_target_crop_length_final` | `Target_Crop_Length` — require sufficient sequence-target coverage. |
| `min_termini_away_cosine_final` | `Termini_Away_Cosine` — direct both ends away from the target. |
| `min_n_terminus_away_cosine_final` | `N_Terminus_Away_Cosine` — direct the N terminus away. |
| `min_c_terminus_away_cosine_final` | `C_Terminus_Away_Cosine` — direct the C terminus away. |

Off-target stage ceilings use `max_detarget_iptm_<stage>`, and acceptance rejects a design still holding an off-target through `max_detarget_interface_residues_final`. Entries `i_pTM_detarget`, `i_pAE_detarget` and `Interface_Residues_detarget` measure the states a campaign avoids, one filter each, without naming them. Measurements such as beta-sheet fraction, interface amino-acid counts, all-atom clashes and structured-residue confidence use a `filters` entry rather than a top-level shortcut. Filter parameters are listed with the [measurement definitions](outputs.md#measurement-parameters).

## Autotuning and parameter sweeps

| Setting | Default | Why change it |
| --- | --- | --- |
| `autotune` | true | Adjust screen/refine length to observed progress; false keeps a fixed schedule. |
| `autotune_loss_weights` | false | Also explore objective-weight multipliers. |
| `desperation` | true | Climb the [desperation ladder](#the-desperation-ladder) once the campaign has accepted nothing for long enough. A separate flag from `autotune`; false keeps the campaign at its own settings. |
| `desperation_trajectories` | 750 | Trajectories since the last accepted design before the first rung of that ladder is taken. |
| `parameter_sweep` | Absent | Compare controlled variants of selected settings; `true` uses default axes, or supply the object below. |

The autotuner reviews blocks of ten trajectories and moves two things only: the screen and refine stage lengths, each held between half and twice the length the campaign configured, and, where `autotune_loss_weights` is set, the `weights_*` the campaign changed itself. An alternated weight returns to the campaign's own value as soon as a design is accepted. The autotuner never touches recycles, target flexibility, the validation pool, `initial_guess` or `bigbang_initialization`. Its values live in `.campaign_state.json`, and each trajectory's `autotuned` column records what had been moved when it ran.

### The desperation ladder

**`desperation` is a separate flag from `autotune`.** The autotuner moves run controls the campaign can afford to have moved; the ladder trades away settings the campaign asked for. After `desperation_trajectories` trajectories since the last accepted design, one rung is climbed per further block of fifty trajectories, so at the defaults the whole ladder spans 750 to 1050 trajectories and every rung is given long enough for an acceptance to appear. A rung replaces the one below it rather than adding to it, and the whole ladder is dropped as soon as a design is accepted.

| Rung | Runs at |
| --- | --- |
| 1 | `initial_guess` |
| 2 | `target_flexibility` 0.5 |
| 3 | `initial_guess`, `target_flexibility` 0.5 |
| 4 | `validation_model` `multimer` |
| 5 | `validation_model` `multimer`, `initial_guess` |
| 6 | `validation_model` `multimer`, `initial_guess`, `target_flexibility` 0.5 |
| 7 | `validation_model` `multimer`, `initial_guess`, `target_flexibility` 0.5, `design_recycles` 3 |

The initial guess and the flexibility are tried alone before they are tried together, so a rung that works says which change bought the design. Rungs 4 to 6 move validation off the held-back monomer models onto held-out multimer models, splitting the multimer pool 3 to design and 2 to validate, for a target whose interface the monomer models cannot resolve. More recycles come last because they cost only time; a campaign that already asks for more than three keeps its own count. The rung is read off the campaign's own tables, so every worker and a resumed campaign stand on the same one, and a `trajectory_only` campaign never climbs at all.

**A design accepted on a rung was accepted against an easier design task, judged by a validation loosened to match, and nothing about a wet-lab experiment is loosened with it.** Flexibility and the initial guess reach both predictors deliberately, because a binder that folds only against a loosened target would otherwise pass every design filter and then be failed by a rigid validation. Treat such a design as a weaker candidate than one accepted at the settings the campaign asked for. The campaign log prints a `desperation:` line naming the rung and the settings it runs at on every trajectory the ladder applies to, and those same settings appear in that trajectory's `autotuned` column in `1_Trajectories/!_Trajectories.csv`. `--core benchmark` switches `desperation` and `autotune` off together, with a fixed `campaign_seed` and `reproducible_gradients`, which is what a controlled comparison needs; see `settings/core/benchmark.json` in your BindCraft2 repo and [input tiers](#input-tiers-and-overrides).

```json
{
  "max_trajectories": 150,
  "parameter_sweep": {
    "axes": ["screen_steps", "refine_steps"],
    "levels": [0.5, 2.0],
    "max_arms": 5,
    "block_trajectories": 10
  }
}
```

| Sweep option | Default | Meaning |
| --- | --- | --- |
| `axes` | Derived from active biological objectives and stages | Settings to compare one at a time against the baseline. |
| `levels` | `[0.5,2.0]` | Multipliers applied to each axis. |
| `multiplier` | Unset | Use one multiplier instead of `levels`. |
| `max_arms` | 5 | Total arms including the unchanged baseline. |
| `block_trajectories` | 10 | Advance arms in equal blocks, useful when a job ends early. |

A sweep needs `max_trajectories`, divides its budget between arms, and disables ordinary autotuning. It cannot vary model-construction settings or a weight already sampled from an array. Pin binder length for cleaner paired comparisons. Read [sweep outputs](outputs.md#sweeps): a highest-ranked arm is not persuasive unless `resolved` is true, and should be confirmed in a fresh campaign.

## Output and execution settings

### The name a design carries

`campaign_name` identifies the experiment and `binder_name` optionally identifies the molecule. Identical labels are not repeated. Names include modality, binder length and a recipe hash by default; redesigned candidates add `_candidate<n>` or `_seq<n>`, and multistate structures add the target name or sampled target span. See [file identities](outputs.md#names-and-metadata).

| Setting | Default | Why change it |
| --- | --- | --- |
| `campaign_name`, `binder_name` | Unset | Give files meaningful experiment and binder labels. |
| `project_folder` | `Binders` | Choose the results location. |
| `hash_design_names` | true | False uses a shorter per-campaign counter; hashes remain recorded. |
| `save_design_frames` | false | Keep one structure per recorded update/state. |
| `save_design_sequences` | false | Keep the compressed amino-acid probability arrays for designed chains over the recorded updates. |
| `save_design_trajectory` | false | Keep only the fold the trajectory ended on, before redesign: one file per target state, plus the unbound binder where the trajectory predicted one. |
| `save_design_animations` | false | Keep interactive trajectory viewers; also enables frames. |
| `save_loss_plots` | false | Keep metric plots; also enables frames. |
| `save_failed_trajectories` | true | Keep structures from attempts that produced no accepted sequence. False retains metric records. |
| `save_failed_refolds` | true | Keep predicted structures for rejected ProteinMPNN candidates. Rows remain recorded if false. |
| `save_binder_monomers` | true | Keep a free-binder structure when that state was predicted. |
| `archive_trajectories` | false | Zip each completed trajectory folder. |
| `sparse_output` | false | Disable optional structures, viewers, plots and the sequence archive unless individually requested; useful for transfers. |
| `relax_accepted_designs` | false | Save an additional restrained, clash-minimised complex. This does not replace the prediction. |
| `relax_steps`, `relax_learning_rate` | 200, 0.02 | Duration and step size of optional relaxation. |
| `relax_restraint_backbone`, `relax_restraint_sidechain` | 10, 0.5 | Restrain coordinates toward their starting positions. |
| `relax_weight_bond`, `relax_weight_clash` | 100, 5 | Relative penalties for distorted bonds and clashes. |
| `relax_overlap_tol`, `relax_min_sep` | 0.4 Å, 2.5 Å | How far two atoms may approach below contact, and the separation no pair is pushed under. |
| `length_bucket_size` | 32 | Pad lengths to reuse compiled calculations; 1 disables padding. |
| `compile_next_length` | true | Prepare the next length while the current trajectory runs. |
| `subbatch_size` | `auto` | Split large calculations to reduce memory; an integer fixes the chunk size, `null` disables chunking. |
| `attention_backend` | `auto` | Choose attention implementation; leave automatic unless diagnosing performance. |
| `use_cueq` | false | Enable optional cuEquivariance kernels when installed. |
| `auto_multi_gpu` | true | Use the visible GPU allocation automatically; false keeps a single process. |
| `workers_per_gpu`, `max_workers_per_gpu` | `auto`, 8 | Set or cap concurrent design workers per card. |
| `design_workers` | Unset | Cap total workers across the allocation. |
| `gpu_ids` | All visible | Select a subset of visible GPUs. |
| `worker_launch_stagger` | 0 seconds | Space worker starts to reduce startup pressure. |

Worker packing is limited by GPU memory, host memory and the campaign's attempt budget; sweep fan-out uses the budget of one arm. A trajectory-only run uses one worker per GPU. Lengths can be divided among workers, so faster length groups can appear more often in results. See [execution and environment overrides](installation.md#gpu-and-memory-controls) for scheduling, caches and GPU visibility.

## Installation

Use the [installation and running guide](installation.md) for local environments, Slurm, containers, offline nodes and troubleshooting.

## Running on a cluster

See [Slurm submission](installation.md#slurm-and-other-schedulers).

## Outputs and how to read them

See the [complete output and measurement reference](outputs.md), including every result file, confidence scale, structural metric, ranking command and sweep record.

### Ranking and refiltering

See [ranking and refiltering](outputs.md#ranking-and-refiltering).
