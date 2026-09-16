# BC2 outputs and measurements

[Settings](reference.md) · [Installation and running](installation.md) · [README](../../README.md)

[Files](#files) · [Table conventions](#reading-the-tables) · [Measurements](#measurements) · [Trajectory viewers](#trajectory-records-and-viewers) · [Ranking and filtering](#ranking-and-refiltering) · [Sweeps](#sweeps)

Open `3_Ranked/!_Ranked.csv` first, then inspect the corresponding structures. Acceptance means the configured computational checks passed. It does not establish affinity, specificity or experimental stability. If nothing has been accepted, inspect `2_Refolded/!_Refolded.csv` for failed filters, then `1_Trajectories/!_Trajectories.csv` for attempts stopped before redesign.

## Files

Paths below are inside `project_folder`. A stage folder appears when its first result is written; its absence means the campaign has not written results for that stage.

| File or folder | What it contains / when to use it |
| --- | --- |
| `1_Trajectories/!_Trajectories.csv` | One row per completed or terminated gradient-design attempt. See its final metrics and termination stage. |
| `1_Trajectories/<design>/<design>_losses.csv` | One row per recorded sequence update: stage, update number and prediction/objective readings. |
| `1_Trajectories/<design>/<design>_sequences.npz` | Compressed amino-acid probability arrays for the designed chains over the recorded updates; written only when `save_design_sequences` is set (off by default). |
| `1_Trajectories/<design>/<design>_<stage>_<update>_<state>.cif` | Optional intermediate structures; enabled by `save_design_frames`, animations or loss plots. |
| `1_Trajectories/<design>/<design>_trajectory[_<state>].cif` | The fold the trajectory ended on, written before ProteinMPNN redrew its sequence; enabled by `save_design_trajectory`. One file per target state, suffixed by state where a campaign has several, and `_trajectory_monomer.cif` for the unbound binder of an induced-fit or fold-switching campaign. |
| `1_Trajectories/<design>/<design>_losses.png` | Optional per-metric trajectory plots, enabled by `save_loss_plots`. |
| `1_Trajectories/<design>/<design>_trajectory.html` | Optional interactive trajectory viewer, enabled by `save_design_animations`. |
| `1_Trajectories/<design>/<design>_forced_targeting_<state>.pdb` | The temporarily modified target used for epitope focusing; inspect which surface was protected. |
| `1_Trajectories/<design>.zip` | Archived trajectory folder when `archive_trajectories` is enabled. |
| `2_Refolded/!_Refolded.csv` | Every scored ProteinMPNN candidate, including its outcome and failed filters. |
| `2_Refolded/Complexes/<design>_candidate<n>[_<target>].cif` | The candidate's predicted complex and metadata. Rejected structures are kept unless `save_failed_refolds` is false. |
| `2_Refolded/BinderMonomer/<design>_candidate<n>_monomer.cif` | The binder repredicted alone, superposed on its bound pose; written when the free binder is predicted (e.g. the `Binder_RMSD` filter is active) and `save_binder_monomers` is set. |
| `2_Refolded/filtered.csv` | Passing shortlist from `bindcraft filter` with its default candidate table. |
| `3_Ranked/!_Ranked.csv` | Every accepted sequence, best-first by `i_pDAE`, rewritten after each acceptance; the single record of what was accepted. Missing scores sort last. When the campaign closes it drops any row whose complex is no longer in the folder (delete a `.cif` to reject a design; its prediction survives in `2_Refolded/Complexes/`) and, on a resumed run, designs replacements to refill the target. |
| `3_Ranked/<design>_seq<n>[_<target>].cif` | Accepted predicted complexes. Multiple target states get separate files. |
| `3_Ranked/<design>_seq<n>_monomer.cif` | Free binder, when predicted and `save_binder_monomers` is enabled; may contain several binder chains. |
| `3_Ranked/<design>_seq<n>.html` | Copy of the originating trajectory viewer when animations are enabled. |
| `3_Ranked/relaxed/` | Additional restrained, clash-minimised complexes when relaxation is enabled. Original predictions remain alongside them in `3_Ranked/`. |
| `3_Ranked/ranked_by_<metric>.csv` | A new ranking written by `bindcraft rank`; the standard ranking is retained. |
| `summary.csv` | Long-form counts and metric summaries across attempts, stages, candidates and accepted designs. |
| `campaign_metadata.json` | Input provenance, complete resolved settings and derived choices. Different metadata on a resumed run gets a content-addressed filename. |
| `.campaign_state.json` | Shared counters, autotuning state and the recipe hashes claimed before design (including interrupted attempts, so a recipe is never repeated); used to resume and coordinate workers. Keep with the campaign. |
| `.redesigned_sequences.txt` | Sequences claimed before validation; prevents duplicate ProteinMPNN sequences across workers and resumed runs. |
| `workers/campaign_settings.json` | Resolved settings supplied to parallel workers. |
| `workers/worker_<NN>_gpu_<id>.log` | Each worker's complete console record; useful when one worker fails. |
| `<arm>/`, `sweep.csv`, `best_settings.json` | Per-arm campaigns and comparison records from a parameter sweep. |
| `scored.csv` | Cached coordinate-derived measurements when ranking an external structure folder. |
| `bindcraft_<jobid>.out` | Console output from the supplied Slurm script, in the submission directory. |

Commands may write an explicitly requested `--output` or `--rejected` filename elsewhere. Temporary `.partial` files and lock files coordinate safe writing; they are not extra design results. Compiled-model caches are also working files, not binder evidence.

Older campaigns with flat `trajectories.csv`, `candidates.csv`, `accepted.csv`, `ranked.csv`, `trajectories/` and `accepted/` are read and resumed in their existing layout. Do not rearrange them manually to match the newer folders.

## Reading the tables

| Column / convention | Meaning |
| --- | --- |
| `design` | File identity. `_candidate<n>` counts ProteinMPNN draws; `_seq<n>` identifies retained sequences. |
| `hash` | Design-recipe identity. The same hash can have several sequence candidates. It is also the join key back to `1_Trajectories/!_Trajectories.csv`, which is the only table carrying `terminated` and `autotuned`. |
| `trajectory` | Claimed attempt number; used to derive its random seed. |
| `terminated` | Stage at which gradient design stopped. Blank means it completed, not that ProteinMPNN accepted a sequence. |
| `outcome` | Candidate `passed` or `rejected`; passing candidates may still fall outside `kept_sequences`. |
| `failed_filters` | Comma-separated failed checks, including a target suffix where relevant. “not measured” is different from a low score. |
| `autotuned` | Settings that differed from the campaign's own when that attempt ran, from the autotuner and from the [desperation ladder](reference.md#the-desperation-ladder) alike. An entry naming `initial_guess`, `target_flexibility`, `validation_model` or a raised `design_recycles` means the attempt ran on a rung of that ladder, against an easier task than the campaign asked for. |
| `rank` | Position in that particular ranking, not a probability of experimental success. |
| `targets`, `target_weights` | Target names and weights in the order used by multi-target metric cells. |
| `Binder_Sequence` | One-letter sequence, with `/` between binder chains. |
| `Interface_Binder_Residues`, `Interface_Target_Residues` | Contacting residues in the numbering of the written structures. Binder chains use `/` in the same order as the sequences. |
| `meta_<name>` | User-supplied descriptive metadata. |

Final tables keep one column per metric. For multiple targets, cells contain semicolon-separated readings in `targets` order, with empty positions retained for missing readings. Targets are ordered by weight. Trajectory logs instead use columns such as `<state>.iptm`, where `<state>` is a target name or `binder_alone`. Do not treat a blank as zero or average binding and detargeting states together.

Confidence metrics are averaged over the validation models used; geometric metrics use the first model's coordinates. A refold rejected before the full ensemble completes carries only the available model readings. Per-model measurements are stored in structure metadata. Compare candidates with this distinction in mind.

### Names and metadata

Names begin with optional `campaign_name` and `binder_name`, followed by modality, binder length and recipe hash (or a campaign-local counter when `hash_design_names` is false). Identical campaign and binder labels are not repeated. Multiple target files add their target name; cropped sequence targets add the sampled residue span. A crop span in the name identifies the design window, while `Target_Crop_Length` during validation can include restored flanks.

The mmCIF `bindcraft` metadata records design identity, source/model provenance, candidate outcome, filter readings, per-model scores and redesign settings. Sampled objective weights are stored with the design. Residue annotations record designed/paratope positions so later scoring can recover their roles. `campaign_metadata.json` additionally records absolute input paths, resolved settings, binder length choices, source revision (`-dirty` when modified), SHA-256 checkpoint hashes and a `resolved` block for model choices, stage budgets and binder chain configuration. Keep this file when moving results.

The structure's B-factor column stores per-residue pLDDT on a **0–100** scale. Local quality annotations include `pLDDT`, `PAE_mean` and per-residue `i_pDAE` where available. This B-factor convention applies to BC2 predictions; experimental B-factors mean something else.

## Measurements

Most biological measurements are recorded automatically only when relevant; others are computed by `score`, `rank` or an explicitly requested filter. Missing data means the necessary state, annotation or atoms were unavailable. Direction below describes biological interpretation; a “higher” score is not always preferable for every experiment.

### Confidence and error

| Measurement | Scale / interpretation |
| --- | --- |
| `pLDDT` | Mean binder confidence in the bound state, 0–1; higher is more confident. |
| `Unbound_Binder_pLDDT` | Confidence of the binder predicted alone, 0–1. Peptides need not have a confident free fold. |
| `Target_pLDDT` | Mean confidence of the target in that predicted state, 0–1. |
| `SS_pLDDT` | Binder confidence over helix/sheet residues, 0–1; excludes loops. |
| `Binder_pLDDT` | Mean binder CA B-factor when scoring a structure file; 0–100 for BC2 predictions. It is not recoverable from true experimental B-factors. |
| `pTM` | Predicted confidence in the entire complex geometry, 0–1; higher is better. |
| `i_pTM` | Interface confidence, 0–1; higher is better for a binding target. |
| `i_pAE` | Mean interface PAE divided by **31 Å**; lower is better. A value of 0.35 is about 10.85 Å of mean interface PAE. |
| `i_pDAE` | Distance-masked interface TM confidence, 0–1; higher is better. Uses contacts within 8 Å by default and is BC2's standard ranking score. |
| `i_pTM_detarget`, `i_pAE_detarget` | The corresponding measurements on an explicitly selected off-target. Interpret the desired direction as avoidance rather than binding. |
| `pae`, `PAE_mean` | Pairwise predicted aligned error matrix and its row mean, in Å. These are not the normalized `i_pAE` scale. |
| `plddt`, `ptm`, `iptm` in trajectory records | Prediction-level readings: residue confidence arrays use 0–1, and scalar TM scores use 0–1. Per-chain/stage losses are separate readings. |

`i_pDAE`, `i_pTM`, pLDDT and PAE measure prediction confidence, not affinity. A high score can still describe a biologically inaccessible pose. Review membranes, glycans, full-length target context and the intended assay geometry.

### Contacts, shape and composition

| Measurement | What it measures / how to use it |
| --- | --- |
| `Interface_Residues` | Binder residues with any atom within 4 Å of the target; an interface-size count. |
| `Interface_Residues_detarget` | The same count for a selected off-target. |
| `Interface_<X>_Count` | Contacting binder residues of amino acid X. All 20 one-letter names are available: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y. |
| `Interface_BuriedArea` | Binder-side loss of solvent-accessible area upon complex formation, Å²; not the sum of both partners' buried areas. |
| `Surface_Hydrophobicity` | Fraction of solvent-exposed residues of the free binder that are hydrophobic (A,C,V,I,L,M,F,W,Y). Exposure uses relative SASA ≥0.2. |
| `Backbone_Clashes` | Interchain CA atom pairs within 2.5 Å. It is not an all-backbone or all-atom clash count. |
| `All_Atom_Clashes` | Interchain atom pairs within 2.5 Å. Intrachain clashes are not included. |
| `Binder_Chain_Breaks` | Consecutive CA distances outside 3.3–4.3 Å within binder chains. |
| `Binder_Helix_Fraction`, `Binder_BetaSheet_Fraction`, `Binder_Loop_Fraction` | Fractions of binder residues assigned helix, sheet or other by secondary-structure analysis; 0–1. |
| `Binder_Length` | Total residues over the binder assembly; includes each oligomer copy. |
| `Binder_Mass_kDa` | Sequence-derived molecular mass of the binder assembly, kDa. Does not include unmodelled modifications. |
| `Binder_pI` | Sequence-derived isoelectric point. |
| `Binder_Net_Charge` | Sequence-derived charge at pH 7.4 when scored from structure. If a table lacks it, ranking can estimate K+R−D−E from sequence. |
| `Binder_Extinction` | Estimated molar extinction coefficient at 280 nm, M⁻¹ cm⁻¹, including the geometric cystine estimate. |
| `Binder_Hydrophobic_Fraction` | Hydrophobic fraction of the whole sequence, added during ranking; different from surface hydrophobicity. |
| `Binder_Cysteines` | Cysteine count over the binder assembly. |
| `Binder_Disulfides` | Cysteine pairs with Cβ separation 3.8±1 Å and sequence separation ≥3, counted as a pairing: each cysteine bonds to its closest compatible partner and to nothing else, so three mutually close cysteines are one pair and not three. This geometric count does not establish an S–S bond. |
| `Binder_Free_Cysteines` | `max(0, cysteines − 2 × paired cysteine count)`; inspect actual pairings and sulphur geometry. |
| `Relaxed_Backbone_Clashes`, `Relaxed_All_Atom_Clashes` | Corresponding interchain clash measurements after optional relaxation, where the campaign requested that clash check. |

### Biological design checks

| Measurement | What it measures / how to use it |
| --- | --- |
| `Hotspot_Contact_Fraction` | Fraction of named target hotspots contacted by the binder, 0–1. |
| `Coldspot_Contact_Fraction` | Fraction of named coldspots contacted, 0–1; lower is better for avoidance. |
| `Off_Paratope_Contact_Fraction` | Fraction of contacting binder residues outside the designated paratope, 0–1. |
| `Off_Epitope_Contact_Fraction` | Fraction of contacted target residues outside the protected hotspot neighbourhood, 0–1. |
| `Epitope_Residues_Contacted` | Count of contacted residues in that neighbourhood. |
| `Receptor_Chains_Contacted` | Number of input receptor chains engaged by the binder. |
| `Target_Crop_Length` | Number of target residues in the predicted state, including validation flanks when present. |
| `Cyclic_Closure_Distance` | Distance from C-terminal C to N-terminal N, Å. Inspect orientation as well as distance; a close pair alone does not establish cyclisation. |
| `Scaffold_Sequence_Retained_Fraction` | Fraction of held framework residues retaining the scaffold sequence. |
| `Scaffold_Framework_RMSD` | Aligned framework displacement from the supplied scaffold, Å. |
| `Framework_Packing_Fraction` | Coverage of the non-binding framework by paratope loops; use according to the requested extended/folded-back conformation. |
| `Interdomain_Contact_Fraction` | Contacts between separate domains; lower supports domain separation. |
| `Domain_Separation_Ratio` | Domain-centre separation relative to domain radii; larger supports spatially distinct domains. |
| `Oligomer_Symmetry_RMSD` | Deviation under cyclic permutation of binder copies, Å; smaller is more symmetric. |
| `Protomer_Identity_Fraction` | Agreement between redesigned copy sequences before tying; 0–1. |
| `Binder_RMSD`, `Induced_Fit_RMSD` | Aligned free-to-bound binder displacement, Å. Desired direction depends on whether the binder should retain or change its fold. |
| `Induced_Fit_Interface_RMSD` | Movement of the binding surface after aligning the core, Å; larger supports the requested induced fit. |
| `Induced_Fit_TM` | Free-to-bound whole-fold similarity, 0–1; 1 is unchanged. A switching design asks for a ceiling. |
| `Termini_Distance` | N-to-C terminal CA distance of the addressed binder chain, Å. |
| `Termini_Away_Cosine`, `N_Terminus_Away_Cosine`, `C_Terminus_Away_Cosine` | Direction of both, N or C termini relative to the target: +1 away, −1 toward. |
| `MHC_Anchor_Score` | Combined sequence-based MHC anchor proxy; lower is the humanization objective, not a measured immune response. |
| `Protease_Site_Score` | Cleavage propensity of the discrete sequence under the configured protease panel; not a measured half-life. |
| `Exposed_Loop_Fraction` | Fraction of loop residues that are solvent-exposed (relative SASA ≥0.2); the denominator is loop residues, not all binder residues. |
| `Terminus_Exposure` | Mean relative SASA of terminal residues, by default three at each end of every binder chain. |

### Measurement parameters

Use `filters.<metric>.params` to change a measurement's definition. Changing a cutoff changes the meaning of the reported result; keep it with the campaign metadata. Default state is the appropriate complex, except `Unbound_Binder_pLDDT` and free/bound comparisons. Common parameters are `prediction_state`, `binder`, `target` or `chain`, and `reference_state` for comparison measurements. Scaffold checks automatically receive `scaffold`, `scaffold_edits` and `scaffold_chain` from the campaign; replace them only when deliberately changing the reference.

| Measurement | Additional parameters and defaults |
| --- | --- |
| `Interface_Residues` | `cutoff=4.0` |
| `Interface_Residues_detarget` | `cutoff=4.0` |
| `i_pDAE` | `cutoff=8.0` |
| `Backbone_Clashes` | `cutoff=2.5` |
| `All_Atom_Clashes` | `cutoff=2.5` |
| `Off_Paratope_Contact_Fraction` | `cutoff=4.0` |
| `Hotspot_Contact_Fraction` | `cutoff=4.0` |
| `Coldspot_Contact_Fraction` | `cutoff=4.0` |
| `Interface_Hydrophobicity` | `cutoff=4.0` |
| `Induced_Fit_Interface_RMSD` | `cutoff=8.0`, `interface_residues=()` |
| `Off_Epitope_Contact_Fraction` | `cutoff=4.0`, `epitope_cutoff=EPITOPE_CUTOFF` |
| `Epitope_Residues_Contacted` | `cutoff=4.0`, `epitope_cutoff=EPITOPE_CUTOFF` |
| `Framework_Packing_Fraction` | `cutoff=4.5`, `sequence_separation=20` |
| `Interdomain_Contact_Fraction` | `n_domains=2`, `min_domain_size=50`, `domain_contact_cutoff=8.0` |
| `Domain_Separation_Ratio` | `n_domains=2`, `min_domain_size=50`, `domain_contact_cutoff=8.0` |
| `Binder_Chain_Breaks` | `minimum_bond=3.3`, `maximum_bond=4.3` |
| `Receptor_Chains_Contacted` | `cutoff=4.0` |
| `MHC_Anchor_Score` | `species='human'`, `coupling_weight=0.5`, `hydrophobicity_weight=0.0`, `mhc_class_ii_weight=1.0`, `temperature=0.1` |
| `Terminus_Exposure` | `terminus_length=3` |
| `Binder_Disulfides` | `distance=3.8`, `tolerance=1.0`, `sequence_separation=3` |
| `Interface_Helix_Fraction` | `secondary_structure_code='a'`, `cutoff=4.0` |
| `Interface_BetaSheet_Fraction` | `secondary_structure_code='b'`, `cutoff=4.0` |
| `Interface_Loop_Fraction` | `secondary_structure_code='c'`, `cutoff=4.0` |
| `N_Terminus_Away_Cosine` | `terminus='n'` |
| `C_Terminus_Away_Cosine` | `terminus='c'` |
| `Termini_Away_Cosine` | `terminus='both'` |
| `Interface_<X>_Count` | `cutoff=4.0` |

Distance parameters use Å; sequence separations and terminal lengths count residues. `epitope_cutoff` defines the protected target neighbourhood; `interface_residues` explicitly fixes the moving interface for an RMSD comparison. Domain parameters match the settings reference. Humanization parameters choose the sequence panel and its relative contributions. `terminus` selects `n`, `c` or `both`; the named terminal metrics already select the appropriate choice. Amino-acid count names select their one-letter residue; a custom `amino_acid` parameter can override that selection.

## Trajectory records and viewers

`losses.csv` records `phase`, `round` and scalar `<state>.<metric>` values. `loss` is the aggregate objective; named objective columns correspond to the [loss table](reference.md#losses). They are optimisation readings, not independent acceptance scores. A rotating trajectory writes only the active state on that update, so other state columns can be blank. Plot lines are separated by state; dashed boundaries mark stage changes.

`sequences.npz` contains one array per `<state>.<chain>` for designed chains, shaped as recorded updates × residues × 20 amino acids. It holds the actual mixed/hardened sequence weights supplied during optimisation, not raw logits. Amino-acid order is the model's standard one-letter order; use the viewer to inspect it without reading the archive. Target sequences are not redundantly stored each round. It is written only when `save_design_sequences` is set, off by default, since nothing downstream reads it back.

The HTML viewer shows the structure, pairwise error matrix, confidence history and designed-chain sequence probabilities. State buttons select targets; the frame slider labels stage and update. A free/bound `switch` segment is a morph for viewing the two predictions, not a molecular dynamics trajectory or a kinetic pathway. The viewer needs a browser supporting compressed-stream decoding and internet access for its 3Dmol.js dependency; structures and PNG plots remain usable offline. No animation is produced when fewer than two valid frames survive.

## Ranking and refiltering

```bash
bindcraft rank results/pdl1 --list
bindcraft rank results/pdl1 --on i_pTM
bindcraft filter results/pdl1
bindcraft filter results/pdl1 --where 'i_pAE=0.45' --where 'Interface_Residues>=7'
bindcraft rank results/pdl1/2_Refolded/filtered.csv --on i_pTM
```

Filtering reuses existing candidates. `METRIC=VALUE` follows that metric's configured direction; `>=` and `<=` explicitly choose a direction. A threshold without a target suffix applies to every binding target, excluding detarget states. Candidates with missing required values do not silently pass. The report lists how many candidates each threshold removes; missing columns in the original campaign filter are reported rather than guessed.

| Command option | Use |
| --- | --- |
| `rank --on METRIC` | Select a metric; repeat to break ties using additional metrics. |
| `rank --lowest-first`, `--highest-first` | Override automatic direction, useful for context-dependent geometry. |
| `rank --table accepted` / `candidates` / `trajectories` | Choose the source table; default accepted. |
| `rank --top N`, `filter --top N` | Limit the console display. |
| `rank --output FILE`, `filter --output FILE` | Choose the written CSV. |
| `rank --list`, `filter --list` | List available measurements. |
| `filter --where EXPRESSION` | Add a threshold; repeat for several. |
| `filter --filters FILE` | Read a JSON filter configuration. |
| `filter --table candidates` / `accepted` / `trajectories` | Choose the source table; default candidates. |
| `filter --rejected FILE` | Save rejected rows and the thresholds they failed. |
| `rank --binder CHAINS`, `--target CHAINS`, `--rescore` | Specify roles or rebuild cached scores when ranking a structure folder. |

Multi-target rankings can add `<metric>_mean`, `_worst`, `_best`, `_spread` and `_selectivity`. The first three summarize binding targets; spread is max–min; selectivity is the gap between the weakest binding target and the strongest off-target in the chosen metric's direction. A positive margin favours the intended distinction. These are computational score margins, not affinity ratios. Check direction when comparing conformational measurements; campaign filter directions take precedence over generic ranking defaults.

### Structures from other sources

```bash
bindcraft score my_complex.cif --binder B --target A
bindcraft score my_complex.cif --binder B --target A --hotspots 'A54,A56' --coldspots 'A90-95'
bindcraft rank my_binders --on Interface_BuriedArea --binder B --target A
```

`score` prints a JSON measurement object. It accepts `--binder`, `--target`, `--hotspots` and `--coldspots`. Without explicit roles it treats the first chain as target and the remaining chains as binder (a one-chain file is binder alone). Always specify roles when this does not match your files. A folder ranking caches coordinate-derived measurements in `scored.csv`; `--rescore` recomputes them after changing files or roles. Prediction confidence cannot be reconstructed from coordinates alone; only scores stored in BC2 metadata or per-residue pLDDT can be reused. A default structure-folder ranking falls back to `Binder_pLDDT`, which is inappropriate for true experimental B-factors.

### Rebuild, archive or transfer

```bash
bindcraft campaign_output results/pdl1
bindcraft archive results/pdl1
bindcraft unarchive results/pdl1
```

`campaign_output` rebuilds summaries and rankings; a parent folder containing several campaigns also gets a combined summary. Archiving zips completed trajectory directories; unarchiving restores them. Keep the complete campaign folder to resume. For experimental selection alone, keep ranked structures, tables and metadata, plus any rejected candidates you want to inspect.

## Summaries

`summary.csv` columns are `campaign`, `scope`, `metric`, `samples`, `mean`, `std`, `min`, `max`. `campaign` scope contains counts and `terminated:<stage>` counts; `trajectory` and stage scopes summarize optimisation records; `final` summarizes refolded candidates; `accepted` summarizes retained designs. Per-target summaries append `:<target>` to the scope. `samples` is the number of available readings, not necessarily the number of attempted designs. Standard deviation describes the recorded sample, not uncertainty in experimental performance.

## Sweeps

| Field / file | Interpretation |
| --- | --- |
| `<arm>/` | An ordinary campaign for one parameter choice; compare it using the same output guide. |
| `sweep.csv` | One row per arm, plus the actual values of each swept setting. |
| `arm`, `rank`, `trajectories`, `accepted_designs` | Identity, comparative order and observed counts. |
| `accepted_per_trajectory` | Yield within that arm; sensitive to small samples. |
| `accepted_interface_ptm`, `trajectory_interface_ptm` | Interface-confidence summaries for accepted designs and all recorded trajectories. |
| `paired_trajectories`, `paired_change`, `paired_error` | Matched attempts shared with the baseline, their mean change and uncertainty estimate. |
| `resolved` | Whether the change clears the comparison's evidence rule; false means the leading arm is not established as an improvement. |
| `best_settings.json` | Leading arm, comparison metrics and proposed settings, with `resolved` retained. |

Sweep seeds are paired against a baseline and arms advance in blocks. A stopped campaign can therefore still be compared on matched attempts. The sweep objective is interface-confidence improvement, not binding affinity or biological function; confirm adopted settings in a fresh campaign.
