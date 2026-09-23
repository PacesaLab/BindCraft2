# Example campaigns

[Back to the README](../README.md) · [Modality settings](../README.md#design-modalities) · [Compatibility chart](../README.md#combining-modalities) · [Full reference](../docs/reference.md)

Start with [pdl1.json](pdl1.json) to use named presets. It names the shipped [hPDL1 target](../settings/target/hPDL1.json) and the `binder` modality, and requests 10 accepted designs and sets an output folder. BC2 supplies the detailed design settings and filters.

Run from the repository root, with the BC2 environment active and a GPU available:

```bash
bindcraft design examples/pdl1.json
```

Choose a VHH scaffold and the humanization property:

```bash
bindcraft design examples/pdl1.json --modality VHH --humanize --set 'project_folder=results/pdl1_vhh'
```

For your own campaign, copy the small file within this directory and edit the target it names, its modality and its output folder. `bindcraft design --list-targets` names every target BC2 ships; for one it does not, write your own `targets` entries in place of the name:

```bash
cp examples/pdl1.json examples/my_target.json
```

```bash
bindcraft design examples/my_target.json
```

`campaign_name` is what the campaign calls itself, and it leads the name of every design it writes, so designs from several campaigns share a folder and still sort by the campaign that made them. It is usually what the binder is called, and writing the same thing under `binder_name` as well is read as one name rather than two. A campaign against more than one target closes each complex it writes with the target that complex holds, so an accepted ortholog-pair design leaves `_hPDL1` and `_mPDL1` behind.

A named target ships its own structure under [settings/target/](../settings/target/). Target and custom scaffold paths you write yourself are read from the JSON file’s directory. Running the quickstart from the repository root writes to `results/pdl1/`, as set by `project_folder`. Named scaffold presets resolve their own files under `scaffolds/`. `number_of_final_designs` counts accepted designs. The quickstart has no attempt limit; set `max_trajectories` only if you want one. `metadata.json` is a separate optional input for author or project fields, supplied with `--metadata`.

The other examples below demonstrate explicit settings for particular experiments. **Their explicit values override any presets you add**, including lengths, amino-acid preferences and filters. Use the small file when switching formats through `--modality`; consult a detailed example when you want to customise that experiment. See [input tiers and precedence](../docs/reference.md#input-tiers-and-overrides).

## Choose an example

| File | Modality | What it requests |
| --- | --- | --- |
| [pdl1.json](pdl1.json) | Preset-based quickstart | The `hPDL1` target plus the `binder` preset; use this file with modality and property flags. |
| [pdl1_denovo.json](pdl1_denovo.json) | De novo miniprotein | One structured target; 60–100 residue binders. |
| [pdl1_ortholog_pair.json](pdl1_ortholog_pair.json) | Multitargeting | One sequence against the `hPDL1` and `mPDL1` targets, each with its own hotspots. |
| [pdl1_detarget_pd1.json](pdl1_detarget_pd1.json) | Detargeting | The `hPDL1` target with the `hPD1` off-target; explicit detarget confidence ceilings. |
| [il7ra_focused_epitope.json](il7ra_focused_epitope.json) | Forced targeting and coldspots | Drive contact onto a named ten-residue IL-7Rα patch and require the accepted design to touch it; the rest of the exposed face is kept free by the target's own coldspots. |
| [dynorphin_idr.json](dynorphin_idr.json) | Disordered target | Dynorphin A(1–13) from FASTA; a fixed 13-residue target window. |
| [il2_receptor.json](il2_receptor.json) | Multi-chain receptor | The IL-2 receptor beta/gamma assembly; require contacts to both chains. |
| [pdl1_vhh.json](pdl1_vhh.json) | Fold conditioning | A VHH scaffold with editable CDRs, variable loop lengths and defined framework positions. Aromatics are downweighted so the CDRs do not fill with tryptophan. |
| [pdl1_arp.json](pdl1_arp.json) | Fold conditioning | An ARP (consensus ankyrin-repeat protein) scaffold: the randomised repeat positions are edited, four of them resizable, and the ankyrin framework is held. Aromatics are downweighted so the repeats do not fill with tryptophan. |
| [pdl1_scfv.json](pdl1_scfv.json) | Fold conditioning | An scFv scaffold: the heavy and light variable domains as the two chains they are, with no linker between them, and a resizable heavy-chain CDR3. Aromatics are downweighted so the CDRs do not fill with tryptophan. |
| [pdl1_peptide.json](pdl1_peptide.json) | Peptide | A linear peptide of 12–25 residues, validated on the multimer pool, with the fold confidences recorded rather than filtered. |
| [pdl1_cyclic_peptide.json](pdl1_cyclic_peptide.json) | Cyclic peptide | Head-to-tail closure geometry for peptides of 7–20 residues. |
| [pdl1_homotrimer.json](pdl1_homotrimer.json) | Homo-oligomer | Three identical protomers of 40–60 residues each. |
| [pdl1_multidomain.json](pdl1_multidomain.json) | Multidomain | Two domains on one chain, with size, separation and linker objectives. |
| [pdl1_mixed_topology.json](pdl1_mixed_topology.json) | Mixed topology | Apply a 50% helix ceiling and a 20% beta-sheet floor. |
| [pdl1_induced_fit_interface.json](pdl1_induced_fit_interface.json) | Induced fit at the interface | Encourage at least 5 Å of predicted interface movement between free and bound states. |
| [pdl1_induced_fit_global.json](pdl1_induced_fit_global.json) | Global induced fit | Encourage a whole-fold difference between predicted free and bound states, with a TM-score ceiling of 0.6. |
| [pdl1_fold_switching.json](pdl1_fold_switching.json) | Fold switching | Assign different intended binder shapes to the PD-L1-bound and binder-alone states. |
| [pdl1_disulfide.json](pdl1_disulfide.json) | Disulfides | Allow cysteines, favour pairing and require at least one geometrically detected disulfide. |
| [pdl1_humanization.json](pdl1_humanization.json) | Humanization | Apply sequence preferences and an MHC-II anchor-score ceiling. |
| [pdl1_protease_stability.json](pdl1_protease_stability.json) | Protease resistance | Penalise cleavage motifs, exposed-loop proxies and exposed termini. |
| [pdl1_termini_distance.json](pdl1_termini_distance.json) | Nearby termini | Pull the chain ends together and require an end-to-end distance of at most 10 Å. |
| [pdl1_termini_orientation.json](pdl1_termini_orientation.json) | Accessible termini | Orient both chain ends away from the target. |
| [pdl1_seeded_design.json](pdl1_seeded_design.json) | Seeded de novo design | Start the gradient stages from a binder sequence you already have rather than from noise, at the length of that sequence. |
| [pdl1_mpnn_redesign.json](pdl1_mpnn_redesign.json) | Redesign a given sequence | Start from a binder sequence instead of designing one: no gradient stages, 20 unconstrained ProteinMPNN redesigns of its fold, each refolded and scored. |
| [pdl1_mpnn_redesign_max2.json](pdl1_mpnn_redesign_max2.json) | Redesign, at most 2 substitutions | The same, with `redesign_max_positions: 2` so every candidate is a double mutant of the sequence given, and 40 candidates for coverage. |
| [pdl1_mpnn_redesign_evaluation.json](pdl1_mpnn_redesign_evaluation.json) | Evaluate given sequences | `redesign_max_positions: 0` generates nothing: a parent and four point mutants are each folded, refolded by the validation ensemble and scored against the full battery. |
| [pdl1_parameter_sweep.json](pdl1_parameter_sweep.json) | Parameter sweep | Split 150 trajectories over a baseline and the four arms `"parameter_sweep": true` implies: binder helicity over -0.6 and 0.3, and the interface-contact weight at half and at double, at one pinned binder length so the arms run the same trajectories. |

Named presets supply objectives and acceptance thresholds together. When customising the detailed examples, preserve the checks that correspond to your biological objective; changing a weight alone does not necessarily require an accepted design to have that property. Share thresholds use fractions: `0.5` means 50%, and `0.2` means 20%. For scaffold edits, match the chain letters and residue numbering of the scaffold you actually use.

Acceptance is computational. These examples do not specify expected experimental success rates, and the recorded confidence, geometry and sequence-propensity metrics do not establish affinity or biological function.
