# What to look out for (common pitfalls)

[Full guide](../design-guide.md) · [Settings](../reference.md) · [Outputs](../outputs.md)

- **Prepare the target.** Strip waters/ligands you don't want, keep the biologically relevant
  assembly, and make sure the epitope you name is actually solvent-exposed in that structure. BC2
  designs against what you give it, membrane and glycan context included or not.
- **Hotspots are optional but steer the campaign.** BC2 runs fine with none — it reads the whole
  surface and finds a site. Name 3–6 exposed residues when you care *where* the binder lands (a
  specific functional epitope, or a large target where you want to focus the budget); omit them to let
  it choose. If it binds but off-target, add `forced_targeting`.
- **Even a great score isn't a binder.** Treat the ranked list as *candidates to test*, not answers.
  Confidence metrics rank designs against each other; they do not predict wet-lab success.
- **Watch the `autotuned` column.** Designs accepted on the desperation ladder are weaker; a campaign
  that only produced them is telling you the task is too hard as posed.
- **Zero accepted designs is information.** Read `failed_filters` in `2_Refolded/!_Refolded.csv`. If
  everything fails `i_pTM`/`i_pAE`, the epitope may be undruggable or mis-chosen; if it fails
  `Unbound_Binder_pLDDT`, the binders bind but don't fold on their own (try a different length or
  modality).
- **Custom antibody/ARP scaffolds need correct numbering.** The engine rejects Kabat insertion
  codes — use the shipped scaffolds unless you have sequentially renumbered your own.
- **Multi-target and detargeting.** You can supply several targets (weighted) and mark off-targets
  with `"objective": "detarget"` to design for specificity; read the `_detarget` metrics as
  *avoidance*, not binding.
- **Reproducibility.** `campaign_seed` fixes the draws within one setup but does not guarantee
  identical numbers across machines/GPUs. A campaign **resumes by default** — rerun the same command
  against the same folder to continue it.
