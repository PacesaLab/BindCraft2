# Release notes

[Design Overview](design-guide.md) · [Reference Documentation](reference.md) · [Outputs and Measurements](outputs.md)

## 1.0.4

This release reworks how conformational change is specified and measured, and makes a campaign
file's modality explicit. Both change the results you get from existing settings files, so read
the first two sections before re-running an older campaign.

### Conformational design: one objective, measured on the confident fold

**`induced_fit_global` is gone.** The whole-fold objective is now `fold_switching`, which compares
a prediction state against a reference state directly instead of going through the induced-fit
machinery. `weights_induced_fit_global` has been removed from the settings vocabulary, so a file
or `--set` naming it is now **rejected outright** rather than quietly ignored.

> **Migration:** replace `weights_induced_fit_global` with `weights_fold_switching`. If you were
> also setting `induced_fit_tm_target` or `max_induced_fit_tm_final` by hand, see the new defaults
> below. The `fold_switch` modality already sets all three for you.

**Free-versus-bound comparisons are now masked by confidence.** They previously ran over the whole
binder chain, so a disordered tail flailing between the two predictions counted as conformational
change. They now run only over residues predicted above a 0.7 pLDDT floor. This measures what the
objectives are actually asking about — whether the *structured* part of the binder moved.

**Because of that masking, the acceptance thresholds were recalibrated.** The same physical motion
now reports a smaller number than it did before, so the bars moved with it:

| Filter | Before | Now |
| --- | --- | --- |
| `min_induced_fit_interface_rmsd_final` | 5.0 Å | **2.0 Å** |
| `max_induced_fit_tm_final` | 0.6 | **0.75** |

These are not a loosening. The previous bars were unreachable in practice once a confidently
predicted fold is what gets measured. The **training targets are deliberately unchanged** —
`induced_fit_delta` stays at 5.0 Å and `induced_fit_tm_target` at 0.6 — so each objective still
optimises past the bar that accepts it.

> One consequence worth knowing: the adaptive early exit for the binder-alone block compares
> against the *training target*, not the acceptance bar. A design that already clears the 2 Å
> filter will keep refining until it reaches 5 Å or exhausts `induced_fit_monomer_steps`.

**`Binder_RMSD` and `Induced_Fit_RMSD` are now separate measurements.** They previously shared one
body, which meant tuning one silently moved the other — and they ask opposite questions of the
same comparison:

- `Binder_RMSD` asks *does the binder hold its fold?* It counts the whole chain, flailing tail
  included, because a tail that moves is evidence the fold is not held.
- `Induced_Fit_RMSD` asks *did the fold really move?* It discounts low-confidence residues, because
  a flailing tail is not conformational change.

### Campaigns without a modality now load the `binder` preset

A campaign file that names no `modality` previously loaded **no modality layer at all**, despite
the CLI documenting `binder` as the default. Those campaigns silently ran without `binder.json`'s
`Binder_RMSD` stability gate.

Naming nothing now loads `binder`, as documented. If you have a settings file with no `modality`
key, it will pick up `binder_lengths` of 60–180, `aa_bias {"C": 0}`, `min_monomer_plddt_final` of
0.7 and a `Binder_RMSD` ceiling of 3.5 Å — any of which your own file still overrides, because the
campaign file is applied last.

> **Check your file if it designs with cysteine.** The `binder` preset excludes cysteine. If your
> campaign needs disulfides, name the `disulfide_staple` property, which re-allows it. The shipped
> `pdl1_disulfide.json` now does exactly this.

Shipped examples now name their own modality rather than relying on a `--modality` flag, so running
an example gives the same result whether or not you pass one. `pdl1_arp.json` now names `ARP`,
which means it finally gets the aromatic downweighting its description has always promised.

### Modality order no longer changes the result

Modalities were applied strictly in the order written, last one wins. That made the order
load-bearing in a way nothing documented: `induced_fit` and `fold_switch` switch the `Binder_RMSD`
gate *off*, because it measures exactly the displacement they exist to create. Naming `binder`,
`homo_oligomer` or `large_binder` **after** them put the gate back at 3.5 Å, so every design that
achieved the objective was rejected by the filter — the campaign burned its full `max_trajectories`
and finished with nothing accepted and no error.

```json
"modality": ["fold_switch", "binder"]    // before 1.0.4: gate back on, zero designs
```

Binder formats are now always applied before conformational objectives, whichever order you write,
so both spellings resolve identically and an objective can never overwrite the format it qualifies.
Ordering among formats of the same kind is unchanged — `binder_lengths` still comes from whichever
you name last.

### Unsupported combinations are refused at campaign start

Combinations that cannot mean anything coherent now stop the campaign instead of silently producing
something else. Naming two scaffolds — `["ARP", "VHH"]` — is one: each brings its own framework and
a campaign designs one binder, so the last named used to win and the rest were dropped without a
word. The full compatibility table is in
[Choosing a modality](design-guide/03-choosing-a-modality.md).

### Cyclic peptides default to 6–20 residues

`cyclic_peptide` now draws lengths of 6–20, matching its description. It previously said 7–20 and
drew 6–16. Note this differs from the 7-residue floor set upstream; name `binder_lengths` yourself
if you want a different range.

### Two shipped examples had filters that contradicted their presets

`pdl1_humanization.json` and `pdl1_protease_stability.json` carried inline thresholds stricter than
the property presets they are built from, so they rejected designs the property considers good:

- `max_mhc_anchor_score_final` 0.5 → **1.95**
- `max_exposed_loop_fraction_final` 0.35 → **0.7**

### Renamed example

`pdl1_induced_fit_global.json` is now **`pdl1_fold_switch_whole.json`** and writes to
`results/pdl1_fold_switch_whole`. Its `induced_fit_monomer_*` settings were dropped: those drive
the binder-alone block, which only an `induced_fit_interface` objective reaches, so under
`fold_switch` they did nothing.

### Smaller behaviour changes

- **GPU selection accepts UUIDs.** `gpu_ids` still takes ordinals (`"0,1"`), and now also takes
  CUDA UUIDs. Asking for a GPU that is not visible to the process now raises a clear error instead
  of silently selecting the wrong card.
- **A lost design worker is now reported and surfaces in the exit status.** A worker killed by a
  signal returns 128 + the signal number, so a shell or scheduler can see the campaign failed.
- The binder-alone diagnostic logged as `global_tm` is now `fold_tm`.
- The package reports `1.0.4`. `main` had been reporting `1.0.1` through the v1.0.2 and v1.0.3
  releases.

### Known issues

- **Use one target with `fold_switch`.** A fold-switching campaign that names several targets
  runs rather than being refused, but the mutate stage fixes a single target, so the
  `fold_switching` losses are dropped for that stage and the conformational objective silently
  stops applying. `induced_fit` still refuses more than one target outright.
- Several examples still restate much of the preset they now name — `pdl1_multidomain.json`
  repeats 26 of its preset's keys, `pdl1_cyclic_peptide.json` 10 — with identical values. Because
  the campaign file is applied last, those copies win, so editing the preset alone will not change
  those examples. Edit the example, or delete the redundant keys from it first. (Where an example
  deliberately narrows a preset — a tighter `binder_lengths`, a heavier `weights_fold_switching` —
  that is the layering working as intended.)
