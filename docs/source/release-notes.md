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

> **Migration:** the objective has three retired spellings and all three now refuse to load, so
> fix them together rather than one failure at a time:
>
> - `weights_induced_fit_global` → `weights_fold_switching`
> - `losses: {"induced_fit_global": ...}`, block or scalar → `losses: {"fold_switching": ...}`
> - `losses.fold_switching.params.binder_shapes` → the top-level `binder_shapes` setting
>
> If you were also setting `induced_fit_tm_target` or `max_induced_fit_tm_final` by hand, see the
> new defaults below. The `fold_switch` modality already sets all of them for you.

**Free-versus-bound comparisons are now masked by confidence.** They previously ran over the whole
binder chain, so a disordered tail flailing between the two predictions counted as conformational
change. They now run only over residues predicted above a 0.7 pLDDT floor. This measures what the
objectives are actually asking about — whether the *structured* part of the binder moved.

**The induced-fit acceptance bar moved with that masking**, from
`min_induced_fit_interface_rmsd_final` 5.0 Å to **2.0 Å**: excluding a flailing tail lowers the
measured displacement, so the previous bar was no longer reachable. The **training target is
deliberately unchanged** — `induced_fit_delta` stays at 5.0 Å — so the objective still optimises
past the bar that accepts it.

The fold-switch ceiling is a separate story, told under *Fold switching* below: the masking briefly
broke the TM-score itself, and the number has been set from what the modality achieves rather than
from compensating for that.

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

### Fold switching: a corrected TM-score, and a displacement to go with it

**The TM-score was being computed on the wrong length, and is fixed.** TM-score takes its distance
scale `d0` from the length of the fold being compared. When conformational comparisons became
confidence-masked, the weight of the confident subset was passed in place of that length. A
62-residue binder with 21 confident residues drove `d0` to its 0.5 Angstrom floor, where a tenth of
an Angstrom reads as a changed fold: one design moving **0.38 A** scored TM 0.70 and was accepted.
Scored on its own length it is **0.98**, which is what the structure shows. `d0` is now taken from
the residue count of the fold, as TM-score defines it, while the score is still averaged over the
confident residues. Reported TM values rise accordingly and are comparable with TM-scores computed
anywhere else.

**`max_induced_fit_tm_final` is 0.85.** The previous 0.75 existed to compensate for the broken scale
and rejects every design once the scale is right. 0.85 is set from what the modality actually
produces: a large conformational change within one fold, rather than a textbook fold switch, which
would need TM below 0.5.

**A displacement is now asked for as well**, in Angstroms, which carries no length-dependent scale:

```
loss = relu(TM - tm_target)^2  +  relu(rmsd_target - RMSD)^2
```

`fold_switch_delta` is the Angstroms of movement the loss optimises toward, named to match the
`induced_fit_delta` it sits beside rather than the design target a campaign binds. It defaults
to **1.0 A** and is set by the `fold_switch` modality, alongside a
matching `min_induced_fit_rmsd_final` of 1.0 gating `Induced_Fit_RMSD` on acceptance. It is guidance
first and a gate second, so trajectories are pushed toward real movement rather than only screened
for it afterwards. Designs already moving more than 1 A are unaffected.

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

Two of those four are acceptance gates that were not applied before, so a campaign that was
accepting designs may now accept fewer. This is the change most likely to alter a file you already
have, because it needs no edit on your part.

It also makes the shipped examples forward-only: twelve of them name no modality and rely on this
default, so carrying one back to 1.0.3 resolves it without any of the four.

The fourteen examples that do need a non-default modality now name it rather than relying on a
`--modality` flag, so running one gives the same result whether or not you pass the flag.
`pdl1_arp.json` now names `ARP`, which means it finally gets the aromatic downweighting its
description has always promised.

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

Order no longer decides anything between two formats either. Where both set the same thing, the
winner is fixed: a **framework** (`ARP`, `Fab`, `scFv`, `VHH`) beats a **chain kind** (`peptide`,
`cyclic_peptide`, `large_binder`), which beats an **assembly** (`homo_oligomer`, `multidomain`),
which beats **`binder`** — the fallback, which now yields to anything else named. So
`["homo_oligomer", "peptide"]` designs 12–25 residues per copy whichever way round it is written.
`multidomain` is exempt for one setting, `weights_binder_contacts`, which it zeroes deliberately.

### Unsupported combinations are refused at campaign start

Combinations that cannot mean anything coherent now stop the campaign instead of silently resolving
to whichever was written last. Two scaffolds — `["ARP", "VHH"]` — are refused, because each brings
its own framework and a campaign designs one binder. So are the pairs that name two different
things for one chain to be: `peptide` with `cyclic_peptide`, `large_binder` or `multidomain`, and
`cyclic_peptide` with `large_binder` or `multidomain`. The full compatibility table is in
[Choosing a modality](design-guide/03-choosing-a-modality.md).

### Cyclic peptides draw the full 7–20 residues their description promises

`cyclic_peptide` said "A 7-20 residue peptide" and drew 7–16, so the top of its own stated range was
unreachable. The ceiling now matches the description. Name `binder_lengths` yourself for any other
range.

### Examples no longer restate what their modality already says

Every shipped example now keeps only the settings that differ from the presets it resolves
through. Fourteen name a modality; the other twelve take the `binder` default described above.
That removed 146 redundant keys across 21 files, and `pdl1_multidomain.json` went from 38 keys
to 9. The scaffold examples no longer carry a second copy of the `mutate_positions` string their
preset already supplies.

No example resolves differently as a result of the cleanup; the files are shorter, not different.
The one example whose behaviour did move, `pdl1_cyclic_peptide.json`, moved because of the length
range above and not because of this. If you copied an example as a starting point, the keys you
see now are the ones that example is actually making a decision about.

### One fold-switching example, not two

`pdl1_induced_fit_global.json` and `pdl1_fold_switching.json` are both replaced by a single
**`pdl1_fold_switch.json`**, writing to `results/pdl1_fold_switch`. The two differed in three loss weights and in the fold-switching `tm_target`: the retired
`pdl1_fold_switching.json` asked 0.45, pushing for a larger fold change, where the surviving
example takes the modality's 0.6. Acceptance is unchanged either way — a TM-score ceiling of
0.85 and at least 1 Angstrom of movement — so the two always accepted the same designs. Its `induced_fit_monomer_*` settings were dropped:
those drive the binder-alone block, which only an `induced_fit_interface` objective reaches, so
under `fold_switch` they did nothing. Its explicit `binder_shapes` were dropped too — the
schedule derives the same free/bound pair on its own.

### Smaller behaviour changes

- **GPU selection accepts UUIDs.** `gpu_ids` still takes ordinals (`"0,1"`), and now also takes
  CUDA UUIDs. Asking for a GPU that is not visible to the process now raises a clear error instead
  of silently selecting the wrong card.
- **A lost design worker is now reported and surfaces in the exit status.** A worker killed by a
  signal returns 128 + the signal number, so a shell or scheduler can see the campaign failed.
- The binder-alone diagnostic logged as `global_tm` is now `induced_fit_tm`. It is the same
  number as the `Induced_Fit_TM` column, and only induced fit prints it, which is what the
  old name obscured: `global` named a retired feature and `fold` names the one objective
  that never reaches this line.
- The package reports `1.0.4`. `main` had been reporting `1.0.1` through the v1.0.2 and v1.0.3
  releases.

### Outputs are stamped with the version of the code that wrote them

`bindcraft_version` was read from installed package metadata, which is a snapshot taken when the
package was installed. A source checkout on `PYTHONPATH` shadows whatever wheel is installed, so a
1.0.4 tree running over a 1.0.0 install stamped **`BindCraft 2 v1.0.0`** onto every structure, every
CSV row and every campaign record. It now reads the version declared beside the code that is
running, and falls back to installed metadata for a wheel with no source tree beside it.

`bindcraft_revision` was already correct — it carries the commit, with `-dirty` when the tree has
uncommitted changes — so a structure written before this fix can still be traced by its revision
even though its version is wrong.

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
