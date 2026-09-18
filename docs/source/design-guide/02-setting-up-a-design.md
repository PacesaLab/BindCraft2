# Setting up a design

[Full guide](../design-guide.md) · [Settings](../reference.md) · [Outputs](../outputs.md)

A campaign is a small JSON file, a simple example is:

```json
{
  "campaign_name": "my_pdl1_binders",
  "project_folder": "results/pdl1",
  "modality": "binder",
  "targets": [
    { "name": "PDL1", "target_path": "PDL1.pdb", "chains": "A", "hotspots": "A54,A56,A66,A115" }
  ],
  "binder_lengths": [60, 100],
  "number_of_final_designs": 10,
  "max_trajectories": 2000
}
```
You can find this example in `examples/pdl1_custom_target.json`

Run it from the root directory with:

```bash
bindcraft design examples/pdl1_custom_target.json
```

These are only specifications your JSON file *must* have, however there are many more you may want to use to customize your run. 

```{note}
Shipped example targets (hPDL1 and PD-L1) need no path:
Using `"target": "hPDL1"` reuses a prepared structure that ships with BC2 with its
hotspots already chosen. 
`bindcraft design --list-targets --list-modalities --list-properties`
shows what targets come with a BindCraft2 installation.
```

The settings that matter most when you set up a run:

| Setting | What it does | Advice |
| --- | --- | --- |
| `targets[].target_path` | The path and file name where your target structure can be found. | Prepare it first (see [What to look out for](07-common-pitfalls.md)). Multi-character chain names are fine. |
| `targets[].chains` | The structure and which chains are the target | Multiple chains can be selected via `"chains": "A,B"`. |
| `targets[].hotspots` | The residues you want contacted, e.g. `A54,A56,B12-16` | **Optional but high-leverage.** BC2 works fine with none given — it then reads the whole surface and finds its own site. Name hotspots when you care *where* it binds, or the target is large and you want to steer it; leave them off to let it discover a site. |
| `binder_lengths` | `[80,80]` fixes 80; `[60,100]` is a range; `[60,80,100]` is a choice list | Smaller binders are easier but bury less interface; longer binders reach flatter epitopes. The modality sets a sensible default range. |
| `number_of_final_designs` | How many accepted designs to collect | Order 10–100 for a real campaign; a few for a smoke test. |
| `max_trajectories` | Attempt budget | The safety cap. A hard target may need thousands of attempts per design. |
| `modality` | Binder format and objective (see [Choosing a modality](03-choosing-a-modality.md)) | Pick this to match what you want to make. |

```{important}
Everything else has a defensible default. Resist the urge to tune weights and change presets on your first design run. See what the default settings produce, and then adjust accordingly.
```

## Target inputs — structured, disordered, and multiple

BC2 designs against whatever you put in `targets[].target_path`, and it accepts three kinds of input.

**Structured domain (PDB or mmCIF).** The normal case — a folded domain with coordinates. Select the
chains with `chains`, name the epitope with `hotspots`, and residues to keep clear with `coldspots`
(both use your structure's numbering). BC2 designs against exactly that conformation, so give it the
biologically relevant assembly (see [What to look out for](07-common-pitfalls.md)).

**Disordered region or motif (FASTA).** If `target_path` is a FASTA **sequence**, the target has no
structure, so BC2 treats it as an intrinsically disordered region (IDR) and **co-folds it with the
binder** — this is how you bind disordered proteins, peptide motifs and linear epitopes. Because a
long IDR has no single fold, BC2 doesn't use the whole sequence at once:
- `crop_fasta_sequence` (default `[10,40]`) sets the length of the sequence **window** sampled each
  trajectory; different trajectories see different windows, so the campaign scans along the sequence.
  `false` uses the full sequence.
- `idr_crop_count` (default 1) treats several windows as separate target states at once.
- `validation_crop_flank` (default 5) restores up to five residues on each side of the window at
  validation, so a design isn't leaning on the artificial cut ends; `min_target_crop_length_final`
  (metric `Target_Crop_Length`) requires enough coverage.
- A FASTA target carries no residue numbers or backbone, so `hotspots`, `coldspots` and
  `forced_targeting` don't apply to it.

**Multiple targets (one `targets` list).** List more than one target object to design a single binder
against several structures at once. Each object carries a `weight`, and **the weight does both jobs**:
its **magnitude** sets relative importance, and its **sign** sets the goal — a **positive** weight (the
default, `1`) means *bind*, a **negative** weight means *avoid*. `"objective": "detarget"` is simply an
explicit alias for a negative weight (BC2 forces the weight negative when you set it); `"objective":
"target"` is the default. So the default target is **positive/binding**, and you detarget by making the
weight negative (or setting `objective: "detarget"`).
- **Positive targets** (default) — the binder must bind **all** of them: one **cross-reactive /
  multi-specific** binder, each target's weight setting its pull in the shared objective (and the order
  of the per-target values in the result CSV cells). One binder is redesigned against all of them
  together (`multitarget_tied_redesign`).
- **Off-targets** (negative `weight`, or `"objective": "detarget"`) — the binder is actively
  **repelled** for **specificity**: bind the target, miss the paralog. A named preset such as `hPD1`
  may already carry this. Off-targets are visited on a rotation and pushed down until their interface
  confidence drops below `max_detarget_iptm` (0.4); a design is rejected unless it holds few enough
  off-target interface residues (`max_detarget_interface_residues_final`).

Under the hood a multi-target trajectory **rotates** through the target slots, spending update steps on
each in turn — positive targets pulling the binder on, off-targets pushing it off — then merges what it
learned into the one shared binder sequence. Read the `_detarget` metrics as *avoidance*, not binding,
and remember a computational miss is not a guarantee of experimental specificity.

*Example — one binder that engages both human and mouse PD-L1 but avoids human PD-1:*

```json
{
  "campaign_name": "crossreactive_pdl1",
  "project_folder": "results/pdl1_xreact",
  "modality": "binder",
  "binder_lengths": [70, 100],
  "number_of_final_designs": 20,
  "targets": [
    { "name": "hPDL1", "target_path": "hPDL1.pdb", "chains": "A", "hotspots": "A54,A56,A66,A115", "weight": 1.0 },
    { "name": "mPDL1", "target_path": "mPDL1.pdb", "chains": "A", "hotspots": "A54,A56,A66,A115", "weight": 1.0 },
    { "name": "hPD1",  "target_path": "hPD1.pdb",  "chains": "A", "weight": -0.5 }
  ]
}
```

Reading it:
- **`hPDL1` and `mPDL1` at `weight: 1.0`** — both positive and equal, so the single binder is designed
  to bind **both** orthologs with equal importance (a cross-reactive anti-PD-L1 binder). Hotspots are
  given per target, each in that structure's own numbering, so you can aim the same epitope on both.
- **`hPD1` at `weight: -0.5`** — the negative sign makes it an **off-target** the binder is pushed
  *away* from (specificity against PD-1), and the magnitude `0.5` means that repulsion is applied at
  half the strength of the binding objective. `"weight": -0.5` and `"objective": "detarget"` with a
  weight of `0.5` are equivalent.
- In the result tables, per-target metrics (`i_pTM`, `i_pAE`, …) appear as semicolon-separated cells
  ordered by weight — the two PD-L1 targets first, then the PD-1 `_detarget` reading, which you read as
  avoidance. A design is accepted only if it binds both PD-L1s and stays under the detarget ceilings on
  PD-1.

Swap the explicit `target_path` entries for shipped names (`"target": ["hPDL1", "mPDL1"]`) when the
targets are presets; add or drop off-targets to trade breadth against specificity.
