# De novo binder design with BindCraft2

BindCraft2 (BC2) designs protein binders against a target of your choosing. You give it a
structure (or sequence) and, ideally, the patch you want the binder to sit on; it hallucinates
candidate binders with AlphaFold gradients, redesigns their sequence with ProteinMPNN,
re-predicts each candidate from scratch to check it holds up, filters, and ranks what survives.

This page explains what actually happens during a run, which knobs matter for everyday designs,
how to pick a modality, and how to read the output so you can tell a promising design from a
number that only looks good. It is the BC2 companion to the original
[BindCraft wiki](https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft);
the biology intuition carries over, but the settings, modalities and outputs below are BC2's.

> **One-line mental model:** a campaign keeps *trying* trajectories until it *accepts* enough
> designs. A trajectory is one hallucinated binder; a design is an accepted, re-predicted,
> filter-passing sequence. You ask for N final designs and a trajectory budget; it runs until it
> has N or runs out of budget.

---

## 1. How a campaign runs

Every accepted design has been through four stages. Understanding them tells you what each output
folder means and where a design can fail.

**1. Trajectory (hallucination).** AlphaFold is run in reverse: the binder sequence is optimised by
gradient descent so the predicted complex looks like a good binder. This happens in ordered stages
— `screen → refine → anneal → harden → mutate` — that move from soft, exploratory sequences to a
hard, single amino-acid sequence. Each stage has a confidence floor; a trajectory that can't meet it
is dropped early. The structure the trajectory ends on is the *hallucinated* binder — it is **not**
yet a real prediction, because AlphaFold was being pushed toward it.

**2. Redesign (ProteinMPNN).** The hallucinated backbone is handed to ProteinMPNN, which draws
several new sequences for it (default 10 candidates). This washes out AlphaFold-specific sequence
quirks and gives sequences that a structure-prediction-free model believes fold to that backbone.

**3. Validation refold.** Each redesigned sequence is predicted **from scratch** by AlphaFold (by
default an ensemble of 2 held-out monomer models), with no gradient pushing it. This is the honest
test: does the sequence actually fold and bind on its own? The filters are applied here.

**4. Ranking.** Sequences that pass every filter are accepted, written to `3_Ranked/`, and ranked
best-first by **`i_pDAE`** (a distance-masked interface confidence, higher is better). This is the
list you actually inspect.

Design-time metrics (from stage 1) are optimistic by construction. **Trust the validation numbers
(stage 3), not the trajectory numbers.**

### Starting from a binder you already have

If you already have a binder sequence — an accepted design, a hit from the bench, a panel of point
mutants — you can skip stage 1 entirely. Write the sequences under `binder_sequences` and add
`mpnn_redesign: true`:

```json
{
  "target": "hPDL1",
  "mpnn_redesign": true,
  "binder_sequences": { "parent": "SAEMKEVEEKFEKVKKAIE..." },
  "redesign_max_positions": 2,
  "redesign_interface": true,
  "sequence_candidates": 40
}
```

Each sequence is folded once and judged on the campaign's `_final` filters, then stages 2–4 run
exactly as above: MPNN draws candidates off that fold, each is refolded from scratch and scored, and
the survivors are ranked. `redesign_max_positions` decides how far a candidate may move — leave it out
for unconstrained redesigns, set `2` for double mutants, or set `0` to fold and score the sequences you
gave and generate nothing at all. `redesign_interface: true` is worth adding, as above: without it the
interface of the sequence you gave is held and every substitution lands elsewhere. `Binder_Mutations` in
the output tables says how far each candidate moved from its parent. See [Models and sequence redesign](reference.md#models-and-sequence-redesign).

Leaving `mpnn_redesign` out but keeping `binder_sequences` is a different, also useful thing: the gradient
stages run in full, starting from the sequence you gave instead of from noise, as
[pdl1_seeded_design.json](../examples/pdl1_seeded_design.json) does. Nothing holds a design near that
seed unless you set `redesign_max_positions`, so treat it as de novo design with a head start, and read
`Binder_Mutations` to see how far it travelled. Either way the seed fixes the length, so writing
`binder_lengths` alongside `binder_sequences` is refused, as is a scaffold modality, rather than one of
them being quietly ignored. A length a modality preset happens to carry is not refused — the seed simply
decides the length instead — so `--modality binder` stays usable with a seed.

---

## 2. Setting up a design

A campaign is a small JSON file. The minimum is a target, a binder length range, and how many
designs you want:

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

Run it with:

```bash
bindcraft design my_campaign.json
```

Shipped example targets need no path — `"target": "hPDL1"` reuses a prepared structure with its
hotspots already chosen. `bindcraft design --list-targets --list-modalities --list-properties`
shows what ships.

The settings that matter most when you set up a run:

| Setting | What it does | Advice |
| --- | --- | --- |
| `targets[].target_path` / `chains` | The structure and which chains are the target | Prepare it first (see §7). Multi-character chain names are fine. |
| `targets[].hotspots` | The residues you want contacted, e.g. `A54,A56,B12-16` | **Optional but high-leverage.** BC2 works fine with none given — it then reads the whole surface and finds its own site. Name hotspots when you care *where* it binds, or the target is large and you want to steer it; leave them off to let it discover a site. |
| `binder_lengths` | `[80,80]` fixes 80; `[60,100]` is a range; `[60,80,100]` is a choice list | Smaller binders are easier but bury less interface; longer binders reach flatter epitopes. The modality sets a sensible default range. |
| `number_of_final_designs` | How many accepted designs to collect | Order 10–100 for a real campaign; a few for a smoke test. |
| `max_trajectories` | Attempt budget | The safety cap. A hard target may need thousands of attempts per design. |
| `modality` | Binder format and objective (see §3) | Pick this to match what you want to make. |

Everything else has a defensible default. Resist the urge to tune weights and stage lengths on your
first campaign.

### Target inputs — structured, disordered, and multiple

BC2 designs against whatever you put in `targets[].target_path`, and it accepts three kinds of input.

**Structured domain (PDB or mmCIF).** The normal case — a folded domain with coordinates. Select the
chains with `chains`, name the epitope with `hotspots`, and residues to keep clear with `coldspots`
(both use your structure's numbering). BC2 designs against exactly that conformation, so give it the
biologically relevant assembly (see §7).

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

---

## 3. Choosing a modality

The **modality** sets the binder format and the conformational objective. Name it with `"modality":
"..."` (or `--modality`). You can combine some, e.g. `["VHH", "induced_fit"]`.

| Modality | What it makes | Choose it when |
| --- | --- | --- |
| `binder` | A de novo miniprotein, ~60–180 residues, folded from nothing | The default and most reliable. General de novo binder against a structured epitope. |
| `large_binder` | A longer de novo binder (~250–600 residues) with extended optimisation | Chiefly to add **mass to a small target for cryo-EM / structural biology**; also for large or flat epitopes that need more interface. **Name `binder_lengths`** (see `bigbang` in §5 for large complexes). |
| `peptide` | A 12–25 residue linear peptide, judged bound without a free fold | A short linear peptide into a **groove or pocket** — not a flat surface (see intuition below). |
| `cyclic_peptide` | A 6–16 residue head-to-tail cyclic peptide | Cyclic peptide binders; the ring pre-pays part of the binding entropy. Read `Cyclic_Closure_Distance`. |
| `homo_oligomer` | Identical copies of one 40–120 residue chain (lengths are per copy) | A symmetric homo-oligomeric binder — and the natural choice for a **symmetric target** (e.g. homotrimeric TNFα) a single-chain binder struggles with. Set `copies`. |
| `multidomain` | Two domains on one 120–300 residue chain | A two-domain binder with separation/linker objectives. |
| `VHH` | Single-domain antibody scaffold, editable CDRs; samples extended and folded-back CDR3 | Single-domain antibody (VHH) format (see conformation note below). |
| `scFv` | Heavy + light variable domains as two chains, no linker designed | scFv-format binders. |
| `Fab` | Heavy + light chains, editable variable domains, constant body kept off the target | Fab-format binders. |
| `ARP` | Ankyrin Repeat protein — a consensus ankyrin-repeat scaffold with editable repeat positions | Ankyrin-repeat (ARP) binders. |
| `induced_fit` | The interface moves ≥5 Å between the free and bound prediction | The binder should change shape on binding. |
| `fold_switch` | The whole fold differs free vs bound (TM-score ≤ 0.6) | You explicitly want a fold-switching binder. |

### How to choose — biophysical intuition

- **Not sure? Use `binder`.** De novo miniproteins are the most reliable modality: AlphaFold predicts
  idealised secondary structure well, and a de novo binder can shape a paratope complementary to
  almost any epitope.
- **Match the binder shape to the epitope shape.**
  - *Flat, featureless interfaces* are hard for anything small. A short peptide binds a flat surface
    poorly: a linear chain pays a large conformational-**entropy** penalty on binding, and a flat
    surface offers too little buried area and too few pockets to pay it back. Use a larger de novo
    `binder`/`large_binder` that can lay a complementary face across the surface.
  - *Grooves, clefts, pockets and cryptic/concave sites* suit extended elements — a `peptide`
    threading a groove, or a `VHH` reaching in with an extended CDR3. This is exactly where VHHs
    excel and flatter binders struggle.
- **Cyclic beats linear for peptides.** `cyclic_peptide` constrains the backbone, pre-paying part of
  the entropy cost, so it typically binds better than a linear peptide of similar length.
- **Antibody-format modalities (`VHH`, `scFv`, `Fab`) usually perform worse than de novo modalities.**
  AlphaFold leans on co-evolutionary (MSA) signal, and engineered/immune scaffolds carry weak
  co-evolutionary information for their hypervariable loops, so the paratope is predicted less
  reliably. Expect lower success rates and use these only when you need that *format*, not the best
  binder. And **not every target is VHH-addressable** — a flat epitope with no cleft for the CDR3 is a
  poor VHH target.
- **`induced_fit`/`fold_switch` are objectives, not formats.** They build on the de novo `binder` base
  and *require* the fold to move on binding. Use them only when that change is the goal; for a rigid
  binder they just make the task harder.

### What each modality is for

| Modality | Typical applications |
| --- | --- |
| `binder` | The general-purpose choice: research reagents and pulldowns, biosensors, crystallisation/cryo-EM fiducials, therapeutic-lead and targeting domains. Most reliable to fold and bind. |
| `large_binder` | Primarily a **structural-biology tool**: a rigid binder adds **mass and recognisable features to a small target for cryo-EM** (and can act as a crystallisation chaperone), making an otherwise too-small particle tractable. Secondarily, big or flat epitopes that need more buried area, higher-avidity single chains, and longer fusion/scaffolding domains. |
| `peptide` | Inhibitors that thread a **groove or cleft** (protein–protein interfaces with a linear hotspot, active-site channels), tool compounds and targeting peptides. Poor on flat surfaces. |
| `cyclic_peptide` | Macrocycle-style binders wanting protease resistance and rigidity; the ring's lower entropy makes it a better peptide binder than a linear one of the same length. |
| `homo_oligomer` | Symmetric, multivalent binders — avidity, receptor **clustering/agonism**, and self-assembling building blocks. Especially good for **symmetric targets** that a monomeric binder handles poorly (e.g. homotrimeric **TNFα**): a matched Cₙ oligomer can engage every protomer of the symmetric target at once. |
| `multidomain` | Single-chain **biparatopic/bispecific** reach across two epitopes (or two targets), and larger, higher-avidity architectures. |
| `VHH` | Single-domain antibodies for **concave/cryptic epitopes and enzyme active sites**, intrabodies, crystallisation chaperones, imaging, and modular fusion building blocks. |
| `scFv` | Variable-fragment format for **CAR-T binding domains** and bispecific/multispecific building blocks — when the downstream construct needs an scFv specifically. Least stable of the antibody formats. |
| `Fab` | The classic therapeutic/diagnostic antibody fragment: more stable and manufacturable than an scFv, and the base the scFv here is derived from. |
| `ARP` | Ankyrin Repeat protein — non-antibody, disulfide-free, high-stability scaffold: cheap microbial production, intracellular use, and easy multivalent fusions. |
| `induced_fit` | Binders for targets that **change shape on binding** (conformational selection), and allosteric or state-selective binders. |
| `fold_switch` | Conditional/switchable binders and sensors, where the binder is meant to adopt a different fold free vs bound. |

### Default scaffolds — where they come from

The scaffold modalities edit a fixed backbone instead of folding from nothing. Each ships a canonical
framework in `scaffolds/`, with the **framework held fixed and only the binding loops/repeat positions
editable** (`mutate_positions` marks which, and `min_scaffold_sequence_retained_final` keeps the
framework sequence intact so the output is a real, expressible member of that format).

The three antibody-format scaffolds are built on **human germline** frameworks, chosen deliberately so
the framework carries no IP: the germline is retained and only the CDRs are designed.

| Modality | Scaffold file | Framework (human germline unless noted) |
| --- | --- | --- |
| `Fab` | `scaffolds/Fab.cif` | VH **IGHV3-23\*01** (IMGT M99660) + VL **IGKV1-39\*01** (IMGT X59315). VH3 is the most stable heavy family and Vκ1 the preferred light family, so VH3-23/Vκ1-39 is the safe, well-behaved pairing. Variable domains editable; the constant body is kept off the target. |
| `scFv` | `scaffolds/scFv.cif` | The **same IGHV3-23\*01 + IGKV1-39\*01** framework as the Fab, as its VH (1–119) and VL (1–107). BC2 models the **two variable domains as separate chains with no linker** — it designs the domains; you add the VH–VL linker yourself when you build the construct. |
| `VHH` | `scaffolds/VHH.cif` | A **human IGHV3-23\*01** autonomous single-domain (VHH-format) VH (IMGT M99660; FR4 from IGHJ4\*01), CDRs editable, both CDR3 conformations sampled (see below). It uses the human-germline **GLEW** FR2, not the camelid ERE hallmark. |
| `ARP` (Ankyrin Repeat protein) | `scaffolds/ARP.cif` | A full-consensus designed ankyrin-repeat protein — the repeat framework held fixed with the variable repeat positions opened. A synthetic consensus scaffold, not an antibody germline. |

> **Naming and IP.** BC2 deliberately builds on non-proprietary frameworks: the antibody scaffolds use
> human germline sequences (IGHV3-23, IGKV1-39, IGHJ4, with the VHH on the human-germline GLEW
> framework), and the ARP is a long-established full-consensus ankyrin-repeat fold. The names
> "DARPin®" and "Nanobody®" are registered trademarks, which is why BC2 uses **ARP** and **VHH**
> instead. This is not a freedom-to-operate opinion — confirm FTO on your designed sequences, the
> chosen format and any downstream construct with your own counsel before development. Supply your own
> scaffold if you need a specific framework; a custom antibody CIF must be **sequentially renumbered**
> (the engine rejects Kabat insertion codes).

### VHH: extended vs folded-back paratope

A VHH's long CDR3 can sit in two very different poses, and BC2 samples both by default
(`"paratope_conformations": ["extended", "folded_back"]`):

- **`extended`** — CDR3 projects away from the framework as a convex "finger" that reaches into
  **concave epitopes**: enzyme active sites, clefts, pockets, cryptic sites. The classic VHH
  advantage. In this pose the former VH/VL interface (the FR2 hallmark patch, normally buried against
  a light chain) is left solvent-exposed, so BC2 **mutates those exposed framework residues to soluble
  (hydrophilic) versions** — the aromatics are downweighted and the interface patch redesigned — to
  keep the single domain soluble and non-sticky on its own.
- **`folded_back`** — CDR3 folds back over the framework, presenting a **flatter, more compact
  paratope** for **flatter or convex epitopes**, where an extended loop would find nothing to grip.

Leave both on to let the campaign find which fits your epitope; restrict to one
(`"paratope_conformations": ["extended"]`) when you already know the geometry and want the whole
budget spent on it.

---

## 4. Helpful properties and objectives

**Properties** are optional booleans (`"humanize": true`, or `--humanize`) that add a biological
objective and its associated filters. They stack on top of a modality.

Every property works the same two ways: a **gradient term** (`weights_*`) pulls the trajectory toward
the goal while the binder folds, and an **acceptance filter** (`*_final`) rejects any re-predicted
design that didn't actually achieve it. Loosen the filter to keep borderline designs; drop the property
to stop optimising for it. Because each adds an objective *and* a bar, **stacking many makes acceptance
rarer** — add only what your experiment needs. The overall roles:

| Property / objective | Pushes the design toward | Rejects unless |
| --- | --- | --- |
| `forced_targeting` | contact concentrated on the declared hotspots | ≥50% of hotspots contacted |
| `humanize` | humanized sequence (*planned*) + low predicted MHC anchor load | MHC anchor score under its ceiling |
| `disulfide_staple` | a geometrically valid disulfide (cysteine allowed) | ≥1 disulfide formed |
| `protease_stable` | fewer protease-cleavage motifs, buried loops and termini | protease-site / exposed-loop / terminus-exposure scores under their ceilings |
| `termini_accessible` | both chain ends angled away from the target | termini-away angle clears its floor |
| `termini_together` | N and C termini within ~7 Å | termini distance under ~10 Å |
| `mixed_topology` | less helix, more β-sheet | ≤50% helix **and** ≥20% sheet |
| `induced_fit` (objective) | the interface moving ≥5 Å free→bound | free-vs-bound interface RMSD above its floor |
| `fold_switch` (objective) | the whole fold differing free vs bound | free-vs-bound TM-score under 0.6 |
| `multidomain` (objective) | two separated domains joined by a linker | domain-separation / interdomain-contact / chain-break checks |
| detargeting (negative `weight`) | the binder repelled from the off-target | off-target `i_pTM` and interface residues under their ceilings |

`initial_guess` and `bigbang` are the exceptions — they change *how* optimisation is initialised, not
what is optimised, so they carry no filter (see §5).

### What each one actually does — and what it does *not* tell you

**`forced_targeting` — the "lysine trick".** During the early gradient stages BC2 mutates every exposed
target residue *outside* a shell around your hotspots (`forced_targeting_shell`, default 10 Å) to
**lysine**, turning the rest of the surface into a hostile, positively-charged patch so the binder has
nowhere attractive to sit but your epitope; the real target sequence is restored before hardening and
validation. Requires a structured target and `hotspots`. *Caveat:* it steers **where** the binder
lands, not how tightly it binds, and the trick runs only at design time — a design still has to clear
the normal interface filters against the true target.

**`humanize` — immunogenicity proxy.** In development, currently scores the designed sequence against a panel of common **MHC anchor motifs**
(MHC class I: 13 common HLA-A/B alleles; MHC class II: common HLA-DRB1 alleles, weighted highest via
`humanization_mhc2_weight`). 
The corresponding filter is `MHC_Anchor_Score`. 
*Caveat:* currently this is a coarse MHC presentation estimator over a fixed allele panel. It does **not** establish that a designed protein is non-immunogenic and should be treated as a proxy.

**`protease_stable` — a small serum-protease panel plus a burial term.** It penalises predicted cleavage
against four canonical proteases — **trypsin** (after K/R), **chymotrypsin** (after F/Y/W/L/M),
**elastase** (after A/V/G/S) and **pepsin** (after F/L/W/Y), each blocked by a following proline — and
separately penalises **exposed loops and exposed termini**, since proteases attack flexible,
solvent-exposed backbone. Filters cap the protease-site score, the exposed-loop fraction and terminus
exposure. *Caveat:* four textbook proteases and a burial proxy are **not** a measurement of serum
half-life or stability; real proteolysis involves many more enzymes, plus glycosylation, formulation
and clearance.

**`disulfide_staple` — a geometric disulfide.** It allows cysteine (`aa_bias`) and rewards a disulfide
whose geometry matches a real S–S bond (~3.8 Å with a minimum sequence separation), requiring at least
one in the accepted design. Its natural use is a **terminally-linked (disulfide-cyclised) peptide** or
a stapled mini-binder — a chemistry route to rigidity and an alternative to a head-to-tail
`cyclic_peptide`. *Caveat:* it rewards a **plausible** disulfide geometry, not a verified bond; the
protein still has to fold and oxidise correctly, and disulfides won't survive a reducing (e.g.
intracellular) environment.

**`termini_accessible` / `termini_together` — chain-end geometry.** The first angles both the N and C
termini *away* from the target, so a fusion partner, tag or immobilisation chemistry can be attached
without clashing the interface; the second pulls the two termini *close together* (within ~7 Å,
filtered under ~10 Å) for grafting, cyclisation or loop insertion. *Caveat:* geometry only — it says
nothing about whether the intended fusion or cyclisation will express or fold.

**`mixed_topology` — force some β-sheet.** It removes the default helicity reward and penalises helix
while requiring sheet, steering away from the all-α helical bundles de novo design tends to produce;
it caps helix at ≤50% and requires ≥20% sheet. *Caveat:* β-rich de novo folds are harder to design and
predict, so expect a lower hit rate.

**`initial_guess` / `bigbang` — how AlphaFold is initialised.** Both change the *starting coordinates*
AlphaFold works from, not the objective, so neither adds a filter of its own.
- **`initial_guess`** re-predicts each redesigned candidate **starting from the pose the trajectory
  folded** — instead of predicting the sequence from a blank slate, AlphaFold begins its recycling from
  the design's own backbone. This helps it converge to the intended fold and interface for **difficult
  motifs** (extended loops, shallow or unusual interfaces, disordered-region binders) that a
  from-scratch prediction can miss.
- **`bigbang`** (`bigbang_initialization`) seeds the **gradient design stages** from the coordinates on
  hand rather than from the origin, giving AlphaFold a foothold on **large complexes (>~600 aa)** it
  struggles to build from nothing — which is where reprediction of a large motif otherwise fails.

The trade-off is **bias**: because the predictor is handed a structure close to the answer, the
validation is slightly **less independent**, so a design that only holds up because it was given its
own pose is a potential false positive — expect a **modest rise in false-positive rate** versus a fully
from-scratch refold. That bias is deliberate and bounded, and importantly **both options have been
experimentally validated** — designs accepted with them have yielded real binders — so they are sound
tools for hard targets and difficult motifs. Use them when a target won't repredict otherwise; where
you can, spot-check a few winners with the option off. (`initial_guess` is also a rung of the
[desperation ladder](#5-desperation-and-autotuning-what-runs-on-its-own).)

### Combining modalities and properties

Most modalities and properties **stack freely** — `VHH` + `humanize`, `binder` + `protease_stable` +
`termini_accessible`, a structured target + `forced_targeting` + several off-targets all work. A few
combinations are contradictory, and BC2 **refuses them at start-up with an explanatory error** rather
than designing something incoherent:

| These don't combine | Why |
| --- | --- |
| a **scaffold modality** (`VHH`, `scFv`, `Fab`, `ARP`) with `cyclic_peptide`, `homo_oligomer` (`copies` > 1), `fold_switch`, or `mixed_topology` | a fixed framework already sets the fold and the chain, so it can't also be cyclised, copied into an oligomer, told to switch fold, or told to change its secondary structure |
| `homo_oligomer` (`copies` > 1) with `multidomain` | the domain split doesn't engage across identical oligomer copies |
| a **FASTA / disordered target** with `forced_targeting` or `coldspots` | both need residue numbers and a resolved backbone that a sequence target doesn't carry |
| `induced_fit` with **detargeting** | induced fit freezes one bound structure to compare the free state against, so it designs against a single target |
| `binder_sequences` with a **scaffold modality**, or with a `binder_lengths` you write yourself | both decide what the binder starts as: a sequence you gave already fixes the sequence and its length, and no stage can insert or delete a residue. Refused outright rather than silently ignored |

Everything else is fair game — targeting options (hotspots, coldspots, forced targeting, detargeting),
developability properties (humanize, protease_stable, disulfide_staple), termini controls and topology
all layer onto any compatible modality. But because each property adds an objective **and** a filter,
**more is not better**: every extra requirement makes acceptance rarer and trades against the
interface. Start from the modality plus the one or two properties your experiment truly needs, confirm
designs appear, then add more.

---

## 5. Desperation and autotuning (what runs on its own)

BC2 adapts a stalled campaign automatically. Two separate mechanisms:

**Autotuner** (`autotune`, on by default). Every ten trajectories it nudges the `screen` and
`refine` stage lengths (and, if you enabled `autotune_loss_weights`, weights you changed yourself)
within safe bounds. It never touches recycles, the validation pool, or the starting conformation.
Harmless; leave it on.

**Desperation ladder** (`desperation`, on by default). If the campaign has accepted **nothing** for
`desperation_trajectories` (default 750) trajectories, it starts trading away difficulty to get
*any* design, one rung every further 50 trajectories:

| Rung | Runs at |
| --- | --- |
| 1 | `initial_guess` |
| 2 | `target_flexibility` 0.5 |
| 3 | `initial_guess` + `target_flexibility` 0.5 |
| 4 | validation on held-out `multimer` models |
| 5 | multimer validation + `initial_guess` |
| 6 | multimer validation + `initial_guess` + `target_flexibility` 0.5 |
| 7 | the above + `design_recycles` 3 |

The ladder is dropped the moment a design is accepted.

> ⚠️ **Every rung of the ladder raises the expected false-positive rate.** A design accepted on a
> higher rung was accepted against an *easier* design task and judged by a *looser* validation, so it
> is more likely to fail in the wet lab than one accepted at your requested settings — the filters
> passing does **not** mean the experiment will. The campaign log prints a `desperation:` line naming
> the rung, and the trajectory's `autotuned` column (`target_flexibility`, `initial_guess`,
> `multimer`, raised `design_recycles`) records how far down the ladder it was accepted. Weight each
> design by that, order more replicates of ladder-accepted ones, and if a whole campaign only produced
> rung-6/7 designs, the target/epitope is probably too hard as posed — revisit the epitope or the
> length rather than trusting the output.

The `benchmark` core profile (`"core": "benchmark"`) sets a fixed `campaign_seed` and turns
`autotune` and `desperation` off, for a run you can reproduce.

**`bigbang`** (and `bigbang_initialization`) seeds the gradient stages from the coordinates already
on hand rather than from the origin. Its payoff is in **large complexes (>~600 aa)**, where AlphaFold
struggles to build a fold from scratch and a coordinate start gives it a foothold — reach for it with
`large_binder`, `multidomain` or large targets. Small binders that fold easily from the origin gain
little, so it is off by default.

---

## 6. Reading the outputs

A campaign folder has three numbered stage folders plus records:

```
results/pdl1/
├── 1_Trajectories/   hallucinations (attempts)
│   ├── !_Trajectories.csv        one row per attempt, with its termination stage
│   └── <design>/                 per-trajectory losses (and optional frames/animations)
├── 2_Refolded/       honest re-predictions of every redesigned sequence
│   ├── !_Refolded.csv            every scored candidate + why it failed
│   ├── Complexes/                the predicted complex of each candidate
│   └── BinderMonomer/            the binder predicted alone (when checked)
└── 3_Ranked/         the designs that passed
    ├── !_Ranked.csv              THE result: accepted designs, best-first by i_pDAE
    └── <design>_seq<n>*.cif      the accepted structures
```

**Open `3_Ranked/!_Ranked.csv` first.** It is the single record of accepted designs, ranked. If it's
empty, look in `2_Refolded/!_Refolded.csv` at the `failed_filters` column to see *why* candidates
were rejected, then `1_Trajectories/!_Trajectories.csv` for attempts that died before redesign.

The metrics that matter, and how to read them:

| Metric | Range / direction | What it tells you |
| --- | --- | --- |
| `i_pDAE` | 0–1, higher better | **BC2's ranking score.** Distance-masked interface confidence. |
| `i_pTM` | 0–1, higher better | Interface confidence. Default acceptance floor **0.7**. |
| `i_pAE` | normalised, lower better | Mean interface PAE ÷ 31 Å. `0.35 ≈ 10.85 Å`; default ceiling **0.35**. |
| `pTM` | 0–1, higher better | Whole-complex confidence. Default floor 0.55. |
| `pLDDT` | 0–1, higher better | Binder confidence in the bound state. |
| `Unbound_Binder_pLDDT` | 0–1, higher better | Confidence of the binder **predicted alone**. Default floor 0.8 (peptides exempt). |
| `Interface_Residues` | count | Binder residues within 4 Å of target. Default floor **7** — a real interface, not a glancing touch. |
| `Binder_RMSD` | Å, lower better | Free-vs-bound binder displacement. On de novo/large/homo modalities it must be ≤ 3.5 Å (the binder should fold the same alone as bound). |

> **None of these are affinity.** A perfect `i_pTM` describes a *confident predicted pose*, not a
> tight or specific binder, and not a biologically accessible one. Always sanity-check the pose
> against membranes, glycans, the full-length target and your assay geometry before ordering.

In predicted structure files, the B-factor column stores per-residue pLDDT on a **0–100** scale
(BC2 convention, not experimental B-factors).

---

## 7. What to look out for (common pitfalls)

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

---

## 8. A sensible first campaign

1. Prepare and inspect your target; decide whether to name hotspots (optional — name them to focus a
   specific epitope, or leave them off to let BC2 find a site).
2. Start with `"modality": "binder"` at its default lengths, a handful of designs and a few hundred
   trajectories as a smoke test.
3. Open `3_Ranked/!_Ranked.csv`. If it's empty, read `failed_filters` and adjust the epitope, length
   or modality — not the loss weights.
4. Once designs appear at your requested settings (not on desperation rungs), scale
   `number_of_final_designs` and `max_trajectories` up for the real run.
5. Rank/inspect the top designs, check the poses by eye, and order a diverse set — top `i_pDAE` is a
   starting point, not a guarantee.

For the exhaustive list of every setting and its default, see
[`reference.md`](reference.md); for every output file and measurement, see
[`outputs.md`](outputs.md).
