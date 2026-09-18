# Choosing a modality

[Full guide](../design-guide.md) · [Settings](../reference.md) · [Outputs](../outputs.md)

The **modality** sets the binder format and the conformational objective. Name it with `"modality":
"..."` (or `--modality`). You can combine some, e.g. `["VHH", "induced_fit"]`.

| Modality | What it makes | Choose it when |
| --- | --- | --- |
| `binder` | A de novo miniprotein, ~60–180 residues, folded from nothing | The default and most reliable. General de novo binder against a structured epitope. |
| `large_binder` | A longer de novo binder (~250–600 residues) with extended optimisation | Chiefly to add **mass to a small target for cryo-EM / structural biology**; also for large or flat epitopes that need more interface. **Name `binder_lengths`** (see `bigbang` in [Desperation and autotuning](05-desperation-and-autotuning.md) for large complexes). |
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

## How to choose — biophysical intuition

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

## What each modality is for

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

## Default scaffolds — where they come from

The scaffold modalities edit a fixed backbone instead of folding from nothing. Each ships a canonical
framework in `scaffolds/`, with the **framework held fixed and only the binding loops/repeat positions editable** (`mutate_positions` marks which, and `min_scaffold_sequence_retained_final` keeps the
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

## VHH: extended vs folded-back paratope

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
