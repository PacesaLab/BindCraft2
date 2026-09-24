# Helpful properties and objectives

[Design Overview](../design-guide.md) · [Reference Documentation](../reference.md) · [Outputs and Measurements](../outputs.md)

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
what is optimised, so they carry no filter (see [Desperation and autotuning](05-desperation-and-autotuning.md)).

## What each one actually does — and what it does *not* tell you

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
[desperation ladder](05-desperation-and-autotuning.md).)

## Combining modalities and properties

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

Everything else is fair game — targeting options (hotspots, coldspots, forced targeting, detargeting),
developability properties (humanize, protease_stable, disulfide_staple), termini controls and topology
all layer onto any compatible modality. But because each property adds an objective **and** a filter,
**more is not better**: every extra requirement makes acceptance rarer and trades against the
interface. Start from the modality plus the one or two properties your experiment truly needs, confirm
designs appear, then add more.
