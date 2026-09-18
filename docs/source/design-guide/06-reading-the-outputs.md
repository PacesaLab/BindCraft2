# Reading the outputs

[Full guide](../design-guide.md) · [Settings](../reference.md) · [Outputs](../outputs.md)

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
