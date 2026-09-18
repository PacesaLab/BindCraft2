# Desperation and autotuning (what runs on its own)

[Full guide](../design-guide.md) · [Settings](../reference.md) · [Outputs](../outputs.md)

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
