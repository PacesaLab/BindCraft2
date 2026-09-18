# A sensible first campaign

[Full guide](../design-guide.md) · [Settings](../reference.md) · [Outputs](../outputs.md)

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
[`reference.md`](../reference.md); for every output file and measurement, see
[`outputs.md`](../outputs.md).
