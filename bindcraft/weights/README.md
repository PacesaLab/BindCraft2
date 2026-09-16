# Shipped model weights

`proteinmpnn/` holds the ProteinMPNN checkpoints BC2 redesigns with, so an installation carries them
and no campaign names a path. Three variants of the same four models, each directory named for the one it holds: `weights_neutral/`
is the original release, `weights_negative/` the soluble-protein retraining BC2 uses by default, `weights_positive/`
the HyperMPNN thermostability retraining, each named for the surface charge it favours.
`mpnn_variant` selects between them and `mpnn_model` between `v_48_002`, `v_48_010`, `v_48_020` and
`v_48_030`, the backbone-noise levels the models were trained at. Each checkpoint is an `.npz`
holding one array per parameter, keyed `<module>|<parameter>` beside the `num_edges` and
`noise_level` it was trained with, so loading one reads no pickle. Neutral and negative weights are
from `dauparas/ProteinMPNN` (MIT), positive weights from `meilerlab/HyperMPNN` (MIT).

`alphafold/` is empty in the repository and is where a container bakes its AlphaFold parameters. The
2022-12-06 release is 5.3 GB, so an ordinary installation downloads it once into
`~/.cache/bindcraft/alphafold` on its first campaign instead. `bindcraft fetch-weights` does the same
download on demand, which is what a compute node with no route to the internet needs run for it on a
login node first. The parameters are CC-BY-4.0, DeepMind, and their licence travels in the archive.
