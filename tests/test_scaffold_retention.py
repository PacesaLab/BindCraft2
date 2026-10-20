# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest", "jax", "numpy", "optax", "biotite==1.8.0"]
# ///
"""Exercise scaffold metrics with held residues and shipped presets on CPU."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from bindcraft.filters import (
    build_filters,
    evaluate_design_filters,
    scaffold_framework_rmsd_metric,
    scaffold_sequence_retained_metric,
)
from bindcraft.protein import (
    Protein,
    ResidueFlags,
    StructurePrediction,
    parse_scaffold_edits,
    structure_chain_names,
)
from bindcraft.protein_preparation import prepare_binder_chains
from bindcraft.settings import (
    BinderDesignSettings,
    BinderSettings,
    record_modality_filter,
    scaffold_check_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
VHH = str(ROOT / "scaffolds/VHH.cif")


@pytest.mark.parametrize("edits", ["", "A1!"])
def test_unchanged_scaffold(edits: str) -> None:
    """Declaring a held residue must not invalidate an unchanged scaffold."""
    scaffold = Protein.from_structure(VHH, chains="A")["A"]
    binder = scaffold.with_scaffold("A", edits, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(binder.sequence, scaffold.sequence)
    np.testing.assert_array_equal(binder.atoms, scaffold.atoms)
    assert_scaffold_metrics(binder, VHH, edits, "A")


def assert_scaffold_metrics(
    binder: Protein, scaffold_path: str, edits: str, chain: str
) -> None:
    """Check both consumers of the framework correspondence helper."""
    states = {"complex": {"binder": binder}}
    predictions = {
        "complex": StructurePrediction(protein_complex=states["complex"], metrics={})
    }
    assert (
        scaffold_sequence_retained_metric(
            states,
            predictions,
            scaffold=scaffold_path,
            scaffold_edits=edits,
            scaffold_chain=chain,
        )
        == 1.0
    )
    assert scaffold_framework_rmsd_metric(
        states,
        predictions,
        scaffold=scaffold_path,
        scaffold_edits=edits,
        scaffold_chain=chain,
    ) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("modality", ["VHH", "Fab", "scFv", "ARP"])
def test_shipped_scaffold_presets(modality: str) -> None:
    """Preset edits preserve the protected sequence and its coordinates."""
    settings = json.loads((ROOT / f"settings/modality/{modality}.json").read_text())
    scaffold_path = str(ROOT / settings["binder_scaffold"])
    edits = settings["mutate_positions"]
    for chain, scaffold in Protein.from_structure(scaffold_path).items():
        binder = scaffold.with_scaffold(chain, edits, jax.random.PRNGKey(0))
        assert_scaffold_metrics(binder, scaffold_path, edits, chain)


@pytest.mark.parametrize("index", [0, 1], ids=["held", "unlisted-framework"])
def test_changed_protected_residue_reduces_retention(index: int) -> None:
    """Held and unlisted framework residues participate in the comparison."""
    scaffold = Protein.from_structure(VHH, chains="A")["A"]
    binder = scaffold.with_scaffold("A", "A1!", jax.random.PRNGKey(0))
    binder = binder.replace(
        sequence=binder.sequence.at[index].set(np.roll(binder.sequence[index], 1))
    )
    states = {"complex": {"binder": binder}}
    predictions = {
        "complex": StructurePrediction(protein_complex=states["complex"], metrics={})
    }
    retained = scaffold_sequence_retained_metric(
        states,
        predictions,
        scaffold=VHH,
        scaffold_edits="A1!",
        scaffold_chain="A",
    )
    assert retained == pytest.approx((len(scaffold) - 1) / len(scaffold))


@pytest.mark.parametrize("edits", ["A5-7(5)", "A1!,A5-7(5)"])
def test_changed_designed_residue_does_not_reduce_retention(edits: str) -> None:
    """Length edits and changes inside designed spans preserve correspondence."""
    scaffold = Protein.from_structure(VHH, chains="A")["A"]
    binder = scaffold.with_scaffold("A", edits, jax.random.PRNGKey(0))
    assert len(binder) == len(scaffold) + 2
    assert int(binder.flags[4]) & ResidueFlags.DESIGN
    changed = binder.replace(
        sequence=binder.sequence.at[4].set(np.roll(binder.sequence[4], 1))
    )
    assert_scaffold_metrics(changed, VHH, edits, "A")


def test_displaced_held_residue_increases_framework_rmsd() -> None:
    """The RMSD comparison must include explicitly held residues."""
    scaffold = Protein.from_structure(VHH, chains="A")["A"]
    binder = scaffold.with_scaffold("A", "A1!", jax.random.PRNGKey(0))
    assert_scaffold_metrics(binder, VHH, "A1!", "A")
    displaced = binder.replace(atoms=binder.atoms.at[0].add(5.0))
    states = {"complex": {"binder": binder}}
    predictions = {
        "complex": StructurePrediction(
            protein_complex={"binder": displaced}, metrics={}
        )
    }
    rmsd = scaffold_framework_rmsd_metric(
        states, predictions, scaffold=VHH, scaffold_edits="A1!", scaffold_chain="A"
    )
    assert rmsd is not None and rmsd > 0.1


@pytest.mark.parametrize("modality", ["VHH", "Fab", "scFv"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_preset_gate_with_automatic_chain_matching(modality: str, seed: int) -> None:
    """Use production parameters and reject a changed held residue per chain."""
    settings = json.loads((ROOT / f"settings/modality/{modality}.json").read_text())
    settings["binder_scaffold"] = str(ROOT / settings["binder_scaffold"])
    settings["filters"] = {}
    metric = "Scaffold_Sequence_Retained_Fraction"
    record_modality_filter(
        settings,
        metric,
        0.0,
        higher=True,
        threshold_setting="min_scaffold_sequence_retained_final",
        **scaffold_check_parameters(settings),
    )
    assert "scaffold_chain" not in settings["filters"][metric]["params"]
    scaffold_path = settings["binder_scaffold"]
    edits = settings["mutate_positions"]
    design_settings = BinderDesignSettings(
        targets=[],
        binder=BinderSettings(scaffold=scaffold_path, scaffold_edits=edits),
        settings=settings,
    )
    filters = build_filters(settings["filters"], design_settings.designed_binder_chain)
    scaffolds = Protein.from_structure(scaffold_path)
    chain_pairs = list(
        zip(
            design_settings.binder_chains,
            structure_chain_names(scaffold_path),
        )
    )
    key = jax.random.PRNGKey(seed)
    binders = prepare_binder_chains(design_settings, key)
    states = {"complex": binders}
    predictions = {"complex": StructurePrediction(protein_complex=binders, metrics={})}
    outcome, metrics = evaluate_design_filters(filters, states, predictions)
    assert outcome is True, outcome
    assert metrics[metric] == 1.0
    assert scaffold_framework_rmsd_metric(
        states, predictions, **scaffold_check_parameters(settings)
    ) == pytest.approx(0.0, abs=1e-3)

    for name, chain in chain_pairs:
        held_start = next(
            start
            for start, _, _, flags in parse_scaffold_edits(edits, chain, key)
            if not flags & ResidueFlags.DESIGN
        )
        scaffold = scaffolds[chain]
        original_index = int(np.flatnonzero(scaffold.residue_index == held_start)[0])
        binder = binders[name]
        # Coordinates identify the held residue even after upstream length edits.
        matches = np.flatnonzero(
            np.asarray(binder.atoms == scaffold.atoms[original_index]).all(axis=(1, 2))
        )
        assert len(matches) == 1
        index = int(matches[0])
        changed = binder.replace(
            sequence=binder.sequence.at[index].set(np.roll(binder.sequence[index], 1))
        )
        changed_states = {"complex": {**binders, name: changed}}
        outcome, metrics = evaluate_design_filters(filters, changed_states, predictions)
        assert outcome == [metric]
        assert 0.0 < metrics[metric] < 1.0
