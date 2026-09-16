import argparse
import textwrap
import json
import math
import os
import sys
from pathlib import Path
from bindcraft.campaign_output import CONFIDENCE_METRIC, RANKING_METRIC, SCORED_FILENAME, structure_paths, design_model_scores
from bindcraft.filters import REGISTERED_FILTER_METRICS, binder_chain_sequences, design_sequence_report
from bindcraft.loss import binder_copy_chains, resolve_binder_role
from bindcraft.protein import BINDER_CHAIN_PREFIX, Protein, ProteinStates, StructurePrediction, StructurePredictions, read_structure_atoms, read_structure_metadata, recorded_number, selected_chain_names, structure_chain_names, target_chain_name
from bindcraft.protein_preparation import merge_receptor_chains, target_binding_site_flags

SCORED_STATE = 'design'
SCORED_PROGRESS_INTERVAL = 10
SCORED_KEPT_INTERVAL = 50
PREDICTED_METRICS = ('pLDDT', 'pTM', 'i_pTM', 'i_pAE', RANKING_METRIC)
CAMPAIGN_ONLY_METRICS = ('Off_Paratope_Contact_Fraction',)

def design_chain_roles(chain_names: list[str], binder: str='', target: str='') -> tuple[list[str], list[str]]:
    binder_chains = selected_chain_names(binder, chain_names) if binder else []
    target_chains = selected_chain_names(target, chain_names) if target else []
    if not binder_chains:
        binder_chains = [name for name in chain_names if name not in target_chains] if target_chains else chain_names[1:] or chain_names
    if not target_chains:
        target_chains = [name for name in chain_names if name not in binder_chains]
    return binder_chains, target_chains

def redesigned_binder_flags(structure: str) -> str:
    stamp = read_structure_metadata(structure)
    return ','.join(f'{span.strip()}+{flag}' for name, flag in (('redesigned_residues', 'DESIGN'), ('paratope_residues', 'CONTACT')) for span in (stamp.get(name) or '').split(',') if span.strip())

def design_states(structure: str, binder: str='', target: str='', hotspots: str='', coldspots: str='') -> tuple[ProteinStates, StructurePredictions, dict[str, tuple[tuple[str, int, int], ...]]]:
    chain_names = structure_chain_names(structure)
    stamp = read_structure_metadata(structure)
    #a design bindcraft wrote says which chains it made and which it aimed at, and a two-chain receptor makes the "every chain but the first" guess wrong
    binder_chains, target_chains = design_chain_roles(chain_names, binder or stamp.get('binder_chains', ''), target or stamp.get('target_chains', ''))
    binder_proteins = Protein.from_structure(structure, chains=','.join(binder_chains), flags=redesigned_binder_flags(structure))
    protein_complex = {BINDER_CHAIN_PREFIX if len(binder_chains) == 1 else f'{BINDER_CHAIN_PREFIX}_{copy_number}': binder_proteins[name] for copy_number, name in enumerate(binder_chains)}
    receptor_chains = {}
    if target_chains:
        target_proteins = Protein.from_structure(structure, chains=','.join(target_chains), flags=target_binding_site_flags(target_chains, hotspots, coldspots))
        protein_complex[target_chain_name('target', SCORED_STATE)] = merge_receptor_chains([target_proteins[name] for name in target_chains])
        if len(target_chains) > 1:
            receptor_chains[target_chain_name('target', SCORED_STATE)] = tuple((name, len(target_proteins[name]), int(target_proteins[name].residue_index[0])) for name in target_chains)
    return {SCORED_STATE: protein_complex}, {SCORED_STATE: StructurePrediction(protein_complex=protein_complex, metrics={})}, receptor_chains

def scored_metric_names() -> tuple[str, ...]:
    return tuple(name for name in sorted(REGISTERED_FILTER_METRICS) if name not in CAMPAIGN_ONLY_METRICS and (not name.endswith('_detarget')))

def scored_metric_guide() -> str:
    return '\n'.join(('checks this prints, under the exact spelling it prints them with:',
                      textwrap.fill(', '.join(scored_metric_names()), width=100, initial_indent='  ', subsequent_indent='  '),
                      '',
                      f"{', '.join(PREDICTED_METRICS)} are readings of the prediction: a design a campaign wrote carries",
                      'them in its own stamp, and they are reported from there rather than recomputed.',
                      'a campaign records these per target as well, written <metric>.<target name>.'))

def score_design(structure: str, binder: str='', target: str='', hotspots: str='', coldspots: str='') -> dict:
    protein_states, predictions, receptor_chains = design_states(structure, binder, target, hotspots, coldspots)
    protein_complex = protein_states[SCORED_STATE]
    binder_chain = next(iter(protein_complex))
    scores: dict = dict(design_sequence_report(predictions, prediction_state=SCORED_STATE, binder=binder_chain, receptor_chains=receptor_chains)) if len(protein_complex) > len(binder_copy_chains(protein_complex, binder_chain)) else {'Binder_Sequence': binder_chain_sequences(protein_complex, binder_chain)}
    for name, metric in REGISTERED_FILTER_METRICS.items():
        if name in CAMPAIGN_ONLY_METRICS or name.endswith('_detarget'):
            continue
        try:
            value = metric(protein_states, predictions, prediction_state=SCORED_STATE, **resolve_binder_role(metric, {}, binder_chain))
        except (KeyError, IndexError, ValueError):
            continue
        if value is not None:
            scores[name] = recorded_number(value)
    return scores

def binder_confidence(structure: str, binder_chains: list[str]) -> float | None:
    readings = [record['b_factor'] for record in read_structure_atoms(structure) if record['chain_id'] in binder_chains and record['name'] == 'CA']
    readings = [reading for reading in readings if math.isfinite(reading) and reading]
    if not readings:
        return None
    confidence = sum(readings) / len(readings)
    return recorded_number(confidence / 100.0 if confidence > 1.0 else confidence)

def scored_structure_row(structure: str, binder: str='', target: str='', hotspots: str='', coldspots: str='') -> dict:
    binder_chains, _ = design_chain_roles(structure_chain_names(structure), binder, target)
    confidence = binder_confidence(structure, binder_chains)
    return {'design': Path(structure).stem, **({CONFIDENCE_METRIC: confidence} if confidence is not None else {}),
            **score_design(structure, binder, target, hotspots, coldspots), 'structure': os.path.basename(structure)}

def score_structure_folder(folder: str, binder: str='', target: str='', hotspots: str='', coldspots: str='', rescore: bool=False) -> list[dict]:
    from bindcraft.campaign_output import read_metric_rows, write_csv_rows
    paths = structure_paths(folder)
    if not paths:
        return []
    scored_path = os.path.join(folder, SCORED_FILENAME)
    scored = {row['design']: row for row in (read_metric_rows(scored_path) if not rescore else [])}
    pending = [path for path in paths if Path(path).stem not in scored]
    kept = len(paths) - len(pending)
    print(f'{folder}: {len(pending)} structure(s) to score off their coordinates' + (f', {kept} already scored beside them' if kept else ''), flush=True)
    for counted, path in enumerate(pending, start=1):
        try:
            scored[Path(path).stem] = scored_structure_row(path, binder, target, hotspots, coldspots)
        except (KeyError, IndexError, ValueError, OSError) as unreadable:
            print(f'  {os.path.basename(path)}: not scored, {unreadable}', file=sys.stderr)
        if counted % SCORED_PROGRESS_INTERVAL == 0 or counted == len(pending):
            print(f'  scored {counted} of {len(pending)} structure(s)', flush=True)
        if counted % SCORED_KEPT_INTERVAL == 0 or counted == len(pending):
            write_csv_rows(list(scored.values()), scored_path, scored_column_order(list(scored.values())))
    return [scored[Path(path).stem] for path in paths if Path(path).stem in scored]

def scored_column_order(rows: list[dict]) -> list[str]:
    leading = ['design', CONFIDENCE_METRIC]
    return [name for name in leading if any(name in row for row in rows)] + [name for name in dict.fromkeys(name for row in rows for name in row) if name not in leading]

def main(arguments: list[str] | None=None) -> None:
    parser = argparse.ArgumentParser(prog='bindcraft score', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=scored_metric_guide(),
                                     description='Score one design structure on every check a campaign takes off coordinates')
    parser.add_argument('structure', help='the design to score, as mmCIF or PDB')
    parser.add_argument('--binder', default='', help='the binder chain(s), written "B" or "B,C" (default: every chain but the first)')
    parser.add_argument('--target', default='', help='the target chain(s) (default: the first chain, as an accepted design is written)')
    parser.add_argument('--hotspots', default='', help='target hotspot residues, written "A56,A99", for Hotspot_Contact_Fraction')
    parser.add_argument('--coldspots', default='', help='target coldspot residues, for Coldspot_Contact_Fraction')
    parsed = parser.parse_args(arguments)
    print(json.dumps(score_design(parsed.structure, parsed.binder, parsed.target, parsed.hotspots, parsed.coldspots), indent=2))
    print(f"# {', '.join(PREDICTED_METRICS)} are readings of the prediction, not of the structure, and are not recomputed here")
    model_scores = design_model_scores(parsed.structure)
    if model_scores:
        print('# what each validation model thought of this design, as the campaign stamped it into the structure')
        print(json.dumps(model_scores, indent=2))
if __name__ == '__main__':
    main()
