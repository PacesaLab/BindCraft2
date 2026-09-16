import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from bindcraft.campaign_output import CONFIDENCE_METRIC, DEFAULT_PROJECT_FOLDER, RANKING_METRIC, RANK_STAGE, REFOLD_STAGE, SCORED_FILENAME, STAGE_TABLE_NAMES, TRAJECTORY_STAGE, structure_paths, TARGET_NAME_COLUMN, TARGET_WEIGHT_COLUMN, numeric_value, on_target_mean, per_target_readings, read_metric_rows, recorded_target_values, stage_folder, stage_table, write_csv_rows

DESIGN_TABLES = {'accepted': RANK_STAGE, 'candidates': REFOLD_STAGE, 'trajectories': TRAJECTORY_STAGE}
METADATA_FILENAMES = ('campaign_metadata.json',)
DESIGN_COLUMNS = ('design',)
TEXT_COLUMNS = ('rank', 'design', 'terminated', 'failed_filters', 'Binder_Sequence', 'Interface_Binder_Residues', 'Interface_Target_Residues', TARGET_NAME_COLUMN, TARGET_WEIGHT_COLUMN)
HYDROPHOBIC_SEQUENCE_LETTERS = 'ACVILMPFWY'
STRUCTURE_FOLDERS = (RANK_STAGE, os.path.join(REFOLD_STAGE, 'Complexes'), REFOLD_STAGE, 'accepted', 'candidates', '.')

LOWER_IS_BETTER_METRICS = frozenset(('i_pAE', 'i_pAE_detarget', 'i_pTM_detarget', 'Interface_Residues_detarget', 'Backbone_Clashes', 'Binder_Chain_Breaks',
                                     'Binder_Free_Cysteines', 'Binder_Length', 'Binder_Loop_Fraction', 'Coldspot_Contact_Fraction', 'Cyclic_Closure_Distance',
                                     'Domain_Separation_Ratio', 'Framework_Packing_Fraction', 'Induced_Fit_Interface_RMSD',
                                     'Induced_Fit_RMSD', 'Induced_Fit_TM', 'MHC_Anchor_Score', 'Off_Epitope_Contact_Fraction', 'Off_Paratope_Contact_Fraction',
                                     'Oligomer_Symmetry_RMSD', 'Binder_RMSD', 'Scaffold_Framework_RMSD', 'Surface_Hydrophobicity',
                                     'Target_Crop_Length', 'Termini_Distance', 'Binder_Hydrophobic_Fraction', 'Binder_Cysteines',
                                     'All_Atom_Clashes', 'Binder_Mass_kDa', 'Exposed_Loop_Fraction', 'Protease_Site_Score',
                                     'Interface_Hydrophobicity', 'Interface_Loop_Fraction', 'Target_RMSD'))

MODALITY_METRICS = {
 'every campaign': {'i_pDAE': 'interface pDAE, how well the two sides agree on the pose (the default)',
                    'i_pTM': 'interface pTM of the predicted complex',
                    'i_pAE': 'interface predicted aligned error',
                    'pTM': 'pTM of the whole complex',
                    'pLDDT': 'binder confidence in the complex',
                    'Unbound_Binder_pLDDT': 'binder confidence when predicted alone',
                    'Target_pLDDT': 'target confidence in the complex',
                    'SS_pLDDT': 'confidence over the structured part of the binder only',
                    'Binder_RMSD': 'how far the binder moves between bound and unbound',
                    'Target_RMSD': 'how far the target backbone moves from its input structure',
                    'Interface_Residues': 'binder residues touching the target',
                    'Interface_BuriedArea': 'surface buried by the interface',
                    'Interface_BuriedArea_Fraction': 'fraction of the binder surface buried by the interface',
                    'Backbone_Clashes': 'backbone atoms too close across the interface',
                    'All_Atom_Clashes': 'every atom pair too close across the interface, side chains included',
                    'Surface_Hydrophobicity': 'exposed hydrophobic fraction, an aggregation risk',
                    'Interface_Hydrophobicity': 'hydrophobic fraction of the interface residues',
                    'Binder_Helix_Fraction': 'helical content of the binder',
                    'Binder_BetaSheet_Fraction': 'sheet content of the binder',
                    'Binder_Loop_Fraction': 'loop content of the binder',
                    'Interface_Helix_Fraction': 'helical content of the interface residues',
                    'Interface_BetaSheet_Fraction': 'sheet content of the interface residues',
                    'Interface_Loop_Fraction': 'loop content of the interface residues',
                    'Binder_Chain_Breaks': 'gaps in the binder backbone',
                    'Termini_Distance': 'distance between binder N and C termini'},
 'epitope steering': {'Hotspot_Contact_Fraction': 'named hotspot residues the binder actually touches',
                      'Coldspot_Contact_Fraction': 'residues the campaign was told to avoid',
                      'Epitope_Residues_Contacted': 'epitope residues in contact',
                      'Off_Epitope_Contact_Fraction': 'contact outside the intended epitope',
                      'Off_Paratope_Contact_Fraction': 'binder contact outside the intended paratope',
                      'Receptor_Chains_Contacted': 'how many target chains the binder reaches',
                      'Target_Crop_Length': 'residues of the target that were predicted'},
 'multi-target, cross-reactivity and detargeting': {'i_pTM_detarget': 'interface pTM against a state the binder should miss',
                                                    'i_pAE_detarget': 'interface pAE against a detarget state',
                                                    'Interface_Residues_detarget': 'interface it still forms on a detarget state'},
 'scaffolded formats (VHH, scFv, ARP)': {'Scaffold_Sequence_Retained_Fraction': 'scaffold sequence kept after redesign',
                                            'Scaffold_Framework_RMSD': 'drift of the templated framework',
                                            'Framework_Packing_Fraction': 'framework residues packing against the target',
                                            'Interdomain_Contact_Fraction': 'contact between the two domains of an scFv',
                                            'Domain_Separation_Ratio': 'how far the domains have drifted apart'},
 'cyclic peptide': {'Cyclic_Closure_Distance': 'gap left between the peptide termini'},
 'termini steering and fusion': {'N_Terminus_Away_Cosine': 'how far the N terminus points away from the target',
                                 'C_Terminus_Away_Cosine': 'how far the C terminus points away from the target',
                                 'Termini_Away_Cosine': 'both termini pooled, for a binder that has to be fused',
                                 'Terminus_Exposure': 'solvent exposure of the terminal residues'},
 'induced fit and fold switching': {'Induced_Fit_RMSD': 'conformational change between bound and unbound binder',
                                    'Induced_Fit_Interface_RMSD': 'the same change over interface residues only',
                                    'Induced_Fit_TM': 'fold similarity between the two states, low means switched'},
 'multi-chain binders and oligomers': {'Oligomer_Symmetry_RMSD': 'how far the protomers depart from symmetry'},
 'developability and immunogenicity': {'Binder_Disulfides': 'disulfide bonds formed in the binder',
                                       'Binder_Free_Cysteines': 'unpaired cysteines left behind',
                                       'Binder_Extinction': 'extinction coefficient, for quantitation',
                                       'MHC_Anchor_Score': 'predicted MHC anchor load, an immunogenicity proxy',
                                       'Protease_Site_Score': 'expected cleavage fraction on a serum protease panel',
                                       'Exposed_Loop_Fraction': 'exposed loop the binder carries, a protease liability',
                                       'Binder_Mass_kDa': 'mass of the binder assembly',
                                       'Binder_pI': 'isoelectric point of the binder',
                                       'Interface_<AA>_Count': 'one count per amino acid at the interface, e.g. Interface_W_Count'},
 'read off the binder sequence': {'Binder_Length': 'residues in the binder',
                                  'Binder_Net_Charge': 'K+R minus D+E',
                                  'Binder_Hydrophobic_Fraction': 'hydrophobic fraction of the sequence',
                                  'Binder_Cysteines': 'cysteines in the sequence'}}

AGGREGATE_HELP = ('per-target aggregates, written <metric>_<aggregate>, are computed for any check a campaign recorded once per target:',
                  '  _worst        the weakest target, which is what a cross-reactive binder is really worth',
                  '  _best         the strongest target',
                  '  _mean         the mean over targets',
                  '  _spread       best minus worst, low when the binder treats every target alike',
                  '  _selectivity  worst design target minus best detarget state, high when the binder discriminates')

def design_column(rows: list[dict]) -> str:
    return next((name for name in DESIGN_COLUMNS if rows and name in rows[0]), '')

def campaign_settings(project_folder: str) -> dict:
    for filename in METADATA_FILENAMES:
        path = os.path.join(project_folder, filename)
        if os.path.exists(path):
            return json.loads(Path(path).read_text()).get('settings', {})
    return {}

def detarget_target_names(settings: dict) -> set[str]:
    return {target['name'] for target in settings.get('targets', []) if target.get('objective') == 'detarget'}

def declared_direction(metric: str, settings: dict | None=None) -> bool | None:
    entry = (settings or {}).get('filters', {}).get(metric.partition('.')[0], {})
    if not isinstance(entry, dict) or 'higher' not in entry or entry.get('mandatory', True) is False:
        return None
    return bool(entry['higher']) if numeric_value(entry.get('threshold')) is not None else None

def metric_is_higher_better(metric: str, settings: dict | None=None) -> bool:
    declared = declared_direction(metric, settings)
    return declared if declared is not None else metric.partition('.')[0] not in LOWER_IS_BETTER_METRICS

def metric_state_columns(rows: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in rows[0] if rows else {}:
        grouped.setdefault(name.partition('.')[0], []).append(name)
    families = {}
    for base, columns in grouped.items():
        readings, kept = ({name: tuple(numeric_value(row.get(name)) for row in rows) for name in columns}, [])
        for name in sorted(columns, key=lambda name: ('.' not in name, name)):
            if any(value is not None for value in readings[name]) and readings[name] not in [readings[held] for held in kept]:
                kept.append(name)
        if len(kept) > 1:
            families[base] = kept
    return families

def state_readings(row: dict, base: str, families: dict[str, list[str]]) -> dict[str, float]:
    #per-target readings are one collapsed cell now ("0.85;0.30"); fall back to legacy base.state columns for older tables
    collapsed = per_target_readings(row, base)
    if len(collapsed) > 1:
        return collapsed
    legacy = {name.partition('.')[2]: numeric_value(row.get(name)) for name in families.get(base, [])}
    return {state: value for state, value in legacy.items() if value is not None}

def per_target_metric_bases(rows: list[dict], families: dict[str, list[str]]) -> set[str]:
    bases = set(families)
    for row in rows:
        for name in row:
            if name not in TEXT_COLUMNS and len(per_target_readings(row, name)) > 1:
                bases.add(name)
    return bases

def derive_state_metrics(rows: list[dict], settings: dict) -> dict[str, bool]:
    detargets = detarget_target_names(settings)
    families = metric_state_columns(rows)
    directions = {}
    for base in per_target_metric_bases(rows, families):
        higher = metric_is_higher_better(base, settings)
        for row in rows:
            per_target = state_readings(row, base, families)
            if len(per_target) < 2:
                continue
            #on-target states only, for detargeting; the detarget targets are gated by their own inverted filters
            on = {name: value for name, value in per_target.items() if name not in detargets} or per_target
            off = {name: value for name, value in per_target.items() if name in detargets}
            on_readings = list(on.values())
            row.setdefault(f'{base}_mean', sum(on_readings) / len(on_readings))
            row.setdefault(f'{base}_worst', min(on_readings) if higher else max(on_readings))
            row.setdefault(f'{base}_best', max(on_readings) if higher else min(on_readings))
            row.setdefault(f'{base}_spread', max(on_readings) - min(on_readings))
            if off:
                worst_on_target = min(on.values()) if higher else max(on.values())
                best_off_target = max(off.values()) if higher else min(off.values())
                row.setdefault(f'{base}_selectivity', worst_on_target - best_off_target if higher else best_off_target - worst_on_target)
        directions.update({f'{base}_mean': higher, f'{base}_worst': higher, f'{base}_best': higher, f'{base}_spread': False, f'{base}_selectivity': True})
    return {name: higher for name, higher in directions.items() if any(name in row for row in rows)}

def derive_sequence_metrics(rows: list[dict]) -> dict[str, bool]:
    for row in rows:
        #multi-chain binders are recorded chain by chain
        sequence = (row.get('Binder_Sequence') or '').replace('/', '').strip()
        if not sequence:
            continue
        row.setdefault('Binder_Length', float(len(sequence)))
        row.setdefault('Binder_Net_Charge', float(sum(map(sequence.count, 'KR')) - sum(map(sequence.count, 'DE'))))
        row.setdefault('Binder_Hydrophobic_Fraction', sum(map(sequence.count, HYDROPHOBIC_SEQUENCE_LETTERS)) / len(sequence))
        row.setdefault('Binder_Cysteines', float(sequence.count('C')))
    derived = ('Binder_Length', 'Binder_Net_Charge', 'Binder_Hydrophobic_Fraction', 'Binder_Cysteines')
    return {name: metric_is_higher_better(name) for name in derived if any(name in row for row in rows)}

def design_structure(project_folder: str, design: str, target: str='') -> str:
    for folder in STRUCTURE_FOLDERS:
        for pattern in (f'{design}.cif', f'{design}.pdb', *((f'{design}_{target}.cif', f'{design}_{target}_*.cif', f'{design}_{target}.pdb') if target else ())):
            kept = sorted(glob.glob(os.path.join(project_folder, folder, pattern)))
            if kept:
                return kept[0]
    return ''

def structure_metric_names() -> set[str]:
    try:
        from bindcraft.filters import REGISTERED_FILTER_METRICS
        from bindcraft.score import CAMPAIGN_ONLY_METRICS
    except ImportError:
        return set()
    return {name for name in REGISTERED_FILTER_METRICS if name not in CAMPAIGN_ONLY_METRICS and (not name.endswith('_detarget'))}

def recompute_structure_metrics(project_folder: str, rows: list[dict], settings: dict, metrics: tuple[str, ...]=()) -> dict[str, bool]:
    from bindcraft.score import score_design
    print(f'scoring {len(rows)} kept structure(s) again for the checks this campaign never recorded', flush=True)
    target = next(iter(settings.get('targets', ())), {})
    design, recomputed = design_column(rows), set()
    for row in rows:
        structure = design_structure(project_folder, row.get(design, ''), target.get('name', ''))
        if not structure:
            continue
        scores = {name: value for name, value in score_design(structure, hotspots=target.get('hotspots', ''), coldspots=target.get('coldspots', '')).items() if isinstance(value, (int, float))}
        row.update({name: value for name, value in scores.items() if name not in row})
        recomputed.update(scores)
        if metrics and (not set(metrics) & recomputed):
            break
    return {name: metric_is_higher_better(name, settings) for name in recomputed}

def recorded_metrics(rows: list[dict]) -> dict[str, bool]:
    return {name: metric_is_higher_better(name) for name in (rows[0] if rows else {}) if name not in TEXT_COLUMNS and any(recorded_target_values(row.get(name)) for row in rows)}

def ordering_value(row: dict, metric: str, higher: bool) -> float:
    value = on_target_mean(row, metric)
    return math.inf if value is None else -value if higher else value

def rank_designs(rows: list[dict], metrics: list[str], directions: dict[str, bool]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: tuple(ordering_value(row, metric, directions.get(metric, True)) for metric in metrics))
    return [{**row, 'rank': rank} for rank, row in enumerate(ordered, start=1)]

def ranked_column_order(rows: list[dict], metrics: list[str]) -> list[str]:
    leading = list(dict.fromkeys(['rank', design_column(rows)] + metrics))
    return [name for name in leading if name] + [name for name in dict.fromkeys(name for row in rows for name in row) if name not in leading]

def ranked_filename(metrics: list[str]) -> str:
    named = re.sub(r'[^0-9A-Za-z]+', '_', '_'.join(metrics)).strip('_')
    return f'ranked_by_{named}.csv'

def registered_extras() -> list[str]:
    catalogued = {name for modality in MODALITY_METRICS.values() for name in modality}
    return sorted(name for name in structure_metric_names() if name not in catalogued and (not re.fullmatch(r'Interface_[A-Z]_Count', name)))

def metric_guide_text(with_registry: bool=False, purpose: str='rank') -> str:
    lines = [f'metrics to {purpose} on, by the modality they belong to:']
    for modality, metrics in MODALITY_METRICS.items():
        lines += [f'  {modality}'] + [f'    {name:<29}{meaning}' for name, meaning in metrics.items()]
    extras = registered_extras() if with_registry else []
    lines += ['  also registered in this build'] + [f'    {name}' for name in extras] if extras else lines
    return '\n'.join(lines + ['', *AGGREGATE_HELP, '',
                              'a check recorded once per target is also addressable per target, written <metric>.<target name>.',
                              'a check the campaign never recorded is recomputed from the structures it kept.',
                              f'run "bindcraft {purpose} <campaign folder> --list" for the metrics this campaign can be {purpose}ed on now.'])

def available_metrics_text(project_folder: str, recorded: dict[str, bool], derived: dict[str, bool], settings: dict, purpose: str='rank') -> str:
    recomputable = [name for name in structure_metric_names() if name not in recorded and name not in derived]
    sections = (('recorded for these designs', recorded), ('derived from those readings', derived))
    lines = [f'{project_folder}: metrics available to {purpose} on']
    for heading, metrics in sections:
        lines += [f'  {heading}'] + [f"    {name:<40}{'higher is better' if higher else 'lower is better'}" for name, higher in sorted(metrics.items())]
    lines += ['  recomputed from the structures this campaign kept, where the structure holds what the check needs'] + [f"    {name:<40}{'higher is better' if metric_is_higher_better(name, settings) else 'lower is better'}" for name in sorted(recomputable)]
    return '\n'.join(lines)

def design_rows(campaign: str, table: str, binder: str='', target: str='', rescore: bool=False) -> tuple[str, list[dict]]:
    if campaign.endswith('.csv'):
        return os.path.dirname(campaign) or '.', read_metric_rows(campaign)
    recorded = read_metric_rows(stage_table(campaign, DESIGN_TABLES[table]))
    paths = [] if recorded else structure_paths(campaign)
    if recorded or not paths:
        return campaign, recorded
    kept = {row['design']: row for row in (read_metric_rows(os.path.join(campaign, SCORED_FILENAME)) if not rescore else [])}
    if all(Path(path).stem in kept for path in paths):
        print(f'{campaign}: reading the {len(paths)} score(s) kept beside the structures', flush=True)
        return campaign, [kept[Path(path).stem] for path in paths]
    try:
        from bindcraft.score import score_structure_folder
    except ImportError as unavailable:
        raise ValueError(f'{campaign} holds {len(paths)} structure(s) with no {SCORED_FILENAME} to read, and scoring them needs the design runtime, which this environment cannot load: {unavailable}')
    return campaign, score_structure_folder(campaign, binder, target, rescore=rescore)

def nothing_to_rank(campaign: str, table: str) -> str:
    return (f'nothing to rank in {campaign}: no {DESIGN_TABLES[table]}/{STAGE_TABLE_NAMES[DESIGN_TABLES[table]]} and no cif or pdb structures.\n'
            f'Name a campaign folder, one of its tables as a .csv, or any folder of binder structures, '
            f'which are scored off their coordinates whether a campaign wrote them or not.')

def default_ranking_metrics(directions: dict[str, bool], campaign: str) -> list[str]:
    for metric in (RANKING_METRIC, CONFIDENCE_METRIC):
        if metric in directions:
            return [metric]
    raise ValueError(f'{campaign} records neither {RANKING_METRIC} nor {CONFIDENCE_METRIC}, so name what to rank on with --on; --list says what it offers')

def rerank_campaign(campaign: str, metrics: list[str]=(), table: str='accepted', higher: bool | None=None, output: str='', binder: str='', target: str='', rescore: bool=False) -> tuple[str, list[dict], dict[str, bool]]:
    project_folder, rows = design_rows(campaign, table, binder, target, rescore)
    if not rows:
        raise ValueError(nothing_to_rank(campaign, table))
    settings = campaign_settings(project_folder)
    directions = {**recorded_metrics(rows), **derive_sequence_metrics(rows), **derive_state_metrics(rows, settings)}
    metrics = list(metrics) or default_ranking_metrics(directions, campaign)
    unrecorded = set(metrics) - set(directions)
    if unrecorded and unrecorded <= structure_metric_names():
        directions.update(recompute_structure_metrics(project_folder, rows, settings, tuple(unrecorded)))
    missing = [metric for metric in metrics if metric not in directions]
    if missing:
        raise ValueError(f"{', '.join(missing)} is not recorded and could not be computed; run with --list to see what this campaign offers")
    directions.update({metric: higher for metric in metrics} if higher is not None else {})
    ranked_rows = rank_designs(rows, metrics, directions)
    return write_csv_rows(ranked_rows, output or os.path.join(stage_folder(project_folder, DESIGN_TABLES[table]), ranked_filename(metrics)), ranked_column_order(ranked_rows, metrics)), ranked_rows, directions

def print_ranking(path: str, ranked_rows: list[dict], metrics: list[str], directions: dict[str, bool], top: int) -> None:
    print(', '.join(f"{metric} ({'higher' if directions[metric] else 'lower'} is better)" for metric in metrics) + f' over {len(ranked_rows)} design(s)')
    design = design_column(ranked_rows)
    for row in ranked_rows[:top]:
        readings = '  '.join(f"{metric}={on_target_mean(row, metric):g}" if on_target_mean(row, metric) is not None else f'{metric}=unmeasured' for metric in metrics)
        print(f"{row['rank']:>4}  {row.get(design, ''):<32} {readings}")
    print(f'{path}', flush=True)

def main(arguments: list[str] | None=None) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser = argparse.ArgumentParser(prog='bindcraft rank', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=metric_guide_text(bool({'-h', '--help'} & set(arguments))),
                                     description='Rerank the designs of a finished campaign on whichever check matters for the molecule you want')
    parser.add_argument('campaign', nargs='?', default='.', help='a campaign folder, one of its tables as a .csv, or any folder of binder structures (default: the current folder)')
    parser.add_argument('--on', action='append', default=[], metavar='METRIC', help=f'what to rank on (default: {RANKING_METRIC}); repeat it to break ties on the next check')
    parser.add_argument('--table', choices=sorted(DESIGN_TABLES), default='accepted', help='which designs to rank (default: accepted)')
    parser.add_argument('--lowest-first', dest='higher', action='store_const', const=False, default=None, help='rank the smallest reading first, against the metric of the check')
    parser.add_argument('--highest-first', dest='higher', action='store_const', const=True, help='rank the largest reading first, against the metric of the check')
    parser.add_argument('--top', type=int, default=20, help='how many designs to print, the whole ranking is written either way (default: 20)')
    parser.add_argument('--list', action='store_true', help='print the metrics this campaign can be ranked on, and rank nothing')
    parser.add_argument('--binder', default='', help='the binder chain(s) of a scored structure, written "B" or "B,C" (default: every chain but the first)')
    parser.add_argument('--target', default='', help='the target chain(s) of a scored structure (default: the first chain, as an accepted design is written)')
    parser.add_argument('--rescore', action='store_true', help='score every structure again rather than reading the scores kept beside them')
    parser.add_argument('--output', '-o', default='', help='where to write the ranking (default: ranked_by_<metric>.csv in the campaign folder)')
    parsed = parser.parse_args(arguments)
    try:
        if parsed.list:
            project_folder, rows = design_rows(parsed.campaign, parsed.table, parsed.binder, parsed.target, parsed.rescore)
            if not rows:
                raise ValueError(nothing_to_rank(parsed.campaign, parsed.table))
            settings = campaign_settings(project_folder)
            recorded = recorded_metrics(rows)
            return print(available_metrics_text(parsed.campaign, recorded, {**derive_sequence_metrics(rows), **derive_state_metrics(rows, settings)}, settings))
        path, ranked_rows, directions = rerank_campaign(parsed.campaign, parsed.on, parsed.table, parsed.higher, parsed.output, parsed.binder, parsed.target, parsed.rescore)
        print_ranking(path, ranked_rows, parsed.on or default_ranking_metrics(directions, parsed.campaign), directions, parsed.top)
    except (ValueError, OSError) as refusal:
        print(f'ranking refused:\n{refusal}', file=sys.stderr)
        raise SystemExit(2)
if __name__ == '__main__':
    main()
