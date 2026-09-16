import fcntl
import functools
import hashlib
import io
import json
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bindcraft.settings import BinderDesignSettings, PreparedTargetState
    from bindcraft.af2 import AlphaFoldDesignModel
    from bindcraft.proteinmpnn import ProteinMPNNSequenceModel
from bindcraft.campaign_log import ALWAYS_SUPPRESSED_METRICS, VERSION, campaign_metadata_written, length_choices, sampled_binder_lengths
from bindcraft.protein import ResidueFlags, is_binder_chain, read_structure_metadata, recorded_number, stamp_keyword, written_chains

DEFAULT_PROJECT_FOLDER = 'Binders'
CAMPAIGN_METADATA_NAME = 'campaign_metadata'
TRAJECTORY_STAGE, REFOLD_STAGE, RANK_STAGE = '1_Trajectories', '2_Refolded', '3_Ranked'
DESIGN_STAGES = (TRAJECTORY_STAGE, REFOLD_STAGE, RANK_STAGE)
STAGE_TABLE_NAMES = {TRAJECTORY_STAGE: '!_Trajectories.csv', REFOLD_STAGE: '!_Refolded.csv', RANK_STAGE: '!_Ranked.csv'}
ACCEPTED_FILENAME = 'accepted.csv'
LEGACY_STAGE_TABLES = {TRAJECTORY_STAGE: 'trajectories.csv', REFOLD_STAGE: 'candidates.csv', RANK_STAGE: 'ranked.csv'}
LEGACY_STAGE_FOLDERS = {TRAJECTORY_STAGE: 'trajectories', REFOLD_STAGE: 'refolded', RANK_STAGE: 'accepted'}
STRUCTURE_SUFFIXES = ('.cif', '.pdb', '.mmcif', '.ent')
SCORED_FILENAME = 'scored.csv'
CONFIDENCE_METRIC = 'Binder_pLDDT'
RANKING_METRIC = 'i_pDAE'
TARGET_VALUE_SEPARATOR = ';'
TARGET_NAME_COLUMN = 'targets'
TARGET_WEIGHT_COLUMN = 'target_weights'
REDESIGNED_FILENAME = '.redesigned_sequences.txt'
SUMMARY_FIELDS = ('campaign', 'scope', 'metric', 'samples', 'mean', 'std', 'min', 'max')
SUMMARY_FILENAME = 'summary.csv'
TRAJECTORY_ARCHIVE_SUFFIX = '.zip'
TRAJECTORY_SCOPE = 'trajectory'

def written_before_the_stages(project_folder: str) -> bool:
    if any(os.path.isdir(os.path.join(project_folder, stage)) for stage in DESIGN_STAGES):
        return False
    return any(os.path.exists(os.path.join(project_folder, name)) for name in LEGACY_STAGE_TABLES.values()) or any(os.path.isdir(os.path.join(project_folder, name)) for name in LEGACY_STAGE_FOLDERS.values())

def stage_folder(project_folder: str, stage: str) -> str:
    return os.path.join(project_folder, LEGACY_STAGE_FOLDERS.get(stage, stage) if written_before_the_stages(project_folder) else stage)

def stage_table(project_folder: str, stage: str) -> str:
    if written_before_the_stages(project_folder):
        return os.path.join(project_folder, LEGACY_STAGE_TABLES[stage])
    return os.path.join(project_folder, stage, STAGE_TABLE_NAMES[stage])

def accepted_table(project_folder: str) -> str:
    return os.path.join(project_folder, ACCEPTED_FILENAME) if written_before_the_stages(project_folder) else stage_table(project_folder, RANK_STAGE)

@contextmanager
def locked_campaign_folder(path: str):
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    folder = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(folder, fcntl.LOCK_EX)
        yield
    finally:
        os.close(folder)

def structure_paths(folder: str) -> list[str]:
    return sorted(str(path) for path in Path(folder).iterdir() if path.suffix.lower() in STRUCTURE_SUFFIXES) if os.path.isdir(folder) else []

def csv_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline='') as csv_file:
        return max(0, sum(1 for _ in csv.reader(csv_file)) - 1)

def is_recorded_reading(value) -> bool:
    return isinstance(value, float) or (getattr(value, 'ndim', None) == 0 and getattr(value, 'dtype', None) is not None and value.dtype.kind == 'f')

DESIGN_IDENTITY_COLUMNS = ('rank', 'trajectory', 'design', 'length', 'outcome')
LEADING_CONFIDENCE_COLUMNS = (RANKING_METRIC, 'i_pTM', 'pLDDT', 'pTM', 'i_pAE', 'Unbound_Binder_pLDDT', 'Target_pLDDT')
SEQUENCE_COLUMNS = ('Binder_Sequence', 'Interface_Binder_Residues', 'Interface_Target_Residues')
TRAILING_METADATA_COLUMNS = ('Timing', 'failed_filters', 'hash', 'phase', 'round', 'terminated', 'autotuned', TARGET_NAME_COLUMN, TARGET_WEIGHT_COLUMN, 'settings_core', 'settings_modality', 'settings_property', 'settings_target', 'settings_overrides')

def column_reading_order(column: str) -> int:
    base = column.split('.')[0]
    if column in DESIGN_IDENTITY_COLUMNS:
        return 0
    if column in TRAILING_METADATA_COLUMNS:
        return 5
    if base in SEQUENCE_COLUMNS:
        return 1
    if base in LEADING_CONFIDENCE_COLUMNS or column in LEADING_CONFIDENCE_COLUMNS:
        return 2
    return 4 if base in ALWAYS_SUPPRESSED_METRICS else 3

BAND_DECLARED_ORDERS = {0: DESIGN_IDENTITY_COLUMNS, 1: SEQUENCE_COLUMNS, 2: LEADING_CONFIDENCE_COLUMNS, 5: TRAILING_METADATA_COLUMNS}

def declared_column_position(column: str, band: int) -> int:
    declared = BAND_DECLARED_ORDERS.get(band, ())
    name = column if column in declared else column.split('.')[0]
    return declared.index(name) if name in declared else len(declared)

def ordered_csv_columns(column_names) -> list[str]:
    ordered = list(dict.fromkeys(column_names))
    return [column for band in range(6) for column in sorted((name for name in ordered if column_reading_order(name) == band), key=lambda name, band=band: (declared_column_position(name, band), ordered.index(name)))]

def recorded_row(row: dict) -> dict:
    return {name: recorded_number(value) if is_recorded_reading(value) else value for name, value in row.items()}

def append_metric_row(csv_path: str, row: dict) -> None:
    previous_rows, column_names = ([], [])
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as metrics_file:
            reader = csv.DictReader(metrics_file)
            previous_rows = list(reader)
            column_names = list(reader.fieldnames or [])
    column_names = ordered_csv_columns(column_names + [name for name in row if name not in column_names])
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    partial_path = f'{csv_path}.partial'
    with open(partial_path, 'w', newline='') as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=column_names, restval='')
        writer.writeheader()
        writer.writerows(previous_rows + [recorded_row(row)])
    os.replace(partial_path, csv_path)

def append_campaign_metrics(csv_path: str, row: dict) -> None:
    with locked_campaign_folder(csv_path):
        append_metric_row(csv_path, row)

COMPLETED_TRAJECTORY = 'completed'

def recorded_rejections(project_folder: str) -> dict:
    trajectory_rows = read_metric_rows(stage_table(project_folder, TRAJECTORY_STAGE))
    candidate_rows = read_metric_rows(stage_table(project_folder, REFOLD_STAGE))
    return {'terminated': dict(Counter(row.get('terminated') or COMPLETED_TRAJECTORY for row in trajectory_rows)),
            'failed_filters': dict(Counter(name for row in candidate_rows for name in (row.get('failed_filters') or '').split(',') if name)),
            'candidates_scored': len(candidate_rows), 'candidates_rejected': sum(1 for row in candidate_rows if row.get('failed_filters'))}

class CampaignProgress:
    def __init__(self, project_folder: str, requested_designs: int, max_trajectories: int | None=None):
        self.project_folder = project_folder
        self.requested_designs = int(requested_designs)
        self.max_trajectories = max_trajectories
        self.state_path = os.path.join(project_folder, '.campaign_state.json')
        os.makedirs(project_folder, exist_ok=True)

    def recovered_state(self) -> dict:
        return {'trajectories': csv_row_count(stage_table(self.project_folder, TRAJECTORY_STAGE)), 'accepted': csv_row_count(accepted_table(self.project_folder)), 'rejections': recorded_rejections(self.project_folder), 'attempted': sorted(designed_recipe_hashes(self.project_folder))}

    @contextmanager
    def locked_progress(self):
        with locked_campaign_folder(self.state_path):
            state = json.loads(Path(self.state_path).read_text()) if os.path.exists(self.state_path) else self.recovered_state()
            yield state
            partial_path = f'{self.state_path}.partial'
            Path(partial_path).write_text(json.dumps(state, sort_keys=True))
            os.replace(partial_path, self.state_path)

    def claim_trajectory(self) -> tuple[int, int] | None:
        with self.locked_progress() as state:
            if state['accepted'] >= self.requested_designs:
                return None
            if self.max_trajectories and state['trajectories'] >= self.max_trajectories:
                return None
            state['trajectories'] += 1
            return state['trajectories'], state['accepted']

    def claim_recipe(self, identity: str) -> bool:
        with self.locked_progress() as state:
            attempted = state.setdefault('attempted', sorted(designed_recipe_hashes(self.project_folder)))
            if identity in attempted:
                return False
            attempted.append(identity)
            return True

    def record_accepted_design(self) -> int:
        with self.locked_progress() as state:
            state['accepted'] += 1
            return state['accepted']

    def campaign_status(self) -> tuple[int, int]:
        with self.locked_progress() as state:
            return state['accepted'], state['trajectories']

    def rejection_counts(self, state: dict) -> dict:
        return state.setdefault('rejections', recorded_rejections(self.project_folder))

    def record_trajectory_outcome(self, terminated_stage: str | None) -> None:
        with self.locked_progress() as state:
            terminated = self.rejection_counts(state)['terminated']
            stage = terminated_stage or COMPLETED_TRAJECTORY
            terminated[stage] = terminated.get(stage, 0) + 1

    def record_candidate_outcome(self, failed_filters) -> None:
        with self.locked_progress() as state:
            rejections = self.rejection_counts(state)
            rejections['candidates_scored'] += 1
            rejections['candidates_rejected'] += 1 if failed_filters else 0
            for name in failed_filters:
                rejections['failed_filters'][name] = rejections['failed_filters'].get(name, 0) + 1

def designed_recipe_hashes(project_folder: str) -> set[str]:
    return {row['hash'] for row in read_metric_rows(stage_table(project_folder, TRAJECTORY_STAGE)) if row.get('hash')}

def redesigned_binder_sequences(project_folder: str) -> set[str]:
    return {row['Binder_Sequence'].replace('/', '') for path in (stage_table(project_folder, REFOLD_STAGE), accepted_table(project_folder))
            for row in read_metric_rows(path) if row.get('Binder_Sequence')}

def redesigned_sequence(project_folder: str, sequence: str) -> bool:
    path = os.path.join(project_folder, REDESIGNED_FILENAME)
    with locked_campaign_folder(path):
        if not os.path.exists(path):
            Path(path).write_text(''.join(f'{drawn}\n' for drawn in sorted(redesigned_binder_sequences(project_folder))))
        if sequence in set(Path(path).read_text().split()):
            return True
        with open(path, 'a') as redesigned_file:
            redesigned_file.write(f'{sequence}\n')
        return False

def accepted_state_suffix(state: 'PreparedTargetState', protein_complex: dict) -> str:
    if not state.crop_index:
        return state.name
    residue_index = protein_complex[state.target_chain].residue_index
    return f'{state.source_target}_{int(residue_index[0]) + 1}-{int(residue_index[-1]) + 1}'

def accepted_state_suffixes(prepared_states, predictions) -> dict[str, str]:
    states = {state.name: state for state in prepared_states if state.name in predictions}
    if len(states) < 2:
        return {name: '' for name in states}
    return {name: f'_{accepted_state_suffix(state, predictions[name].protein_complex)}' for name, state in states.items()}

def trajectory_output_path(trajectory_directory: str, filename: str) -> str:
    return os.path.join(trajectory_directory, f'{os.path.basename(trajectory_directory.rstrip(os.sep))}_{filename}')

def numeric_value(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def weighted_target_order(prepared_states) -> tuple[tuple[str, float], ...]:
    return tuple((state.name, float(state.weight)) for state in sorted(prepared_states, key=lambda state: (-float(state.weight), state.name)))

def target_ordered_row(row: dict, targets: tuple[tuple[str, float], ...]) -> dict:
    if len(targets) < 2:
        return row
    target_names = [name for name, _ in targets]
    collapsed, per_target = {}, {}
    for name, value in row.items():
        base, _, state = name.rpartition('.')
        if base and state in target_names:
            per_target.setdefault(base, {})[state] = value
        else:
            collapsed[name] = value
    for base, readings in per_target.items():
        if base in collapsed:
            readings.setdefault(target_names[0], collapsed.pop(base))
        collapsed[base] = TARGET_VALUE_SEPARATOR.join('' if readings.get(state) is None else f'{recorded_number(readings[state]):g}' if isinstance(readings[state], float) else str(readings[state]) for state in target_names)
    collapsed[TARGET_NAME_COLUMN] = TARGET_VALUE_SEPARATOR.join(target_names)
    collapsed[TARGET_WEIGHT_COLUMN] = TARGET_VALUE_SEPARATOR.join(f'{weight:g}' for _, weight in targets)
    return collapsed

def recorded_target_values(value) -> list[float]:
    if value is None:
        return []
    return [number for number in (numeric_value(part) for part in str(value).split(TARGET_VALUE_SEPARATOR)) if number is not None]

def per_target_readings(row: dict, metric: str) -> dict[str, float]:
    values = recorded_target_values(row.get(metric))
    names = [name for name in str(row.get(TARGET_NAME_COLUMN) or '').split(TARGET_VALUE_SEPARATOR) if name]
    if len(names) != len(values):
        names = [str(position) for position in range(len(values))] if len(values) > 1 else ['']
    return dict(zip(names, values))

def mean_target_value(value) -> float | None:
    values = recorded_target_values(value)
    return statistics.fmean(values) if values else None

def on_target_mean(row: dict, metric: str) -> float | None:
    #rank on the attract (positive-weight) targets only: a detarget reading is already gated at acceptance (Interface_Residues_detarget), so averaging it in would reward binding the off-target
    values = recorded_target_values(row.get(metric))
    if not values:
        return None
    weights = recorded_target_values(row.get(TARGET_WEIGHT_COLUMN))
    if len(weights) == len(values):
        attract = [value for value, weight in zip(values, weights) if weight > 0]
        if attract:
            values = attract
    return statistics.fmean(values)

def read_metric_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='') as metrics_file:
        return list(csv.DictReader(metrics_file))

def timing_stamp(**fields) -> str:
    return ';'.join(f'{name}={round(value, 2) if isinstance(value, float) else value}' for name, value in fields.items())

def parse_timing(cell: str) -> dict[str, str]:
    return {name: value for name, _, value in (pair.partition('=') for pair in cell.split(';')) if name}

def campaign_timing(project_folder: str) -> dict:
    trajectory_rows, candidate_rows = read_metric_rows(stage_table(project_folder, TRAJECTORY_STAGE)), read_metric_rows(stage_table(project_folder, REFOLD_STAGE))
    designs: dict[str, dict] = {}
    for row in trajectory_rows:
        stamp = parse_timing(row.get('Timing', ''))
        if stamp:
            designs[row['design']] = {'worker': stamp.get('worker', ''), 'compiled': stamp.get('compiled') == '1', 'design_seconds': float(stamp.get('design', 0.0)), 'reprediction_seconds': 0.0}
    for row in candidate_rows:
        stamp, design = parse_timing(row.get('Timing', '')), row.get('design', '').rsplit('_candidate', 1)[0]
        if stamp and design in designs:
            designs[design]['reprediction_seconds'] += float(stamp.get('reprediction', 0.0))
    stamps = [stamp for stamp in (parse_timing(row.get('Timing', '')) for row in trajectory_rows + candidate_rows) if stamp.get('start')]
    workers = sorted({design['worker'] for design in designs.values()})
    return {'designs': designs, 'design_count': len(designs),
            'wall_seconds': max((float(stamp['start']) + float(stamp.get('design') or stamp.get('reprediction') or 0.0) for stamp in stamps), default=0.0) - min((float(stamp['start']) for stamp in stamps), default=0.0),
            'per_worker': {worker: {'designs': sum(design['worker'] == worker for design in designs.values()),
                                    'design_seconds': sum(design['design_seconds'] for design in designs.values() if design['worker'] == worker),
                                    'reprediction_seconds': sum(design['reprediction_seconds'] for design in designs.values() if design['worker'] == worker)} for worker in workers}}

def trajectory_metric_means(metric_rows: list[dict]) -> dict[tuple[str, str], float]:
    recorded_values: dict[tuple[str, str], list[float]] = {}
    for row in metric_rows:
        design_stage = row.get('phase') or TRAJECTORY_SCOPE
        for metric, value in row.items():
            number = numeric_value(value) if metric not in ('phase', 'round') else None
            if number is None:
                continue
            for scope in dict.fromkeys((design_stage, TRAJECTORY_SCOPE)):
                recorded_values.setdefault((scope, metric), []).append(number)
    return {key: statistics.fmean(values) for key, values in recorded_values.items()}

def summary_row(campaign: str, scope: str, metric: str, values: list[float]) -> dict:
    return {'campaign': campaign, 'scope': scope, 'metric': metric, 'samples': len(values), 'mean': statistics.fmean(values), 'std': statistics.pstdev(values) if len(values) > 1 else 0.0, 'min': min(values), 'max': max(values)}

def trajectory_directories(project_folder: str) -> list[str]:
    trajectories_directory = stage_folder(project_folder, TRAJECTORY_STAGE)
    return [os.path.join(trajectories_directory, name) for name in sorted(os.listdir(trajectories_directory)) if os.path.isdir(os.path.join(trajectories_directory, name))] if os.path.isdir(trajectories_directory) else []

def trajectory_archives(project_folder: str) -> list[str]:
    trajectories_directory = stage_folder(project_folder, TRAJECTORY_STAGE)
    return sorted(str(path) for path in Path(trajectories_directory).glob(f'*{TRAJECTORY_ARCHIVE_SUFFIX}')) if os.path.isdir(trajectories_directory) else []

def archived_metric_rows(archive_path: str, filename: str) -> list[dict]:
    with zipfile.ZipFile(archive_path) as archive:
        member = next((entry for entry in archive.namelist() if os.path.basename(entry).endswith(filename)), None)
        return list(csv.DictReader(io.TextIOWrapper(archive.open(member), newline=''))) if member else []

def trajectory_metric_rows(project_folder: str, filename: str='losses.csv') -> list[list[dict]]:
    return [read_metric_rows(trajectory_output_path(directory, filename)) for directory in trajectory_directories(project_folder)] + [archived_metric_rows(path, filename) for path in trajectory_archives(project_folder)]

def recorded_trajectory_rows(project_folder: str, design: str, filename: str='losses.csv') -> list[dict]:
    trajectory_directory = os.path.join(stage_folder(project_folder, TRAJECTORY_STAGE), design)
    archive_path = f'{trajectory_directory}{TRAJECTORY_ARCHIVE_SUFFIX}'
    if os.path.isdir(trajectory_directory):
        return read_metric_rows(trajectory_output_path(trajectory_directory, filename))
    return archived_metric_rows(archive_path, filename) if os.path.exists(archive_path) else []

def archive_trajectory_folder(trajectory_directory: str) -> str | None:
    if not os.path.isdir(trajectory_directory):
        return None
    archive_path = f'{trajectory_directory}{TRAJECTORY_ARCHIVE_SUFFIX}'
    partial_path = f'{archive_path}.partial'
    with zipfile.ZipFile(partial_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for directory, _, filenames in os.walk(trajectory_directory):
            for filename in sorted(filenames):
                written_path = os.path.join(directory, filename)
                archive.write(written_path, os.path.relpath(written_path, os.path.dirname(trajectory_directory)))
    os.replace(partial_path, archive_path)
    shutil.rmtree(trajectory_directory)
    return archive_path

DISCARDED_TRAJECTORY_SUFFIXES = ('.cif', '.pdb', '.npz')

def discard_trajectory_structures(trajectory_directory: str) -> int:
    discarded = [path for path in Path(trajectory_directory).rglob('*') if path.is_file() and path.suffix.lower() in DISCARDED_TRAJECTORY_SUFFIXES] if os.path.isdir(trajectory_directory) else []
    for path in discarded:
        path.unlink()
    return len(discarded)

def restore_trajectory_archive(archive_path: str) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(os.path.dirname(archive_path))
    os.remove(archive_path)
    return archive_path[:-len(TRAJECTORY_ARCHIVE_SUFFIX)]

def archive_campaign_trajectories(project_folder: str) -> list[str]:
    return [path for path in (archive_trajectory_folder(directory) for directory in trajectory_directories(project_folder)) if path]

def restore_campaign_trajectories(project_folder: str) -> list[str]:
    return [restore_trajectory_archive(archive_path) for archive_path in trajectory_archives(project_folder)]

def campaign_status_counts(project_folder: str, recorded_trajectories: int) -> dict[str, float]:
    trajectory_rows = read_metric_rows(stage_table(project_folder, TRAJECTORY_STAGE))
    accepted_rows = read_metric_rows(accepted_table(project_folder))
    candidate_rows = read_metric_rows(stage_table(project_folder, REFOLD_STAGE))
    terminated_stages = Counter((row.get('terminated') or 'completed') for row in trajectory_rows)
    counts = {'trajectories': float(max(len(trajectory_rows), recorded_trajectories)), 'redesign_candidates': float(len(candidate_rows)), 'accepted_designs': float(len(accepted_rows))}
    counts.update({f'terminated:{stage}': float(count) for stage, count in sorted(terminated_stages.items())})
    return counts

def summarize_campaign(project_folder: str, campaign: str | None=None) -> list[dict]:
    campaign = campaign or os.path.basename(os.path.normpath(project_folder))
    recorded_values: dict[tuple[str, str], list[float]] = {}
    recorded_trajectories = 0
    for metric_rows in trajectory_metric_rows(project_folder):
        if not metric_rows:
            continue
        recorded_trajectories += 1
        for key, value in trajectory_metric_means(metric_rows).items():
            recorded_values.setdefault(key, []).append(value)
    for scope, path in (('final', stage_table(project_folder, TRAJECTORY_STAGE)), ('candidate', stage_table(project_folder, REFOLD_STAGE)), ('accepted', accepted_table(project_folder))):
        for row in read_metric_rows(path):
            for metric, value in row.items():
                if metric in ('rank', 'design', 'terminated', 'failed_filters', TARGET_NAME_COLUMN, TARGET_WEIGHT_COLUMN):
                    continue
                for target_name, number in per_target_readings(row, metric).items():
                    recorded_values.setdefault((f'{scope}:{target_name}' if target_name else scope, metric), []).append(number)
    status_rows = [summary_row(campaign, 'campaign', metric, [count]) for metric, count in campaign_status_counts(project_folder, recorded_trajectories).items()]
    return status_rows + [summary_row(campaign, scope, metric, values) for (scope, metric), values in sorted(recorded_values.items())]

def write_csv_rows(rows: list[dict], path: str, column_names=SUMMARY_FIELDS) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    partial_path = f'{path}.partial'
    with open(partial_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(column_names))
        writer.writeheader()
        writer.writerows([recorded_row(row) for row in rows])
    os.replace(partial_path, path)
    return path

def write_campaign_summary(project_folder: str, campaign: str | None=None) -> str | None:
    rows = summarize_campaign(project_folder, campaign)
    return write_csv_rows(rows, os.path.join(project_folder, SUMMARY_FILENAME)) if rows else None

def ranking_value(row: dict, metric: str) -> float:
    value = on_target_mean(row, metric)
    return -math.inf if value is None else value

def accepted_structure_present(rank_folder: str, design: str) -> bool:
    written = list(Path(rank_folder).glob(f'{design}.cif')) + list(Path(rank_folder).glob(f'{design}_*.cif'))
    return any(not path.name.endswith('_monomer.cif') for path in written)

def write_ranked_designs(project_folder: str, metric: str, reconcile: bool=False) -> str | None:
    accepted_rows = read_metric_rows(accepted_table(project_folder))
    if reconcile:
        rank_folder = stage_folder(project_folder, RANK_STAGE)
        accepted_rows = [row for row in accepted_rows if accepted_structure_present(rank_folder, row.get('design', ''))]
    if not accepted_rows:
        return None
    ranked_rows = [{**row, 'rank': rank} for rank, row in enumerate(sorted(accepted_rows, key=lambda row: -ranking_value(row, metric)), start=1)]
    return write_csv_rows(ranked_rows, stage_table(project_folder, RANK_STAGE), ordered_csv_columns(ranked_rows[0]))

def rank_accepted_designs(project_folder: str, metric: str=RANKING_METRIC) -> str | None:
    if not os.path.exists(accepted_table(project_folder)):
        return None
    with locked_campaign_folder(accepted_table(project_folder)):
        return write_ranked_designs(project_folder, metric, reconcile=True)

def append_accepted_design(project_folder: str, row: dict, metric: str=RANKING_METRIC) -> str | None:
    with locked_campaign_folder(accepted_table(project_folder)):
        append_metric_row(accepted_table(project_folder), row)
        return write_ranked_designs(project_folder, metric)

@functools.cache
def source_revision() -> str:
    source_tree = str(Path(__file__).resolve().parents[2])
    try:
        revision = subprocess.run(['git', '-C', source_tree, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        try:
            return f'bindcraft {version("bindcraft")}'
        except PackageNotFoundError:
            return 'unknown source'
    uncommitted = subprocess.run(['git', '-C', source_tree, 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
    return f'{revision}-dirty' if uncommitted else revision

def campaign_modality_switches(settings: dict) -> tuple[str, ...]:
    from bindcraft.parameter_sweep import modality_loss_weight_axes
    from bindcraft.settings import requested_campaign_features
    return tuple(feature.switch for feature in requested_campaign_features(settings) if feature.switch) + modality_loss_weight_axes(settings)

def json_compatible(value):
    if isinstance(value, dict):
        return {name: json_compatible(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value

def settings_digest(settings: dict) -> str:
    return hashlib.sha256(json.dumps(json_compatible(settings), indent=2, sort_keys=True, default=str).encode()).hexdigest()[:12]

def campaign_facts(design_settings: 'BinderDesignSettings') -> dict[str, str]:
    settings = design_settings.settings
    lengths = design_settings.binder.lengths
    facts = {'targets': ' '.join(f'{state.name}({state.objective})' for state in design_settings.prepared_states),
             'hotspots': ' '.join(f'{state.name}:{state.hotspots}' for state in design_settings.prepared_states if state.hotspots),
             'binder_length_range': f'{min(lengths)}-{max(lengths)}' if lengths else '',
             'modality_switches': ' '.join(campaign_modality_switches(settings)),
             'settings_digest': settings_digest(settings),
             'written': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    return {name: value for name, value in facts.items() if value}

def structure_metadata(campaign: str, metrics: dict, design_settings: 'BinderDesignSettings | None'=None, **labels: str) -> dict:
    scalars = {name: recorded_number(value) for name, value in metrics.items() if isinstance(value, (int, float)) or (hasattr(value, 'ndim') and value.ndim == 0)}
    return {'bindcraft_version': VERSION, 'bindcraft_revision': source_revision(), **({'campaign': campaign} if campaign else {}), **(campaign_facts(design_settings) if design_settings is not None else {}), **labels, **scalars}

def reprediction_facts(mpnn_model: 'ProteinMPNNSequenceModel', validation_model: 'AlphaFoldDesignModel') -> dict[str, str]:
    return {'mpnn_model': str(mpnn_model.model_name), 'mpnn_variant': str(mpnn_model.variant), 'mpnn_temperature': str(mpnn_model.temperature), 'validation_recycles': str(validation_model.num_recycle)}

def flagged_residue_spans(chain_letter: str, protein, flag: ResidueFlags) -> list[str]:
    numbers = [number for number, flags in zip(protein.residue_index.tolist(), protein.flags.tolist()) if flags & int(flag)]
    runs = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return [f'{chain_letter}{run[0]}' if len(run) == 1 else f'{chain_letter}{run[0]}-{run[-1]}' for run in runs]

def designed_span_stamp(protein_complex: dict, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None) -> dict[str, str]:
    #the letter a written structure gives the binder moves when a multi-chain receptor takes more than one, so the spans are read off the same split
    letters = {written.complex_chain: written.letter for written in written_chains(protein_complex, receptor_chains)}
    stamped = {}
    for name, flag in (('redesigned_residues', ResidueFlags.DESIGN), ('paratope_residues', ResidueFlags.CONTACT)):
        spans = [span for chain in sorted(protein_complex) if is_binder_chain(chain) for span in flagged_residue_spans(letters[chain], protein_complex[chain], flag)]
        if spans:
            stamped[name] = ','.join(spans)
    return stamped

def drawn_weight_stamp(drawn: dict) -> dict[str, str]:
    return {name: f'{value:g}' for name, value in drawn.items() if name.startswith('weights_')}

def model_score_stamp(model_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {f'{metric}.{model_name}': recorded_number(value) for model_name, metrics in model_metrics.items() for metric, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}

def design_model_scores(structure: str) -> dict[str, dict[str, float]]:
    stamp = read_structure_metadata(structure)
    models = tuple(stamp.get('validation_models', '').split())
    scores: dict[str, dict[str, float]] = {}
    for name, value in stamp.items():
        model_name = next((model for model in models if name.endswith(stamp_keyword(f'.{model}'))), None)
        if model_name is None:
            continue
        scores.setdefault(model_name, {})[name[:-len(model_name) - 1]] = float(value)
    return scores

def checkpoint_digest(path: str) -> str:
    with open(path, 'rb') as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()

def write_campaign_metadata(project_folder: str, settings: dict, settings_path: str, checkpoint_paths: tuple[str, ...], metadata: dict[str, str] | None=None) -> str:
    from bindcraft.settings import resolved_campaign_facts
    lengths = sampled_binder_lengths(settings)
    record = {'version': VERSION, 'revision': source_revision(), 'settings_path': str(Path(settings_path).resolve()), 'settings': settings, 'resolved': resolved_campaign_facts(settings), **({'sampled_binder_lengths': length_choices(lengths)} if lengths else {}), 'settings_digest': settings_digest(settings), 'checkpoints': {path: checkpoint_digest(path) for path in checkpoint_paths}, **({'metadata': metadata} if metadata else {})}
    record_text = json.dumps(json_compatible(record), indent=2, sort_keys=True)
    os.makedirs(project_folder, exist_ok=True)
    metadata_path = os.path.join(project_folder, f'{CAMPAIGN_METADATA_NAME}.json')
    if os.path.exists(metadata_path) and Path(metadata_path).read_text() != record_text:
        metadata_path = os.path.join(project_folder, f'{CAMPAIGN_METADATA_NAME}_{hashlib.sha256(record_text.encode()).hexdigest()[:12]}.json')
    Path(metadata_path).write_text(record_text)
    print(campaign_metadata_written(record['version'], record['settings_path'], metadata_path), flush=True)
    return metadata_path

def campaign_folders(path: str) -> list[str]:
    if os.path.exists(stage_table(path, TRAJECTORY_STAGE)) or os.path.isdir(stage_folder(path, TRAJECTORY_STAGE)):
        return [path]
    return [os.path.join(path, name) for name in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, name))] if os.path.isdir(path) else []

def main(arguments: list[str] | None=None) -> None:
    paths = list(sys.argv[1:] if arguments is None else arguments) or [DEFAULT_PROJECT_FOLDER]
    if {'-h', '--help'} & set(paths):
        print('usage: bindcraft campaign_output [<campaign folder> ...]')
        return
    combined: list[dict] = []
    for path in paths:
        for project_folder in campaign_folders(path):
            ranked_path = rank_accepted_designs(project_folder)
            if ranked_path:
                print(f'{os.path.basename(os.path.normpath(project_folder))} accepted designs ranked by {RANKING_METRIC}: {ranked_path}', flush=True)
            rows = summarize_campaign(project_folder)
            if not rows:
                continue
            write_csv_rows(rows, os.path.join(project_folder, SUMMARY_FILENAME))
            combined.extend(rows)
        if len(campaign_folders(path)) > 1:
            write_csv_rows([row for row in combined], os.path.join(path, SUMMARY_FILENAME))
    for row in combined:
        if row['scope'] == 'campaign':
            print(f"{row['campaign']} {row['metric']}={row['mean']:g}", flush=True)
    print(f'{len(combined)} summary rows over {len({row["campaign"] for row in combined})} campaign(s)', flush=True)
if __name__ == '__main__':
    main()
