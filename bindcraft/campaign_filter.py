import argparse
import json
import operator
import os
import sys
from pathlib import Path
from typing import NamedTuple
from bindcraft.campaign_output import DEFAULT_PROJECT_FOLDER, RANKING_METRIC, numeric_value, on_target_mean, per_target_readings, stage_folder, write_csv_rows
from bindcraft.rank import DESIGN_TABLES, available_metrics_text, campaign_settings, detarget_target_names, derive_sequence_metrics, derive_state_metrics, design_column, design_rows, metric_guide_text, metric_is_higher_better, metric_state_columns, rank_designs, ranked_column_order, recompute_structure_metrics, recorded_metrics, structure_metric_names

FILTERED_FILENAME = 'filtered.csv'
FAILED_COLUMN = 'failed_thresholds'
COMPARISONS = {'>=': operator.ge, '<=': operator.le, '!=': operator.ne, '==': operator.eq, '>': operator.gt, '<': operator.lt}
AT_LEAST_AS_GOOD = '='

class DesignThreshold(NamedTuple):
    metric: str
    comparison: str
    value: float

    def holds(self, reading: float) -> bool:
        return COMPARISONS[self.comparison](reading, self.value)

def parse_threshold(expression: str, directions: dict[str, bool]) -> DesignThreshold:
    for comparison in COMPARISONS:
        metric, separator, written = expression.partition(comparison)
        if separator and numeric_value(written) is not None:
            return DesignThreshold(metric.strip(), comparison, numeric_value(written))
    metric, separator, written = expression.partition(AT_LEAST_AS_GOOD)
    if not separator or numeric_value(written) is None:
        raise ValueError(f'a threshold is written METRIC>=VALUE, METRIC<=VALUE, or METRIC=VALUE for whichever of those the check wants; got {expression!r}')
    metric = metric.strip()
    return DesignThreshold(metric, '>=' if directions.get(metric, metric_is_higher_better(metric)) else '<=', numeric_value(written))

def declared_thresholds(filters: dict) -> list[DesignThreshold]:
    thresholds = []
    for name, entry in sorted(filters.items()):
        value = numeric_value(entry.get('threshold') if isinstance(entry, dict) else entry)
        if value is not None:
            thresholds.append(DesignThreshold(name, '>=' if (entry.get('higher', True) if isinstance(entry, dict) else True) else '<=', value))
    return thresholds

def read_filters_file(path: str) -> dict:
    declared = json.loads(Path(path).read_text())
    return declared.get('filters', declared) if isinstance(declared, dict) else {}

def table_columns(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(name for row in rows for name in row))

def threshold_columns(threshold: DesignThreshold, columns: list[str], families: dict[str, list[str]], detargets: set[str]=frozenset()) -> list[str]:
    per_state = families.get(threshold.metric) or [name for name in columns if name.startswith(f'{threshold.metric}.')]
    if per_state:
        return [name for name in per_state if name.partition('.')[2] not in detargets] or per_state
    return [threshold.metric] if threshold.metric in columns else []

def threshold_readings(row: dict, column: str, detargets: set[str]) -> list[float]:
    #a per-target metric is one collapsed cell ("0.65;0.54"); test each attract target's reading, dropping the detarget targets (their own inverted filters gate them)
    per_target = per_target_readings(row, column)
    if len(per_target) > 1:
        attract = {name: reading for name, reading in per_target.items() if name not in detargets}
        return list((attract or per_target).values())
    return list(per_target.values())

def failed_thresholds(row: dict, thresholds: list[DesignThreshold], columns: list[str], families: dict[str, list[str]], detargets: set[str]=frozenset()) -> list[str]:
    missed = []
    for threshold in thresholds:
        readings = [reading for column in threshold_columns(threshold, columns, families, detargets) for reading in threshold_readings(row, column, detargets)]
        if not readings or any(not threshold.holds(reading) for reading in readings):
            missed.append(threshold.metric)
    return missed

def threshold_report(rows: list[dict], thresholds: list[DesignThreshold], columns: list[str], families: dict[str, list[str]], detargets: set[str]=frozenset()) -> list[str]:
    missed_by_design = [failed_thresholds(row, thresholds, columns, families, detargets) for row in rows]
    lines = []
    for threshold in thresholds:
        kept = sum(threshold.metric not in missed for missed in missed_by_design)
        alone = sum(missed == [threshold.metric] for missed in missed_by_design)
        targets = len(threshold_columns(threshold, columns, families, detargets))
        lines.append(f'  {threshold.metric:<32}{threshold.comparison:>3} {threshold.value:<9g} keeps {kept:>4} of {len(rows)}'
                     f'{f", and alone rejects {alone}" if alone else ""}{f", on each of {targets} targets" if targets > 1 else ""}')
    return lines

def campaign_thresholds(settings: dict) -> list[DesignThreshold]:
    return declared_thresholds(settings.get('filters', {}))

def requested_thresholds(expressions: list[str], filters_path: str, directions: dict[str, bool], settings: dict) -> list[DesignThreshold]:
    if not expressions and (not filters_path):
        return campaign_thresholds(settings)
    named = [parse_threshold(expression, directions) for expression in expressions]
    return named + declared_thresholds(read_filters_file(filters_path)) if filters_path else named

def refilter_campaign(campaign: str, expressions: list[str]=(), filters_path: str='', table: str='candidates', output: str='') -> tuple[list[dict], list[dict], list[str], str]:
    project_folder, rows = design_rows(campaign, table)
    if not rows:
        raise ValueError(f'no designs recorded in {campaign}')
    settings = campaign_settings(project_folder)
    directions = {**recorded_metrics(rows), **derive_sequence_metrics(rows), **derive_state_metrics(rows, settings)}
    thresholds = requested_thresholds(list(expressions), filters_path, directions, settings)
    if not thresholds:
        raise ValueError(f'no thresholds to reapply: {campaign} kept no campaign record to read its own filters from, so name one as --where METRIC>=VALUE or a file as --filters')
    columns, families, detargets = table_columns(rows), metric_state_columns(rows), detarget_target_names(settings)
    absent = {threshold.metric for threshold in thresholds if not threshold_columns(threshold, columns, families, detargets)}
    if absent and (expressions or filters_path) and absent <= structure_metric_names():
        recompute_structure_metrics(project_folder, rows, settings, tuple(absent))
        columns = table_columns(rows)
        absent = {threshold.metric for threshold in thresholds if not threshold_columns(threshold, columns, families, detargets)}
    if absent and (expressions or filters_path):
        raise ValueError(f"{', '.join(sorted(absent))} is not recorded and could not be computed; run with --list to see what this campaign offers")
    thresholds = [threshold for threshold in thresholds if threshold.metric not in absent]
    report = threshold_report(rows, thresholds, columns, families, detargets)
    passing, rejected = [], []
    for row in rows:
        missed = failed_thresholds(row, thresholds, columns, families, detargets)
        (rejected if missed else passing).append({**row, FAILED_COLUMN: ','.join(missed)} if missed else row)
    report.append(f'  {"every threshold together":<32}    {"":<9} keeps {len(passing):>4} of {len(rows)}')
    if absent:
        report.append(f'  not recorded in {DESIGN_TABLES[table]}, so not reapplied: {", ".join(sorted(absent))}')
    ranked = rank_designs(passing, [RANKING_METRIC], directions)
    written = output or os.path.join(stage_folder(project_folder, DESIGN_TABLES[table]), FILTERED_FILENAME)
    write_csv_rows(ranked, written, ranked_column_order(ranked, [RANKING_METRIC]))
    return ranked, rejected, report, written

def rejected_column_order(rows: list[dict]) -> list[str]:
    leading = [name for name in (design_column(rows), FAILED_COLUMN) if name]
    return leading + [name for name in table_columns(rows) if name not in leading]

def main(arguments: list[str] | None=None) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser = argparse.ArgumentParser(prog='bindcraft filter', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=metric_guide_text(bool({'-h', '--help'} & set(arguments)), 'filter'),
                                     description="Reapply thresholds to a finished campaign, to see what a different filter would have kept")
    parser.add_argument('campaign', nargs='?', default=DEFAULT_PROJECT_FOLDER, help='the campaign folder, or one recorded table to filter directly')
    parser.add_argument('--where', action='append', default=[], metavar='METRIC>=VALUE', help='a threshold to clear, where METRIC=VALUE asks for at least that good; repeat it and every one has to hold')
    parser.add_argument('--filters', default='', metavar='FILE', help='a filters block as a campaign declares one, or a settings file that declares one')
    parser.add_argument('--table', choices=sorted(DESIGN_TABLES), default='candidates', help='which designs to filter (default: candidates, every sequence the campaign validated)')
    parser.add_argument('--rejected', default='', metavar='FILE', help='also write the designs that miss, each with the thresholds it missed')
    parser.add_argument('--top', type=int, default=20, help='how many passing designs to print, all of them are written either way (default: 20)')
    parser.add_argument('--list', action='store_true', help='print the metrics this campaign can be filtered on, and filter nothing')
    parser.add_argument('--output', '-o', default='', help='where to write the passing designs (default: filtered.csv in the campaign folder)')
    parsed = parser.parse_args(arguments)
    try:
        if parsed.list:
            project_folder, rows = design_rows(parsed.campaign, parsed.table)
            settings = campaign_settings(project_folder)
            return print(available_metrics_text(parsed.campaign, recorded_metrics(rows), {**derive_sequence_metrics(rows), **derive_state_metrics(rows, settings)}, settings, purpose='filter'))
        ranked, rejected, report, written = refilter_campaign(parsed.campaign, parsed.where, parsed.filters, parsed.table, parsed.output)
        print('\n'.join(report))
        design = design_column(ranked)
        for row in ranked[:parsed.top]:
            reading = on_target_mean(row, RANKING_METRIC)
            print(f"{row['rank']:>4}  {row.get(design, ''):<40} {RANKING_METRIC}={reading:g}" if reading is not None else f"{row['rank']:>4}  {row.get(design, '')}")
        print(written, flush=True)
        if parsed.rejected and rejected:
            print(write_csv_rows(rejected, parsed.rejected, rejected_column_order(rejected)), flush=True)
    except (ValueError, OSError) as refusal:
        print(f'filtering refused:\n{refusal}', file=sys.stderr)
        raise SystemExit(2)
if __name__ == '__main__':
    main()
