from bindcraft.campaign_log import desperation_warning
from bindcraft.campaign_output import TRAJECTORY_STAGE, accepted_table, read_metric_rows, stage_table
from bindcraft.settings import DEFAULT_SETTINGS

DESPERATION_TRAJECTORIES = 750
DESPERATION_RUNG_TRAJECTORIES = 50
DESPERATE_FLEXIBILITY = 0.5
DESPERATE_RECYCLES = 3
INITIAL_GUESS = {'initial_guess': True}
FLEXIBILITY = {'target_flexibility': DESPERATE_FLEXIBILITY}
MULTIMER_VALIDATION = {'validation_model': 'multimer'}
DESPERATION_LADDER = (INITIAL_GUESS, FLEXIBILITY, INITIAL_GUESS | FLEXIBILITY, MULTIMER_VALIDATION, MULTIMER_VALIDATION | INITIAL_GUESS, MULTIMER_VALIDATION | INITIAL_GUESS | FLEXIBILITY, MULTIMER_VALIDATION | INITIAL_GUESS | FLEXIBILITY | {'design_recycles': DESPERATE_RECYCLES})
DESPERATION_SETTINGS = tuple(dict.fromkeys(name for rung in DESPERATION_LADDER for name in rung))

def trajectories_since_accepted(project_folder: str) -> int:
    accepted_hashes = {row['hash'] for row in read_metric_rows(accepted_table(project_folder)) if row.get('hash')}
    trajectory_rows = read_metric_rows(stage_table(project_folder, TRAJECTORY_STAGE))
    accepted_at = max((position for position, row in enumerate(trajectory_rows, 1) if row.get('hash') in accepted_hashes), default=0)
    return len(trajectory_rows) - accepted_at

def desperation_rungs(settings: dict, fruitless_trajectories: int) -> int:
    if not settings.get('desperation', DEFAULT_SETTINGS['desperation']) or settings.get('trajectory_only'):
        return 0
    desperate_after = int(settings.get('desperation_trajectories', DESPERATION_TRAJECTORIES))
    if fruitless_trajectories < desperate_after:
        return 0
    return min(len(DESPERATION_LADDER), 1 + (fruitless_trajectories - desperate_after) // DESPERATION_RUNG_TRAJECTORIES)

def desperation_overrides(settings: dict, fruitless_trajectories: int) -> dict:
    rungs = desperation_rungs(settings, fruitless_trajectories)
    if not rungs:
        return {}
    return {name: max(int(settings.get(name) or 0), value) if name == 'design_recycles' else value for name, value in DESPERATION_LADDER[rungs - 1].items()}

def desperate_settings(settings: dict, project_folder: str) -> tuple[dict, str]:
    fruitless_trajectories = trajectories_since_accepted(project_folder)
    overrides = desperation_overrides(settings, fruitless_trajectories)
    if not overrides:
        return settings, ''
    return {**settings, **overrides}, desperation_warning(desperation_rungs(settings, fruitless_trajectories), len(DESPERATION_LADDER), fruitless_trajectories, ', '.join(f'{name} {value}' for name, value in overrides.items()))
