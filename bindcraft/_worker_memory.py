"""Private Apple, CPU-only accounting helpers; no backend or model initialization.

These are provisional admission allowances, not calibrated peak measurements.
Resource ceilings describe child capacity after reserve/coordinator deductions.
"""

GIB = 1 << 30
MIB = 1 << 20


def pinned_settings(settings: dict, profile: dict) -> dict:
    # Ownership, signature freshness and lifecycle enforcement belong to the
    # admission gate. The planner only consumes its supplied execution snapshot.
    if (not isinstance(profile, dict) or not isinstance(profile.get('signature'), str)
            or not profile['signature'] or not isinstance(profile.get('overrides'), dict)
            or type(profile.get('rung')) is not int or not 0 <= profile['rung'] <= 7):
        raise ValueError('execution_profile requires a signature, overrides and rung 0 through 7')
    return {**settings, **profile['overrides']}


def validate_resource_budget(budget: dict | None) -> None:
    if budget is not None and (not isinstance(budget, dict) or any(
            not isinstance(name, str) or type(value) is not int or not 0 <= value < 1 << 63
            for name, value in budget.items())):
        raise ValueError('resource_budget must map resource names to nonnegative integer byte ceilings')


def resource_capacity(name: str, available: int, budget: dict | None) -> int:
    validate_resource_budget(budget)
    if budget is not None and name not in budget:
        raise ValueError(f'previous campaign resource_budget has no ceiling for {name}')
    return min(available, budget[name]) if budget is not None else available
