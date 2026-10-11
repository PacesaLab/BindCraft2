"""Locate presets in an installed wheel or the existing source layout."""

from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
_INSTALLED_PRESETS = _PACKAGE / "_resources" / "settings"
CAMPAIGN_PRESETS = (
    _INSTALLED_PRESETS if _INSTALLED_PRESETS.is_dir() else _PACKAGE.parent / "settings"
)
