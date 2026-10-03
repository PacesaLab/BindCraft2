"""Build through an sdist and exercise presets away from the source checkout."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_installed_presets_and_referenced_structures(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output), str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    installed = tmp_path / "installed"
    with zipfile.ZipFile(next(output.glob("*.whl"))) as wheel:
        wheel.extractall(installed)
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(installed),
            *filter(None, environment.get("PYTHONPATH", "").split(os.pathsep)),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from bindcraft import cli, settings
assert settings.CORE_DEFAULTS
assert cli.CAMPAIGN_PRESETS == settings.CAMPAIGN_PRESETS
assert '_resources' in settings.CAMPAIGN_PRESETS.parts
for tier in ('core', 'modality', 'property', 'target'):
    names = settings.shipped_preset_names(tier)
    assert names, tier
    for name in names:
        preset = settings.read_preset(tier, name)
        if preset.get('binder_scaffold'):
            assert Path(preset['binder_scaffold']).is_file(), (tier, name)
        for target in preset.get('targets', []):
            assert Path(target['target_path']).is_file(), (tier, name)
print('installed presets and referenced structures passed')
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "installed presets and referenced structures passed" in result.stdout
