#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent
ENVIRONMENT_INTERPRETERS = ('.venv/bin/python', '.venv/Scripts/python.exe')
ALREADY_SWITCHED = 'BINDCRAFT_LAUNCHER_SWITCHED'
GETTING_STARTED = """BindCraft designs a binder from one settings file:

    python bindcraft.py examples/pdl1_denovo.json
    python bindcraft.py examples/pdl1_denovo.json number_of_final_designs=1 'binder_lengths=[70,90]'

Any setting of the file is overridden by naming it that way, and examples/README.md lists the
campaigns that ship with the repository. Settings, losses and filters are in docs/reference.md.
Every other bindcraft command is reached the same way: rank, score, archive, fetch-weights.
"""

def setting_assignments(arguments: list[str]) -> list[str]:
    rewritten, preceding = [], ''
    for argument in arguments:
        overrides = '=' in argument and (not argument.startswith('-')) and (not preceding.startswith('-'))
        rewritten += ['--set', argument] if overrides else [argument]
        preceding = argument
    return rewritten

def importable_repository() -> None:
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))

def environment_interpreter() -> str:
    for relative_path in ENVIRONMENT_INTERPRETERS:
        candidate = REPOSITORY / relative_path
        if candidate.exists() and candidate.resolve() != Path(sys.executable).resolve():
            return str(candidate)
    return ''

def missing_dependencies() -> tuple[str, ...]:
    try:
        from bindcraft.selfcheck import missing_modules
    except ImportError:
        return ('bindcraft',)
    return missing_modules()

def run_in_environment(arguments: list[str]) -> int:
    interpreter = environment_interpreter()
    if not interpreter:
        return -1
    print(f'bindcraft: running under {interpreter}, the environment beside this file', flush=True)
    return subprocess.run([interpreter, str(Path(__file__).resolve()), *arguments], env={**os.environ, ALREADY_SWITCHED: '1'}).returncode

def designs_a_campaign(arguments: list[str]) -> bool:
    try:
        from bindcraft import package_modules
        from bindcraft.cli import COMMAND_MODULES, COMMANDS
    except ImportError:
        return False
    named = arguments[0] if arguments else ''
    return named == 'design' or (bool(named) and named not in COMMANDS and named not in COMMAND_MODULES and named not in package_modules() and (not named.startswith('-')))

def design_gpu_note() -> str:
    from bindcraft.design_workers import visible_design_gpus
    gpus = visible_design_gpus()
    if len(gpus) > 1:
        return f'{len(gpus)} GPUs visible, designing on all of them at once'
    if gpus:
        return '1 GPU visible, designing on it alone'
    return 'no GPU visible, which leaves AlphaFold on the CPU and a trajectory too slow to finish; bindcraft.slurm submits this to GPU nodes'

def run_command_line(arguments: list[str]) -> int:
    from bindcraft.cli import main as run_command
    try:
        run_command(arguments)
    except SystemExit as finished:
        return int(finished.code or 0)
    return 0

def main(arguments: list[str] | None=None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    importable_repository()
    designs = designs_a_campaign(arguments)
    arguments = setting_assignments(arguments) if designs else arguments
    if not arguments or arguments[0] in ('-h', '--help'):
        print(GETTING_STARTED, file=sys.stdout if arguments else sys.stderr)
        return run_command_line(['--help'] if arguments else [])
    missing = missing_dependencies()
    if missing and (not os.environ.get(ALREADY_SWITCHED)) and (switched := run_in_environment(arguments)) >= 0:
        return switched
    if missing:
        print(f'bindcraft: this environment has no {", ".join(missing)}; run bash install.sh, or activate the environment that carries them', file=sys.stderr)
        return 2
    if designs:
        print(f'bindcraft: {design_gpu_note()}', flush=True)
    return run_command_line(arguments)
if __name__ == '__main__':
    sys.exit(main())
