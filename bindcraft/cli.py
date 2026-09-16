import ast
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from bindcraft import OPERATOR_COMPILATION_CACHE, command_modules, package_modules
from bindcraft.model_weights import missing_model_weights, model_weights

COMMANDS = ('design', 'archive', 'unarchive', 'fetch-weights')
COMMAND_MODULES = {'filter': 'campaign_filter'}
CAMPAIGN_PRESETS = Path(__file__).parent.parent / 'settings'
USAGE = '\n'.join(('usage: bindcraft design <settings.json> [--core NAME] [--modality NAME[,NAME]] [--humanize ...] [--metadata <metadata.json>] [--set KEY=VALUE]...',
                   '       bindcraft design --list-targets | --list-modalities | --list-properties | --list-core | --list-settings',
                   '       bindcraft score <design.cif> [--binder CHAINS] [--target CHAINS] [--hotspots SPANS]',
                   '       bindcraft rank <campaign folder> [--on METRIC] [--list]',
                   '       bindcraft filter <campaign folder> [--where METRIC>=VALUE]... [--filters FILE] [--list]',
                   '       bindcraft campaign_output [<campaign folder> ...]',
                   '       bindcraft archive|unarchive <campaign folder>',
                   '       bindcraft fetch-weights'))

def module_command(command: str):
    module = COMMAND_MODULES.get(command, command)
    if module not in package_modules():
        return None
    return getattr(importlib.import_module(f'bindcraft.{module}'), 'main', None)

COMMAND_PURPOSE = {'archive': 'Pack every trajectory folder of a finished campaign into one archive each, which is what makes it quick to copy.',
                   'unarchive': 'Unpack the trajectory archives of a campaign, so the rounds inside them can be read again.',
                   'fetch-weights': 'Put both checkpoint sets on this machine now, for a compute node whose network reaches nothing.'}

def usage_command(line: str) -> str:
    words = line.split()
    return words[words.index('bindcraft') + 1] if 'bindcraft' in words[:-1] else ''

def command_help_text(command: str) -> str:
    named = [f"usage: {line.strip().removeprefix('usage: ')}" for line in USAGE.splitlines() if command in usage_command(line).split('|')]
    return '\n'.join((*named, '', COMMAND_PURPOSE[command])) if named and command in COMMAND_PURPOSE else usage_text()

def usage_text() -> str:
    spelled = {name for line in USAGE.splitlines() for name in usage_command(line).split('|')} | set(COMMAND_MODULES.values())
    return '\n'.join((USAGE, *(f'       bindcraft {name} [...]' for name in command_modules() if name not in spelled),
                      '', 'bindcraft design -h names every binder format and design property this package ships.'))

def split_setting_overrides(arguments: list[str]) -> tuple[list[str], list[str]]:
    paths, assignments, overridden = [], [], False
    for argument in arguments:
        if overridden:
            assignments.append(argument)
        elif argument != '--set':
            paths.append(argument)
        overridden = argument == '--set'
    return paths, assignments + [''] if overridden else assignments

def shipped_preset_names(tier: str) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in (CAMPAIGN_PRESETS / tier).glob('*.json')))

def design_property_flags() -> dict[str, str]:
    return {'--' + name.replace('_', '-'): name for name in shipped_preset_names('property')}

DESIGN_HELP = '\n'.join(('usage: bindcraft design <settings.json> [--modality NAME[,NAME]] [--<property>]...',
                         '                        [--metadata <metadata.json>] [--set KEY=VALUE]...',
                         '',
                         'Design binders against the target a settings file names, on the best practice asked for by name.',
                         '',
                         '  <settings.json>     the campaign: the target it designs against, and anything set by hand',
                         '  --core NAME         a core profile applied under every preset, such as benchmark for a reproducible run',
                         '  --modality NAME     the binder format to design, comma separated to combine (default: binder)',
                         '  --set KEY=VALUE     set one setting, over whatever the presets carry; repeat for more',
                         '  --metadata FILE     fields to record with the campaign, which are not settings',
                         '  --list-targets      print the shipped target names alone, one a line',
                         '  --list-modalities   print the modality names alone, one a line',
                         '  --list-properties   print the design property names alone, one a line',
                         '  --list-core         print the core profile names alone, one a line',
                         '  --list-settings     print every setting --set accepts, one a line',
                         '  -h, --help          this help'))

COMMON_SETTINGS = (('project_folder', 'where the campaign writes everything, Binders unless a settings file names one'),
                   ('binder_lengths', 'the binder size, [60,100] for a range to draw from or [80] for one length'),
                   ('number_of_final_designs', 'how many accepted designs to stop at'),
                   ('max_trajectories', 'how many design attempts to spend before giving up'),
                   ('resume', 'true to carry on into a project_folder already written'),
                   ('campaign_seed', 'the seed every trajectory is drawn from'))
SETTING_EXAMPLES = ("  bindcraft design examples/pdl1.json --set 'project_folder=results/my_run'",
                    "  bindcraft design examples/pdl1.json --set 'binder_lengths=[70,90]'")

def campaign_setting_names() -> tuple[str, ...]:
    from bindcraft import module_source
    for node in ast.parse(module_source('settings')).body:
        if 'CAMPAIGN_SETTING_NAMES' in [target.id for target in getattr(node, 'targets', ()) if isinstance(target, ast.Name)]:
            declared = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
            return tuple(sorted(ast.literal_eval(declared)))
    return ()

def common_setting_lines() -> tuple[str, ...]:
    width = max(len(name) for name, _ in COMMON_SETTINGS)
    return tuple(f'  {name:{width}}  {purpose}' for name, purpose in COMMON_SETTINGS)

def preset_descriptions(tier: str) -> dict[str, str]:
    described = {}
    for name in shipped_preset_names(tier):
        try:
            described[name] = json.loads((CAMPAIGN_PRESETS / tier / f'{name}.json').read_text()).get('description') or ''
        except (OSError, ValueError):
            described[name] = ''
    return described

def described_preset_lines(tier: str, flag: bool=False) -> tuple[str, ...]:
    described = preset_descriptions(tier)
    written = {name: '--' + name.replace('_', '-') if flag else name for name in described}
    width = max((len(spelling) for spelling in written.values()), default=0)
    return tuple(f'  {written[name]:{width}}  {description}'.rstrip() for name, description in described.items())

def design_help_text() -> str:
    return '\n'.join((DESIGN_HELP, '',
                      'targets this package ships, as "target" in the campaign file:', *described_preset_lines('target'), '',
                      'binder formats, as --modality NAME:', *described_preset_lines('modality'), '',
                      'design properties, added by their own flag:', *described_preset_lines('property', flag=True), '',
                      'the settings changed most often, as --set KEY=VALUE:', *common_setting_lines(), '',
                      *SETTING_EXAMPLES, '',
                      f'--list-settings names all {len(campaign_setting_names())} of them, and docs/reference.md says what each one does.', '',
                      'A preset only sets a default: --set and the settings file both win over it, and a combination',
                      'this package declares unsupported is refused before the campaign starts.'))

def split_campaign_presets(arguments: list[str]) -> tuple[list[str], list[str]]:
    property_flags = design_property_flags()
    named_flags = {'--modality': 'modality', '--core': 'core'}
    paths, named_values, assignments, pending = [], {'modality': [], 'core': []}, [], None
    for argument in arguments:
        if pending:
            named_values[pending] += [name for name in argument.split(',') if name]
            pending = None
        elif argument in named_flags:
            pending = named_flags[argument]
        elif argument in property_flags:
            assignments.append(f'{property_flags[argument]}=true')
        else:
            paths.append(argument)
    assignments += [f'{setting}={json.dumps(names)}' for setting, names in named_values.items() if names]
    return paths, assignments + [''] if pending else assignments

def split_metadata_file(arguments: list[str]) -> tuple[list[str], str | None]:
    paths, metadata_path, named = [], None, False
    for argument in arguments:
        if named:
            metadata_path, named = argument, False
        elif argument == '--metadata':
            named = True
        else:
            paths.append(argument)
    return paths, '' if named else metadata_path

def design_card_name() -> str | None:
    try:
        listing = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    names = [line.strip() for line in listing.splitlines() if line.strip()]
    return (names[int(visible)] if visible.isdigit() and int(visible) < len(names) else names[0]) if names else None

def campaign_compile_cache_root(settings_path: str, assignments: list[str]=()) -> str | None:
    named = [value for name, _, value in (assignment.partition('=') for assignment in assignments) if name.strip() == 'project_folder']
    try:
        project_folder = named[-1] if named else json.loads(open(settings_path).read()).get('project_folder', 'bindcraft_campaign')
        return os.path.join(os.path.abspath(project_folder), 'compile_cache')
    except (OSError, ValueError, AttributeError):
        return None

def operator_compile_cache_root() -> str | None:
    home_cache = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    root = os.path.join(home_cache, 'bindcraft', 'compile_cache')
    try:
        os.makedirs(root, exist_ok=True)
        return root if os.access(root, os.W_OK) else None
    except OSError:
        return None

def use_campaign_compile_cache(settings_path: str, assignments: list[str]=()) -> str | None:
    if OPERATOR_COMPILATION_CACHE:
        return OPERATOR_COMPILATION_CACHE
    card = design_card_name()
    if card is None:
        return None
    for root in (campaign_compile_cache_root(settings_path, assignments), operator_compile_cache_root()):
        if root is None:
            continue
        compile_cache = os.path.join(root, re.sub(r'[^A-Za-z0-9]+', '_', card).strip('_'))
        try:
            os.makedirs(compile_cache, exist_ok=True)
        except OSError:
            continue
        if not os.access(compile_cache, os.W_OK):
            continue
        os.environ['JAX_COMPILATION_CACHE_DIR'] = compile_cache
        return compile_cache
    return None

def design(settings_path: str, assignments: list[str]=(), metadata_path: str | None=None) -> None:
    compile_cache = use_campaign_compile_cache(settings_path, assignments)
    from bindcraft.campaign import launch_campaign
    from bindcraft.design_workers import running_as_design_worker
    if compile_cache and not running_as_design_worker():
        print(f'compiled graphs cached in {compile_cache}', flush=True)
    from bindcraft.preflight import CampaignPreflightError
    try:
        status = launch_campaign(settings_path, assignments, metadata_path=metadata_path)
    except (CampaignPreflightError, ValueError, OSError) as refusal:
        print(f'campaign refused:\n{refusal}', file=sys.stderr)
        raise SystemExit(2)
    if status:
        raise SystemExit(status)

def archive_trajectories(command: str, project_folder: str) -> None:
    from bindcraft.campaign_output import archive_campaign_trajectories, restore_campaign_trajectories
    packed = (archive_campaign_trajectories if command == 'archive' else restore_campaign_trajectories)(project_folder)
    print(f'{command}d {len(packed)} trajectory folder(s) under {project_folder}', flush=True)

def fetch_weights() -> None:
    try:
        parameters, redesign_weights = model_weights()
    except (ValueError, OSError) as failure:
        print(failure, file=sys.stderr)
        raise SystemExit(2)
    missing = missing_model_weights(parameters, redesign_weights)
    if missing:
        print('checkpoints missing or unfinished:\n' + '\n'.join(f'  {problem}' for problem in missing), file=sys.stderr)
        raise SystemExit(2)
    print(f'AlphaFold parameters {parameters}\nProteinMPNN weights {redesign_weights}')

def main(arguments: list[str] | None=None) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for flag, tier in (('--list-targets', 'target'), ('--list-modalities', 'modality'), ('--list-properties', 'property')):
        if flag in arguments:
            print('\n'.join(shipped_preset_names(tier)))
            return
    if '--list-core' in arguments:
        print('\n'.join(name for name in shipped_preset_names('core') if name not in ('default', 'reference')))
        return
    if '--list-settings' in arguments:
        print('\n'.join(campaign_setting_names()))
        return
    command, rest = (arguments[0], arguments[1:]) if arguments else ('', [])
    if command and command not in COMMANDS and command not in COMMAND_MODULES and command not in package_modules() and (not command.startswith('-')):
        command, rest = 'design', arguments
    if command in COMMANDS and {'-h', '--help'} & set(rest):
        print(design_help_text() if command == 'design' else command_help_text(command))
        return
    if command == 'design':
        settings_paths, assignments = split_setting_overrides(rest)
        settings_paths, preset_assignments = split_campaign_presets(settings_paths)
        settings_paths, metadata_path = split_metadata_file(settings_paths)
        if len(settings_paths) == 1 and metadata_path != '':
            return design(settings_paths[0], preset_assignments + assignments, metadata_path)
    if command in ('archive', 'unarchive') and len(rest) == 1:
        return archive_trajectories(command, rest[0])
    if command == 'fetch-weights' and (not rest):
        return fetch_weights()
    run_module = module_command(command) if command else None
    if run_module:
        return run_module(rest)
    asked_for_help = command in ('-h', '--help')
    print(usage_text(), file=sys.stdout if asked_for_help else sys.stderr)
    raise SystemExit(0 if asked_for_help else 2)
if __name__ == '__main__':
    main()
