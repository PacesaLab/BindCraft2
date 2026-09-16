import csv
import os
import shutil
import jax
import jax.numpy as jnp
from bindcraft.campaign_output import recorded_row, structure_metadata, trajectory_output_path
from bindcraft.filters import residue_confidence_tracks
from bindcraft.protein import AMINO_ACIDS, ATOM_INDEX, BINDER_ALONE, THREE_LETTER_CODE, Protein, StructurePredictions, is_binder_chain, superposed_on_binder, write_structure
import base64
import gzip
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, NamedTuple

ANIMATED_BACKBONE_ATOM_NAMES = ('N', 'CA', 'C', 'O')
PLDDT_BANDS = ((50.0, '#FF7D45', 'very low'), (70.0, '#FFDB13', 'low'), (90.0, '#65CBF3', 'confident'), (100.0, '#0053D6', 'very high'))
PAE_COLOUR_LIMIT = 31.0
PAE_PANEL_SIZE = 128
COORDINATE_SCALE = 100.0
INDUCED_FIT_SWITCH_STEPS = 10
STRUCTURE_VIEWER_URL = 'https://3Dmol.org/build/3Dmol-min.js'

class TrajectoryFrame(NamedTuple):
    backbone_atoms: np.ndarray
    residue_confidence: np.ndarray
    aatype: np.ndarray
    residue_pair_error: np.ndarray | None
    sequence_probabilities: np.ndarray | None
    design_stage: str
    sequence_update: int
    chain_lengths: list[int]
    chain_names: list[str]

def design_stage_boundaries(metric_rows: list[dict]) -> list[tuple[int, str]]:
    boundaries, previous_stage = ([], None)
    for position, row in enumerate(metric_rows):
        if row.get('phase') != previous_stage:
            boundaries.append((position, str(row.get('phase'))))
            previous_stage = row.get('phase')
    return boundaries

def state_metric_series(metric_rows: list[dict], state_names: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    metric_series: dict[str, dict[str, list[float]]] = {}
    for column in sorted({key for row in metric_rows for key in row} - {'phase', 'round'}):
        state_name = next((name for name in state_names if column.startswith(f'{name}.')), '')
        metric_name = column[len(state_name) + 1:] if state_name else column
        if state_name and metric_name.endswith(f'.{state_name}'):
            metric_name = metric_name[:-len(state_name) - 1]
        metric_series.setdefault(metric_name, {})[state_name] = [row.get(column, np.nan) for row in metric_rows]
    return metric_series

def state_active_rounds(metric_rows: list[dict], state_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {state_name: np.array([any(key.startswith(f'{state_name}.') for key in row) for row in metric_rows]) for state_name in state_names}

def write_loss_plot(metric_rows: list[dict], path: str, state_names: tuple[str, ...]=(), dpi: int=120) -> None:
    metric_series = state_metric_series(metric_rows, state_names)
    if not metric_rows or not metric_series:
        return
    columns = min(3, len(metric_series))
    rows = -(-len(metric_series) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 2.6 * rows), sharex=True, squeeze=False)
    stage_boundaries = design_stage_boundaries(metric_rows)
    active_rounds = state_active_rounds(metric_rows, state_names)
    every_round = np.ones(len(metric_rows), dtype=bool)
    for panel, (axis, (metric_name, state_values)) in enumerate(zip(axes.flat, metric_series.items())):
        stage_positions = set()
        for state_name, values in state_values.items():
            active = active_rounds.get(state_name, every_round)
            sequence_updates = np.cumsum(active) - 1
            measured = np.isfinite(np.asarray(values, dtype=float)) & active
            axis.plot(sequence_updates[measured], np.asarray(values, dtype=float)[measured], linewidth=1.2, label=state_name)
            stage_positions.update(int(active[:position].sum()) for position, stage in stage_boundaries[1:])
        axis.set_title(metric_name, fontsize=9)
        axis.tick_params(labelsize=7)
        if not panel and len(state_values) > 1:
            axis.legend(fontsize=6, loc='best')
        for position in sorted(stage_positions):
            axis.axvline(position, color='0.7', linewidth=0.8, linestyle='--')
    for axis in axes.flat[len(metric_series):]:
        axis.axis('off')
    for axis in axes[-1]:
        axis.set_xlabel('sequence update', fontsize=8)
    figure.suptitle('  '.join(f'{stage}@{position}' for position, stage in stage_boundaries), fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)

def superpose_on_reference(coordinates: np.ndarray, reference: np.ndarray, fitted_rows: np.ndarray | None=None, reference_rows: np.ndarray | None=None) -> np.ndarray:
    fitted = coordinates if fitted_rows is None else coordinates[fitted_rows]
    fitted_reference = reference if reference_rows is None else reference[reference_rows]
    coordinate_centre, reference_centre = (fitted.mean(0), fitted_reference.mean(0))
    try:
        left, _, right = np.linalg.svd((fitted - coordinate_centre).T @ (fitted_reference - reference_centre))
    except np.linalg.LinAlgError:
        return coordinates
    rotation = right.T * np.array([1.0, 1.0, np.sign(np.linalg.det(right.T @ left.T))]) @ left.T
    return (coordinates - coordinate_centre) @ rotation.T + reference_centre

def chain_spans(chain_lengths: list[int]) -> list[tuple[int, int]]:
    chain_starts = np.cumsum([0] + list(chain_lengths))
    return list(zip(chain_starts[:-1], chain_starts[1:]))

def binder_atom_rows(frame: TrajectoryFrame) -> np.ndarray:
    spans = dict(zip(frame.chain_names, chain_spans(frame.chain_lengths)))
    binder_residues = [np.arange(*spans[name]) for name in frame.chain_names if is_binder_chain(name)]
    atoms_per_residue = frame.backbone_atoms.shape[1]
    return (np.concatenate(binder_residues)[:, None] * atoms_per_residue + np.arange(atoms_per_residue)).reshape(-1) if binder_residues else np.arange(frame.backbone_atoms.size // 3)

def sequence_display_weights(logits, one_hot_weight, temperature, logit_scale):
    probabilities = jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float32) * logit_scale / temperature, axis=-1)
    return one_hot_weight * jax.nn.one_hot(jnp.argmax(probabilities, axis=-1), probabilities.shape[-1]) + (1.0 - one_hot_weight) * probabilities

def frame_differences(values: np.ndarray, dtype) -> bytes:
    rounded = np.round(np.asarray(values, dtype=np.float64)).astype(np.int32)
    return np.concatenate([rounded[:1], np.diff(rounded, axis=0)]).astype(dtype).tobytes()

def packed_frames(superposed: list[np.ndarray], frames: list['TrajectoryFrame'], panels: list[np.ndarray], origin: np.ndarray) -> str:
    residue_confidence = (np.clip(np.stack([frame.residue_confidence for frame in frames]), 0.0, 1.0) * 255).astype(np.uint8)
    aatype = np.stack([frame.aatype for frame in frames]).astype(np.uint8)
    designed = [frame.sequence_probabilities for frame in frames if frame.sequence_probabilities is not None]
    coordinates = frame_differences((np.stack(superposed) - origin) * COORDINATE_SCALE, np.int16)
    probabilities = frame_differences(np.stack(designed) * 255, np.uint8) if len(designed) == len(frames) else b''
    packed = coordinates + (frame_differences(np.stack(panels), np.uint8) if panels else b'') + residue_confidence.tobytes() + aatype.tobytes() + probabilities
    return base64.b64encode(gzip.compress(packed, 9)).decode()

def pae_panel(residue_pair_error, panel_size: int=PAE_PANEL_SIZE) -> np.ndarray | None:
    if residue_pair_error is None:
        return None
    matrix = np.asarray(residue_pair_error, dtype=np.float32)
    if matrix.shape[0] > panel_size:
        edges = np.linspace(0, matrix.shape[0], panel_size + 1).round().astype(int)
        pooled = np.add.reduceat(np.add.reduceat(matrix, edges[:-1], axis=0), edges[:-1], axis=1)
        matrix = pooled / np.outer(np.diff(edges), np.diff(edges))
    return (np.clip(matrix / PAE_COLOUR_LIMIT, 0.0, 1.0) * 255).astype(np.uint8)

def design_stage_spans(stage_frames: list[str]) -> list[dict]:
    spans: list[dict] = []
    for position, stage in enumerate(stage_frames):
        if not spans or spans[-1]['name'] != stage:
            spans.append({'name': stage, 'start': position, 'end': position + 1})
        else:
            spans[-1]['end'] = position + 1
    return spans

def induced_fit_switch_frames(bound_frame: TrajectoryFrame, unbound_frame: TrajectoryFrame, steps: int=INDUCED_FIT_SWITCH_STEPS) -> list[TrajectoryFrame]:
    bound_spans = dict(zip(bound_frame.chain_names, chain_spans(bound_frame.chain_lengths)))
    unbound_spans = {name: span for name, span in zip(unbound_frame.chain_names, chain_spans(unbound_frame.chain_lengths)) if name in bound_spans and span[1] - span[0] == bound_spans[name][1] - bound_spans[name][0]}
    if not unbound_spans:
        return []
    bound_residues = np.concatenate([np.arange(*bound_spans[name]) for name in unbound_spans])
    unbound_residues = np.concatenate([np.arange(*span) for span in unbound_spans.values()])
    switched_atoms, switched_confidence = (np.array(bound_frame.backbone_atoms), np.array(bound_frame.residue_confidence))
    switched_atoms[bound_residues] = superpose_on_reference(np.asarray(unbound_frame.backbone_atoms[unbound_residues], dtype=np.float64).reshape(-1, 3), np.asarray(bound_frame.backbone_atoms[bound_residues], dtype=np.float64).reshape(-1, 3)).reshape(len(bound_residues), -1, 3)
    switched_confidence[bound_residues] = unbound_frame.residue_confidence[unbound_residues]
    return [bound_frame._replace(backbone_atoms=bound_frame.backbone_atoms * (1.0 - weight) + switched_atoms * weight, residue_confidence=bound_frame.residue_confidence * (1.0 - weight) + switched_confidence * weight, design_stage='switch') for weight in np.concatenate([np.linspace(0.0, 1.0, steps), np.linspace(1.0, 0.0, steps)[1:]])]

def animation_state(name: str, frames: list[TrajectoryFrame], binder_reference: TrajectoryFrame | None=None, origin: np.ndarray | None=None) -> dict | None:
    drawable = [frame for frame in frames if np.isfinite(frame.backbone_atoms).all()]
    if len(drawable) < 2:
        return None
    chain_lengths = drawable[-1].chain_lengths
    residue_count = sum(chain_lengths)
    reference = np.asarray(drawable[-1].backbone_atoms, dtype=np.float64).reshape(-1, 3)
    if binder_reference is not None:
        reference = superpose_on_reference(reference, np.asarray(binder_reference.backbone_atoms, dtype=np.float64).reshape(-1, 3), binder_atom_rows(drawable[-1]), binder_atom_rows(binder_reference))
    superposed = [superpose_on_reference(np.asarray(frame.backbone_atoms, dtype=np.float64).reshape(-1, 3), reference).reshape(residue_count, len(ANIMATED_BACKBONE_ATOM_NAMES), 3) for frame in drawable]
    panels = [frame.residue_pair_error for frame in drawable if frame.residue_pair_error is not None]
    panels = panels if len(panels) == len(drawable) and len({panel.shape for panel in panels}) == 1 else []
    return {'name': name,
            'frames': packed_frames(superposed, drawable, panels, np.zeros(3) if origin is None else origin),
            'panelSize': int(panels[0].shape[0]) if panels else 0,
            'chainLengths': [int(length) for length in chain_lengths],
            'residueCount': residue_count,
            'designedResidueCount': int(drawable[0].sequence_probabilities.shape[0]) if drawable[0].sequence_probabilities is not None else 0,
            'stages': design_stage_spans([frame.design_stage for frame in drawable]),
            'rounds': [frame.sequence_update for frame in drawable],
            'frameCount': len(drawable)}

def write_trajectory_animation(state_frames: dict[str, list[TrajectoryFrame]], path: str, interval: int=120) -> None:
    drawn_frames = [[frame for frame in frames if np.isfinite(frame.backbone_atoms).all()] for frames in state_frames.values()]
    binder_reference = next((frames[-1] for frames in drawn_frames if len(frames) > 1), None)
    origin = np.asarray(binder_reference.backbone_atoms, dtype=np.float64).reshape(-1, 3).mean(0) if binder_reference is not None else np.zeros(3)
    states = [state for state in (animation_state(name, frames, binder_reference, origin) for name, frames in state_frames.items()) if state]
    if not states:
        return
    with open(path, 'w') as animation_file:
        animation_file.write(TRAJECTORY_VIEWER_TEMPLATE.format(library=STRUCTURE_VIEWER_URL, title=os.path.basename(os.path.dirname(os.path.abspath(path))), states=json.dumps(states), interval=interval, pae_limit=int(PAE_COLOUR_LIMIT), coordinate_scale=int(COORDINATE_SCALE), atom_names=json.dumps(list(ANIMATED_BACKBONE_ATOM_NAMES)), residue_names=json.dumps([THREE_LETTER_CODE[amino_acid] for amino_acid in AMINO_ACIDS]), amino_acids=json.dumps(list(AMINO_ACIDS)), plddt_bands=json.dumps([[limit, colour] for limit, colour, _ in PLDDT_BANDS]), plddt_key=' '.join(f'<b style="color:{colour}">&#9632;</b>{band_name}' for _, colour, band_name in PLDDT_BANDS)))

TRAJECTORY_VIEWER_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="{library}"></script>
<style>
body {{ font-family: sans-serif; margin: 0; padding: 8px; color: #222; background: #fff; }}
#layout {{ display: flex; align-items: flex-start; gap: 12px; }}
#viewer {{ position: relative; width: 700px; height: 560px; border: 1px solid #ddd; }}
#side {{ width: 280px; }}
#pae {{ display: block; width: 280px; height: 280px; border: 1px solid #ddd; }}
#sequence {{ display: block; width: 280px; height: 212px; border: 1px solid #ddd; margin-top: 10px; }}
#graphs {{ padding-top: 10px; }}
#confidence {{ display: block; border: 1px solid #ddd; }}
#controls {{ display: flex; align-items: center; gap: 14px; padding-top: 8px; }}
#track {{ width: 520px; }}
#slider {{ width: 100%; margin: 0; }}
#stages {{ display: flex; font-size: 11px; color: #777; }}
#stages div {{ border-left: 1px solid #bbb; padding-left: 3px; overflow: hidden; white-space: nowrap; }}
#stages div.active {{ color: #111; font-weight: 600; }}
#states {{ display: flex; align-items: center; gap: 6px; font-size: 15px; }}
#states span {{ color: #555; }}
#states button {{ font-size: 15px; font-family: inherit; padding: 4px 12px; border: 1px solid #bbb; border-radius: 4px; background: #f4f4f4; color: #333; cursor: pointer; }}
#states button.active {{ background: #1c6fd0; border-color: #1c6fd0; color: #fff; font-weight: 600; }}
#play {{ font-size: 14px; font-family: inherit; padding: 4px 14px; }}
#label {{ font-size: 14px; }}
.caption {{ font-size: 12px; padding-top: 4px; color: #555; }}
</style></head>
<body>
<div id="layout">
  <div>
    <div id="viewer"></div>
    <div class="caption">pLDDT {plddt_key}, drag to rotate</div>
  </div>
  <div id="side">
    <div id="paePanel">
      <canvas id="pae"></canvas>
      <div class="caption">pAE, 0 to {pae_limit} A</div>
    </div>
    <canvas id="sequence"></canvas>
    <div class="caption">designed sequence this update, one column a residue: white to blue on the root of the weight each amino acid holds</div>
  </div>
</div>
<div id="controls">
  <button id="play">pause</button>
  <div id="track">
    <input id="slider" type="range" min="0" value="0">
    <div id="stages"></div>
  </div>
  <span id="label"></span>
  <div id="states"></div>
</div>
<div id="graphs">
  <canvas id="confidence"></canvas>
  <div class="caption">pLDDT per residue, one column a sequence update</div>
</div>
<script>
const predictionStates = {states};
const atomNames = {atom_names};
const residueNames = {residue_names};
const aminoAcids = {amino_acids};
const plddtBands = {plddt_bands}.map(([limit, colour]) => [limit, parseInt(colour.slice(1), 16)]);
const SEQUENCE_GUTTER = 14;
const CONFIDENCE_TRACK_BOX = [992, 480];
const sequenceCells = document.createElement('canvas');
const slider = document.getElementById('slider');
const label = document.getElementById('label');
const canvas = document.getElementById('pae');
const stageRow = document.getElementById('stages');
let active = predictionStates[0];
let frames = null;
let confidenceTrack = null;
let stageCells = [];
let currentFrame = 0;
let playing = false;
function accumulated(differences, frameCount, values) {{
  const stride = differences.length / frameCount;
  values.set(differences.subarray(0, stride));
  for (let frame = 1; frame < frameCount; frame++)
    for (let index = 0; index < stride; index++)
      values[frame * stride + index] = values[(frame - 1) * stride + index] + differences[frame * stride + index];
  return values;
}}
function inflated(state) {{
  const bytes = Uint8Array.from(atob(state.frames), (character) => character.charCodeAt(0));
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer().then((block) => {{
    const coordinateCount = state.frameCount * state.residueCount * atomNames.length * 3;
    const panelCount = state.frameCount * state.panelSize * state.panelSize;
    const residueCount = state.frameCount * state.residueCount;
    const probabilityCount = state.frameCount * state.designedResidueCount * aminoAcids.length;
    const tracksAt = 2 * coordinateCount + panelCount;
    return {{ coordinates: accumulated(new Int16Array(block, 0, coordinateCount), state.frameCount, new Int32Array(coordinateCount)),
              panels: accumulated(new Uint8Array(block, 2 * coordinateCount, panelCount), state.frameCount, new Uint8Array(panelCount)),
              confidence: new Uint8Array(block, tracksAt, residueCount),
              sequence: new Uint8Array(block, tracksAt + residueCount, residueCount),
              probabilities: accumulated(new Uint8Array(block, tracksAt + 2 * residueCount, probabilityCount), state.frameCount, new Uint8Array(probabilityCount)) }};
  }});
}}
function chainBoundaries(state, size) {{
  const boundaries = [];
  let residue = 0;
  for (let chain = 0; chain < state.chainLengths.length - 1; chain++) {{
    residue += state.chainLengths[chain];
    boundaries.push(Math.round(residue * size / state.residueCount));
  }}
  return boundaries;
}}
function trajectoryPdb(state) {{
  const models = [];
  for (let frame = 0; frame < state.frameCount; frame++) {{
    const lines = [];
    let residue = 0;
    state.chainLengths.forEach((length, chain) => {{
      for (let position = 1; position <= length; position++, residue++) {{
        const residueName = residueNames[frames.sequence[frame * state.residueCount + residue]];
        const bFactor = (frames.confidence[frame * state.residueCount + residue] / 2.55).toFixed(2).padStart(6);
        for (let atom = 0; atom < atomNames.length; atom++) {{
          const at = ((frame * state.residueCount + residue) * atomNames.length + atom) * 3;
          const coordinates = [0, 1, 2].map((axis) => (frames.coordinates[at + axis] / {coordinate_scale}).toFixed(3).padStart(8)).join('');
          lines.push('ATOM  ' + String(lines.length + 1).padStart(5) + '  ' + atomNames[atom].padEnd(3) + ' ' + residueName.padStart(3) + ' ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[chain % 26] + String(position).padStart(4) + '    ' + coordinates + '  1.00' + bFactor);
        }}
      }}
      lines.push('TER');
    }});
    models.push('MODEL ' + String(frame + 1).padStart(8) + '\\n' + lines.join('\\n') + '\\nENDMDL');
  }}
  return models.join('\\n');
}}
function confidenceColour(plddt) {{
  return plddtBands.find(([limit]) => plddt < limit || limit >= 100)[1];
}}
function paintConfidenceTrack() {{
  const track = document.getElementById('confidence');
  track.width = active.frameCount;
  track.height = active.residueCount;
  //one square a residue a sequence update: the same scale on both axes, so a short trajectory stays narrow rather than stretching each update across the page
  const cell = Math.min(CONFIDENCE_TRACK_BOX[0] / active.frameCount, CONFIDENCE_TRACK_BOX[1] / active.residueCount, 6);
  track.style.width = Math.max(1, Math.round(cell * active.frameCount)) + 'px';
  track.style.height = Math.max(1, Math.round(cell * active.residueCount)) + 'px';
  const image = track.getContext('2d').createImageData(active.frameCount, active.residueCount);
  for (let frame = 0; frame < active.frameCount; frame++)
    for (let residue = 0; residue < active.residueCount; residue++) {{
      const colour = confidenceColour(frames.confidence[frame * active.residueCount + residue] / 2.55);
      const pixel = 4 * (residue * active.frameCount + frame);
      image.data[pixel] = colour >> 16; image.data[pixel + 1] = (colour >> 8) & 255; image.data[pixel + 2] = colour & 255; image.data[pixel + 3] = 255;
    }}
  for (const boundary of chainBoundaries(active, active.residueCount))
    for (let frame = 0; frame < active.frameCount; frame++) {{
      const pixel = 4 * (boundary * active.frameCount + frame);
      image.data[pixel] = 0; image.data[pixel + 1] = 0; image.data[pixel + 2] = 0;
    }}
  return image;
}}
function drawConfidenceTrack(image, frame) {{
  const context = document.getElementById('confidence').getContext('2d');
  context.putImageData(image, 0, 0);
  context.fillStyle = 'rgba(0, 0, 0, 0.8)';
  context.fillRect(frame, 0, 1, active.residueCount);
}}
function drawSequenceProbabilities(frame) {{
  //one column a designed residue and one row an amino acid, so a settled position is a single square and an undecided one a smear down its column
  const designed = active.designedResidueCount;
  if (!designed) return;
  const track = document.getElementById('sequence');
  const context = track.getContext('2d');
  const image = context.createImageData(designed, aminoAcids.length);
  for (let residue = 0; residue < designed; residue++)
    for (let acid = 0; acid < aminoAcids.length; acid++) {{
      //shaded on the root of the weight: a logit stage spreads a twentieth over every amino acid, which reads as blank against a straight scale
      const weight = Math.sqrt(frames.probabilities[(frame * designed + residue) * aminoAcids.length + acid] / 255);
      const pixel = 4 * (acid * designed + residue);
      image.data[pixel] = Math.round(255 - 255 * weight); image.data[pixel + 1] = Math.round(255 - 172 * weight); image.data[pixel + 2] = 255; image.data[pixel + 3] = 255;
    }}
  sequenceCells.width = designed;
  sequenceCells.height = aminoAcids.length;
  sequenceCells.getContext('2d').putImageData(image, 0, 0);
  context.clearRect(0, 0, track.width, track.height);
  context.imageSmoothingEnabled = false;
  context.drawImage(sequenceCells, SEQUENCE_GUTTER, 0, track.width - SEQUENCE_GUTTER, track.height);
  const rowHeight = track.height / aminoAcids.length;
  context.fillStyle = '#555';
  context.font = Math.min(9, Math.floor(rowHeight)) + 'px sans-serif';
  context.textBaseline = 'middle';
  aminoAcids.forEach((acid, row) => context.fillText(acid, 2, (row + 0.5) * rowHeight));
}}
function paintPairwiseError(index) {{
  if (!active.panelSize) return;
  const size = active.panelSize;
  const context = canvas.getContext('2d');
  const image = context.createImageData(size, size);
  for (let pixel = 0; pixel < size * size; pixel++) {{
    const level = frames.panels[index * size * size + pixel] / 255;
    const shade = Math.round(510 * (level < 0.5 ? level : 1 - level));
    image.data[4 * pixel] = level < 0.5 ? shade : 255;
    image.data[4 * pixel + 1] = shade;
    image.data[4 * pixel + 2] = level < 0.5 ? 255 : shade;
    image.data[4 * pixel + 3] = 255;
  }}
  for (const boundary of chainBoundaries(active, size)) {{
    for (let step = 0; step < size; step++) {{
      for (const pixel of [boundary * size + step, step * size + boundary]) {{
        image.data[4 * pixel] = 0; image.data[4 * pixel + 1] = 0; image.data[4 * pixel + 2] = 0;
      }}
    }}
  }}
  context.putImageData(image, 0, 0);
}}
if (typeof $3Dmol === 'undefined' || typeof DecompressionStream === 'undefined') {{
  document.getElementById('viewer').textContent = typeof DecompressionStream === 'undefined' ? 'this page holds its frames compressed and needs a browser that can inflate them: Chrome 80, Firefox 113, Safari 16.4 or newer' : 'the structure viewer could not be loaded from {library} -- this page needs network access the first time it is opened';
}} else {{
  const viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{ backgroundColor: 'white' }});
  const showFrame = (index) => {{
    currentFrame = index;
    slider.value = index;
    const position = active.stages.findIndex((span) => index >= span.start && index < span.end);
    stageCells.forEach((cell, cellPosition) => cell.classList.toggle('active', cellPosition === position));
    const span = active.stages[position];
    label.textContent = (span ? span.name + ' ' + (index - span.start + 1) + ' / ' + (span.end - span.start) + '  \u00b7  ' : '') + 'update ' + (index + 1) + ' / ' + active.frameCount + '  \u00b7  round ' + active.rounds[index];
    paintPairwiseError(index);
    drawConfidenceTrack(confidenceTrack, index);
    drawSequenceProbabilities(index);
    Promise.resolve(viewer.setFrame(index)).then(() => viewer.render());
  }};
  const stateButtons = predictionStates.map((state, index) => {{
    const button = document.createElement('button');
    button.textContent = state.name;
    button.title = 'show the trajectory predicted on ' + state.name;
    button.addEventListener('click', () => loadState(index));
    return button;
  }});
  if (predictionStates.length > 1) {{
    const caption = document.createElement('span');
    caption.textContent = 'state';
    document.getElementById('states').appendChild(caption);
    stateButtons.forEach((button) => document.getElementById('states').appendChild(button));
  }}
  async function loadState(stateIndex) {{
    active = predictionStates[stateIndex];
    frames = await inflated(active);
    stateButtons.forEach((button, position) => button.classList.toggle('active', position === stateIndex));
    stageRow.textContent = '';
    stageCells = active.stages.map((span) => {{
      const cell = document.createElement('div');
      cell.style.flexGrow = String(span.end - span.start);
      cell.textContent = span.name;
      cell.title = span.name + ' rounds ' + (span.start + 1) + ' to ' + span.end;
      stageRow.appendChild(cell);
      return cell;
    }});
    slider.max = active.frameCount - 1;
    document.getElementById('paePanel').style.display = active.panelSize ? '' : 'none';
    canvas.width = active.panelSize;
    canvas.height = active.panelSize;
    confidenceTrack = paintConfidenceTrack();
    const sequenceCanvas = document.getElementById('sequence');
    sequenceCanvas.width = sequenceCanvas.clientWidth;
    sequenceCanvas.height = sequenceCanvas.clientHeight;
    viewer.clear();
    viewer.addModelsAsFrames(trajectoryPdb(active), 'pdb');
    //every frame carries its own pLDDT, so the bands are painted on the atoms rather than asked of a built-in gradient that cannot hold them
    const model = viewer.getModel();
    for (const atoms of model.frames || []) for (const atom of atoms) atom.color = confidenceColour(atom.b);
    for (const atom of model.selectedAtoms({{}})) atom.color = confidenceColour(atom.b);
    viewer.setStyle({{}}, {{ cartoon: {{ thickness: 0.5, arrows: true }} }});
    Promise.resolve(viewer.setFrame(active.frameCount - 1)).then(() => {{ viewer.zoomTo(); viewer.zoom(0.85); showFrame(0); }});
  }}
  slider.addEventListener('input', () => {{ playing = false; document.getElementById('play').textContent = 'play'; showFrame(Number(slider.value)); }});
  document.getElementById('play').addEventListener('click', (event) => {{ playing = !playing; event.target.textContent = playing ? 'pause' : 'play'; }});
  //inflating a state and parsing its frames blocks, so the clock only starts once the first one is drawable; started earlier its ticks queue up behind that work and the page opens on a jump
  label.textContent = 'reading the trajectory';
  document.getElementById('play').textContent = 'wait';
  loadState(0).then(() => {{
    playing = true;
    document.getElementById('play').textContent = 'pause';
    setInterval(() => {{ if (playing) showFrame((currentFrame + 1) % active.frameCount); }}, {interval});
  }});
}}
</script>
</body></html>
"""

def copy_trajectory_animation(trajectory_directory: str, animation_path: str) -> str | None:
    recorded_animation = trajectory_output_path(trajectory_directory, 'trajectory.html')
    if not os.path.exists(recorded_animation):
        return None
    shutil.copyfile(recorded_animation, animation_path)
    return animation_path

class TrajectoryRecorder:
    def __init__(self, trajectory_directory: str, keep_frames: bool=True, receptor_chains: dict[str, tuple[tuple[str, int, int], ...]] | None=None, keep_sequences: bool=False):
        if os.path.isdir(trajectory_directory) and os.listdir(trajectory_directory):
            raise FileExistsError(f'{trajectory_directory!r} already has trajectory output -- remove it first to re-run')
        os.makedirs(trajectory_directory, exist_ok=True)
        self.trajectory_directory = trajectory_directory
        self.keep_frames = keep_frames
        self.keep_sequences = keep_sequences
        self.receptor_chains = receptor_chains
        self.design_stage = 'phase'
        self.sequence_parameters: Callable[[], tuple] | None = None
        self.metric_rows: list[dict] = []
        self.chain_sequence_history: dict[str, list] = {}
        self.backbone_history: dict[str, list] = {}
        self.bound_complex: dict[str, Protein] | None = None

    def __call__(self, sequence_updates: int, predictions: StructurePredictions) -> None:
        #induced fit case: rounds alternate bound/unbound
        self.bound_complex = next((prediction.protein_complex for name, prediction in predictions.items() if name != BINDER_ALONE), self.bound_complex)
        for name, prediction in predictions.items():
            #induced fit case
            written_complex = superposed_on_binder(prediction.protein_complex, self.bound_complex) if name == BINDER_ALONE and self.bound_complex else prediction.protein_complex
            if self.keep_frames:
                write_structure(written_complex, trajectory_output_path(self.trajectory_directory, f'{self.design_stage}_{sequence_updates:04d}_{name}.cif'), plddt=prediction.metrics.get('plddt'), metadata=structure_metadata('', prediction.metrics, state=name, stage=self.design_stage, round=str(sequence_updates)), residue_metrics=residue_confidence_tracks(predictions, name), receptor_chains=self.receptor_chains)
            chain_names = sorted(prediction.protein_complex)
            designed_chains = [chain for chain in chain_names if is_binder_chain(chain)]
            _, one_hot_weight, temperature, logit_scale = self.sequence_parameters() if self.sequence_parameters else (1.0, 0.0, 1.0, 1.0)
            designed_probabilities = jax.device_get([sequence_display_weights(prediction.protein_complex[chain].sequence, one_hot_weight, temperature, logit_scale) for chain in designed_chains]) if self.keep_sequences or self.keep_frames else []
            if self.keep_sequences:
                for chain, probabilities in zip(designed_chains, designed_probabilities):
                    self.chain_sequence_history.setdefault(f'{name}.{chain}', []).append(probabilities)
            if not self.keep_frames:
                continue
            backbone_atoms = np.concatenate([np.asarray(prediction.protein_complex[chain].atoms[:, [ATOM_INDEX[atom_name] for atom_name in ANIMATED_BACKBONE_ATOM_NAMES]], dtype=np.float32) for chain in chain_names])
            aatype = np.concatenate([np.asarray(prediction.protein_complex[chain].sequence.argmax(-1), dtype=np.int16) for chain in chain_names])
            residue_confidence = np.asarray(prediction.metrics['plddt'], dtype=np.float32) if 'plddt' in prediction.metrics else np.full(len(backbone_atoms), 0.75, dtype=np.float32)
            sequence_probabilities = np.concatenate(designed_probabilities) if designed_probabilities else None
            self.backbone_history.setdefault(name, []).append(TrajectoryFrame(backbone_atoms, residue_confidence, aatype, pae_panel(prediction.metrics.get('pae')), sequence_probabilities, self.design_stage, sequence_updates, [len(prediction.protein_complex[chain]) for chain in chain_names], chain_names))
        recorded = jax.device_get({f'{name}.{key}': value for name, prediction in predictions.items() for key, value in prediction.metrics.items() if getattr(value, 'ndim', 0) == 0})
        self.metric_rows.append({'phase': self.design_stage, 'round': sequence_updates, **{name: float(value) for name, value in recorded.items()}})

    def write_csv(self, path: str) -> None:
        fieldnames = sorted({key for row in self.metric_rows for key in row} | {'phase', 'round'})
        with open(path, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([recorded_row(row) for row in self.metric_rows])

    def write_sequences(self, path: str) -> None:
        np.savez_compressed(path, **{key: np.stack(frames) for key, frames in self.chain_sequence_history.items()})

    def write_loss_plot(self, path: str) -> None:
        write_loss_plot(self.metric_rows, path, tuple(self.backbone_history))

    def write_animations(self, directory: str) -> None:
        unbound_frames = self.backbone_history.get(BINDER_ALONE)
        ordered = sorted(self.backbone_history, key=lambda name: name == BINDER_ALONE)
        state_frames = {name: self.backbone_history[name] + (induced_fit_switch_frames(self.backbone_history[name][-1], unbound_frames[-1]) if unbound_frames and name != BINDER_ALONE else []) for name in ordered}
        write_trajectory_animation(state_frames, trajectory_output_path(directory, 'trajectory.html'))

    def write_trajectory_outputs(self, settings: dict) -> None:
        self.write_csv(trajectory_output_path(self.trajectory_directory, 'losses.csv'))
        if self.keep_sequences:
            self.write_sequences(trajectory_output_path(self.trajectory_directory, 'sequences.npz'))
        if settings.get('save_loss_plots', False):
            self.write_loss_plot(trajectory_output_path(self.trajectory_directory, 'losses.png'))
        if settings.get('save_design_animations', False):
            self.write_animations(self.trajectory_directory)
