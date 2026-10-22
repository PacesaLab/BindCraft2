"""Private Apple campaign argument transport and coordinator lease."""
import json
import os
import pickle
import sys
import tempfile
from bindcraft import campaign as shared
from bindcraft._worker_lifecycle import SupervisorAdmission

def _dispatch_campaign(settings: dict, project_folder: str, af2_weights, mpnn_weights, key, max_trajectories, metadata, *, coordinator_ready=None) -> int | None:
    if not shared._metal_campaign_active() or shared.running_as_design_worker():
        return None
    if max_trajectories is None:
        max_trajectories = settings.get('max_trajectories')
    if max_trajectories is None and (not mpnn_weights or settings.get('trajectory_only')):
        max_trajectories = shared.DEFAULT_TRAJECTORY_ONLY_BUDGET
    budget_settings = {**settings, 'max_trajectories': max_trajectories} if max_trajectories is not None else settings
    # Publish coordinator artifacts only while this invocation owns the lease.
    # Pass the same admission to the Apple launcher to avoid a nested lease.
    with SupervisorAdmission(settings, project_folder, max_trajectories=shared.campaign_trajectory_budget(budget_settings)) as admission:
        if coordinator_ready is not None:
            coordinator_ready()
        return _dispatch_campaign_workers(settings, project_folder, af2_weights, mpnn_weights, key,
                                          max_trajectories, metadata, admission)

def _dispatch_campaign_workers(settings, project_folder, af2_weights, mpnn_weights, key, max_trajectories, metadata, admission):
    # The launcher owns process lifecycle. This private entry preserves Python
    # arguments (including typed/legacy JAX keys) without reconstructing them
    # from a settings file or replacing explicitly supplied weight paths.
    with tempfile.TemporaryDirectory(prefix='bindcraft-worker-') as payload_directory:
        payload_path = os.path.join(payload_directory, 'arguments.pickle')
        with open(payload_path, 'wb') as payload:
            pickle.dump(dict(settings=settings, project_folder=project_folder, af2_weights=af2_weights,
                             mpnn_weights=mpnn_weights, key=key, max_trajectories=max_trajectories,
                             metadata=metadata), payload)
        budget_settings = {**settings, 'max_trajectories': max_trajectories} if max_trajectories is not None else settings
        log_directory = os.path.join(project_folder, 'workers')
        status = shared.dispatch_design_workers({**settings, 'project_folder': project_folder}, log_directory,
                                         shared.design_residue_count(settings),
                                         worker_command=[sys.executable, '-u', '-m', 'bindcraft._worker_lifecycle', payload_path],
                                         project_folders=(project_folder,), trajectory_budget=shared.campaign_trajectory_budget(budget_settings),
                                         admission=admission)
        if status is not None:
            # Preserve the launcher's resolved settings artifact for operators
            # even when its children use the explicit Python argument entry.
            from bindcraft.campaign_output import json_compatible
            os.makedirs(log_directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=log_directory, delete=False) as snapshot:
                json.dump(json_compatible({**budget_settings, 'project_folder': project_folder}), snapshot, sort_keys=True)
            os.replace(snapshot.name, os.path.join(log_directory, 'campaign_settings.json'))
        return status
