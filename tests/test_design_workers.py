"""Device enumeration and worker pinning, on an accelerator nobody here has.

Every test runs on a machine with no accelerator at all: the accelerators are stubs standing in
for what a plugin reports, and the two tests that describe an NVIDIA machine drive the real
nvidia-smi parsing through a fake nvidia-smi on PATH.
"""
import os
import subprocess
import sys
import pytest
from bindcraft import design_workers

VISIBILITY_VARIABLES = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')


class StubDevice:

    def __init__(self, identifier: int, platform: str = 'cuda', memory: dict | None = None) -> None:
        self.id, self.platform, self.memory = identifier, platform, memory

    def memory_stats(self) -> dict | None:
        return self.memory


@pytest.fixture(autouse=True)
def unpinned_host(monkeypatch, tmp_path):
    """No inherited pinning, and no nvidia-smi, unless a test asks for one."""
    for variable in VISIBILITY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv('PATH', str(tmp_path / 'empty-path'))
    #raising=False so this file also runs against a tree that has no design_devices yet,
    #which is how the tests below are shown to be red before the change and green after it.
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [], raising=False)


def fake_nvidia_smi(tmp_path, monkeypatch, body: str) -> None:
    binary_directory = tmp_path / 'bin'
    binary_directory.mkdir(exist_ok=True)
    tool = binary_directory / 'nvidia-smi'
    tool.write_text(body)
    tool.chmod(0o755)
    monkeypatch.setenv('PATH', str(binary_directory))


def refuse_jax(monkeypatch) -> None:
    def refused():
        raise AssertionError('JAX was consulted on a machine nvidia-smi had already answered for')
    monkeypatch.setattr(design_workers, 'design_devices', refused, raising=False)


# --- enumeration -------------------------------------------------------------------------

def test_a_machine_with_no_accelerator_finds_no_device():
    """The host CPU JAX always reports must not be mistaken for a device to design on."""
    assert design_workers.visible_design_gpus() == []


def test_the_host_cpu_is_never_a_design_device():
    """design_devices runs against the real JAX installed here, which on this machine is CPU-only."""
    assert design_workers.design_devices() == []


def test_cuda_visible_devices_is_taken_verbatim_and_jax_is_not_consulted(monkeypatch):
    refuse_jax(monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    assert design_workers.visible_design_gpus() == ['2', '3']


def test_an_empty_cuda_visible_devices_still_finds_nothing(monkeypatch):
    """Slurm sets it empty when the allocation asked for no GPU, and that must stay an empty list."""
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert design_workers.visible_design_gpus() == []


def test_nvidia_smi_answers_before_jax_is_asked(tmp_path, monkeypatch):
    """An NVIDIA machine is enumerated by exactly the call it was enumerated by before."""
    refuse_jax(monkeypatch)
    fake_nvidia_smi(tmp_path, monkeypatch, '#!/bin/sh\nprintf "0\\n1\\n2\\n3\\n"\n')
    assert design_workers.visible_design_gpus() == ['0', '1', '2', '3']


def test_jax_enumerates_when_there_is_no_nvidia_smi(monkeypatch):
    """Fails before this change: a ROCm machine enumerated to nothing."""
    monkeypatch.setattr(design_workers, 'design_devices',
                        lambda: [StubDevice(0, 'rocm'), StubDevice(1, 'rocm')], raising=False)
    assert design_workers.visible_design_gpus() == ['0', '1']


def test_hip_visible_devices_is_honoured_like_its_cuda_counterpart(monkeypatch):
    """Fails before this change: only CUDA_VISIBLE_DEVICES was read."""
    refuse_jax(monkeypatch)
    monkeypatch.setenv('HIP_VISIBLE_DEVICES', '1,2')
    assert design_workers.visible_design_gpus() == ['1', '2']


# --- memory ------------------------------------------------------------------------------

def test_nvidia_smi_memory_is_preferred_and_parsed_as_before(tmp_path, monkeypatch):
    refuse_jax(monkeypatch)
    fake_nvidia_smi(tmp_path, monkeypatch, '#!/bin/sh\nprintf "0, 40960, 81920\\n"\n')
    assert design_workers.design_gpu_memory_gb() == {'0': (40.0, 80.0)}


def test_jax_reports_device_memory_when_there_is_no_nvidia_smi(monkeypatch):
    """Fails before this change: memory was unknown off NVIDIA, which held every card to one worker."""
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [
        StubDevice(0, 'rocm', {'bytes_limit': 64 * 1024 ** 3, 'bytes_in_use': 16 * 1024 ** 3})])
    assert design_workers.design_gpu_memory_gb() == {'0': (48.0, 64.0)}


def test_a_plugin_that_reports_no_memory_is_left_unsized(monkeypatch):
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, 'rocm', None)], raising=False)
    assert design_workers.design_gpu_memory_gb() == {}


# --- which variable pins a worker ----------------------------------------------------------

@pytest.mark.parametrize('platform, expected', [('cuda', 'CUDA_VISIBLE_DEVICES'),
                                                ('rocm', 'HIP_VISIBLE_DEVICES')])
def test_each_platform_is_pinned_by_its_own_runtime_variable(monkeypatch, platform, expected):
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, platform)], raising=False)
    assert design_workers.design_visibility_variable() == expected


def test_an_unknown_platform_has_no_way_to_pin_a_worker(monkeypatch):
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, 'wildly-new')], raising=False)
    monkeypatch.setattr(design_workers, 'installed_accelerator_platforms', set, raising=False)
    assert design_workers.design_visibility_variable() is None


def test_jaxs_shared_gpu_name_is_resolved_by_the_installed_plugin(monkeypatch):
    """JAX calls both CUDA and ROCm devices 'gpu' in some builds; the plugin says which it is."""
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, 'gpu')], raising=False)
    monkeypatch.setattr(design_workers, 'installed_accelerator_platforms', lambda: {'rocm'}, raising=False)
    assert design_workers.design_visibility_variable() == 'HIP_VISIBLE_DEVICES'


def test_an_inherited_pinning_variable_is_the_one_reused(monkeypatch):
    refuse_jax(monkeypatch)
    monkeypatch.setenv('HIP_VISIBLE_DEVICES', '1')
    assert design_workers.design_visibility_variable() == 'HIP_VISIBLE_DEVICES'


# --- the worker processes ------------------------------------------------------------------

class RecordingPopen:
    launched: list = []

    def __init__(self, command, env=None, **keywords):
        RecordingPopen.launched.append(env)
        self.stdout = iter(())

    def wait(self):
        return 0

    def poll(self):
        return 0


@pytest.fixture
def launched(monkeypatch):
    RecordingPopen.launched = []
    monkeypatch.setattr(design_workers.subprocess, 'Popen', RecordingPopen)
    return RecordingPopen.launched


def two_worker_plan():
    return [{'gpu': '0', 'memory_fraction': 0.0}, {'gpu': '1', 'memory_fraction': 0.0}]


def test_a_cuda_worker_is_still_pinned_with_cuda_visible_devices(monkeypatch, tmp_path, launched):
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, 'cuda')], raising=False)
    design_workers.launch_design_workers(two_worker_plan(), str(tmp_path / 'logs'), ['true'])
    assert [environment['CUDA_VISIBLE_DEVICES'] for environment in launched] == ['0', '1']


def test_a_rocm_worker_is_pinned_with_hip_visible_devices(monkeypatch, tmp_path, launched):
    """Fails before this change: a ROCm worker was handed CUDA_VISIBLE_DEVICES, which HIP ignores,
    so every worker opened every device."""
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, 'rocm')], raising=False)
    design_workers.launch_design_workers(two_worker_plan(), str(tmp_path / 'logs'), ['true'])
    assert [environment['HIP_VISIBLE_DEVICES'] for environment in launched] == ['0', '1']
    assert 'CUDA_VISIBLE_DEVICES' not in launched[0]


def test_a_worker_still_carries_its_index_and_count(monkeypatch, tmp_path, launched):
    monkeypatch.setattr(design_workers, 'design_devices', lambda: [StubDevice(0, 'cuda')], raising=False)
    design_workers.launch_design_workers(two_worker_plan(), str(tmp_path / 'logs'), ['true'])
    assert [environment['BINDCRAFT_WORKER_ID'] for environment in launched] == ['0', '1']
    assert {environment['BINDCRAFT_WORKER_COUNT'] for environment in launched} == {'2'}


# --- the plan ------------------------------------------------------------------------------

BASE_SETTINGS = {'binder_lengths': [70], 'trajectory_only': True}


def test_an_unpinnable_platform_does_not_fan_out(monkeypatch, capsys):
    """Without a pinning variable every worker would open device 0, which is worse than one worker."""
    monkeypatch.setattr(design_workers, 'design_devices',
                        lambda: [StubDevice(0, 'wildly-new'), StubDevice(1, 'wildly-new')], raising=False)
    monkeypatch.setattr(design_workers, 'installed_accelerator_platforms', set, raising=False)
    monkeypatch.setattr(design_workers, 'campaign_length_buckets', lambda settings: ())
    plan = design_workers.plan_design_workers(BASE_SETTINGS)
    assert {worker['gpu'] for worker in plan} == {'0'}
    assert 'no way to pin a worker' in capsys.readouterr().out


def test_a_rocm_machine_plans_one_worker_per_device(monkeypatch):
    """Fails before this change: the plan was empty, so run_campaign never fanned out at all."""
    monkeypatch.setattr(design_workers, 'design_devices',
                        lambda: [StubDevice(0, 'rocm'), StubDevice(1, 'rocm')], raising=False)
    monkeypatch.setattr(design_workers, 'campaign_length_buckets', lambda settings: ())
    plan = design_workers.plan_design_workers(BASE_SETTINGS)
    assert [worker['gpu'] for worker in plan] == ['0', '1']
