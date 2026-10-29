# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installation plan and container wiring; does not install packages or use Docker."""
import importlib.util, json, subprocess, sys
from pathlib import Path
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]

def setup_module_source():
    path=ROOT/'scripts/setup_mrtini.py'
    spec=importlib.util.spec_from_file_location('mrtini_setup_under_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_cuda_platform_selection_is_explicit():
    m=setup_module_source();m.supported_platform('cuda','Windows','AMD64');m.supported_platform('cuda','Linux','x86_64')
    with pytest.raises(RuntimeError):m.supported_platform('cuda','Darwin','arm64')

def test_setup_dry_run_has_one_integrated_install_and_no_legacy_build():
    p=subprocess.run([sys.executable,str(ROOT/'scripts/setup_mrtini.py'),'--device','cpu','--dev','--skip-mrstar','--dry-run'],cwd=ROOT,text=True,capture_output=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert '+cpu' in p.stdout and '[dev]' in p.stdout
    assert 'maturin develop' not in p.stdout and 'install_fast_prepass' not in p.stdout
    assert 'no installation performed' in p.stdout

@pytest.mark.parametrize('file,gpu',[('docker-compose.yml',True),('docker-compose.cpu.yml',False)])
def test_compose_paths_and_device(file,gpu):
    cfg=yaml.safe_load((ROOT/'.devcontainer'/file).read_text());dev=cfg['services']['mrtini-dev'];recon=cfg['services']['gammastar-recon']
    assert dev['environment']['MRTINI_DEVICE']==('cuda' if gpu else 'cpu')
    assert ('gpus' in dev)==gpu
    assert recon['build']['context']=='../external/gammaSTAR/sequenceServer/gammastar_recon'
    assert dev['environment']['GAMMASTAR_RECON_HOST']=='gammastar-recon'

def test_devcontainer_is_cross_platform_and_headless():
    cfg=json.loads((ROOT/'.devcontainer/devcontainer.json').read_text())
    assert cfg['initializeCommand']==['python','scripts/fetch_gammastar_recon.py']
    docker=(ROOT/'.devcontainer/Dockerfile').read_text()
    assert 'MPLBACKEND=Agg' in docker and 'QtAgg' not in docker

def test_mrstar_selects_a_compatible_wheel_and_rejects_wrong_python(tmp_path):
    m=setup_module_source()
    from packaging.tags import sys_tags
    tag=next(sys_tags());right=tmp_path/f'mrstar-0.1.0-{tag}.whl';right.touch()
    (tmp_path/'mrstar-99.0.0-cp39-cp39-win32.whl').touch()
    assert m.compatible_mrstar(tmp_path)==right
    right.unlink()
    with pytest.raises(RuntimeError,match='No compatible'):m.compatible_mrstar(tmp_path)
