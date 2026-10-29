# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real installed imports, native exports and a small executed RF layer."""
import importlib.metadata, json, os, subprocess, sys
import pytest
from src.mrtini.doctor import check, expected_triton, kernel_smoke

def test_installed_engine_and_historical_import():
    import mrtini_prepass_rs as native
    from src.mrtini.optimized import compute_graph_fast,execute_graph_fast
    assert callable(compute_graph_fast) and callable(execute_graph_fast)
    from src.mrtini.simulation import compute_graph,execute_graph
    assert compute_graph is compute_graph_fast and execute_graph is execute_graph_fast
    assert all(callable(getattr(native,n)) for n in ('compute_graph_multi','compute_graph_packed','apply_rf_cpu'))
    assert importlib.metadata.version('MRtini')

def test_cpu_installation_check_really_executes_kernel():
    result=check(kernel=True)
    assert result['rf_kernel_check']['backend']=='native'
    assert result['rf_kernel_check']['relative_error']<3e-5

@pytest.mark.parametrize('version,expected',[('2.8.0+cu129','3.4'),('2.9.1+cu128','3.5'),('2.15.0+cpu','3.9')])
def test_version_compatibility(version,expected):assert expected_triton(version)==expected

def test_unknown_version_is_not_silently_paired():
    with pytest.raises(RuntimeError,match='No declared'):expected_triton('9.1.0')

def test_installed_wheel_works_outside_the_checkout(tmp_path):
    # An editable install intentionally depends on its checkout; release wheels
    # must independently import outside it. CI sets this only for wheel checks.
    if os.environ.get('MRTINI_TEST_INSTALLED_WHEEL')!='1':pytest.skip('Wheel-isolation check is enabled by the wheel CI job')
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    p=subprocess.run([sys.executable,'-m','src.mrtini.doctor','--kernel'],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert '[MRtini doctor] OK' in p.stdout
