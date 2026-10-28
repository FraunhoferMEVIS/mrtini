"""Load complete production modules, isolating optional I/O/UI/native imports.

No simulator, tensor operation, or numerical solver is mocked. Package __init__
side effects are bypassed so unit tests do not start the application or GPU.
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

def pytest_configure(config):
    config.addinivalue_line("markers", "regression: behavioral contract previously violated by an older snapshot")

@pytest.fixture(scope="session")
def modules():
    mp = pytest.MonkeyPatch()
    def namespace(name, path=None):
        m = types.ModuleType(name)
        m.__path__ = [] if path is None else [str(path)]
        mp.setitem(sys.modules, name, m)
        return m
    for name, path in [("src", ROOT / "src"), ("src.mrtini", ROOT / "src/mrtini"),
                       ("src.sequence", ROOT / "src/sequence")]:
        namespace(name, path)
    # Sequence importers and visualization are outside these isolated unit tests.
    star = namespace("mrstar")
    star.mrstar = namespace("mrstar.mrstar")
    namespace("pydisseqt")
    namespace("utils")
    viz = namespace("utils.viz")
    viz.pv_bloch_plotter_3D_pos = Mock(side_effect=AssertionError("Unexpected UI call"))
    viz.plot_slice_profile = Mock(side_effect=AssertionError("Unexpected plotting call"))
    helpers = namespace("utils.helpers")
    helpers.make_virtual_z_samples = Mock(side_effect=AssertionError("Unexpected diagnostic helper"))
    helpers.synthesize_Mxyz_from_dists = Mock(side_effect=AssertionError("Unexpected diagnostic helper"))
    namespace("src.voxelgridphantom")
    namespace("src.voxelgridphantom.sim_data").SimData = object
    namespace("MRzeroCore"); namespace("MRzeroCore.phantom")
    namespace("MRzeroCore.phantom.sim_data").SimData = object
    namespace("line_profiler").profile = lambda f: f
    native = namespace("mrtini_prepass_rs")
    native.compute_graph = Mock(name="native.compute_graph")
    native.compute_graph_multi = Mock(name="native.compute_graph_multi")
    loaded = {}
    def load(short, rel, name):
        path = ROOT / rel
        if not path.is_file():
            pytest.fail(f"Required MRtini source missing: {path}. Extract this package at the repository root.")
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, name, m)
        spec.loader.exec_module(m)
        loaded[short] = m
        return m
    try:
        cfg = load("config", "src/mrtini/config.py", "src.mrtini.config")
        mp.setattr(cfg, "DEVICE", "cpu")
        load("constants", "src/mrtini/constants.py", "src.mrtini.constants")
        load("rf", "src/mrtini/RF_banks.py", "src.mrtini.RF_banks")
        load("bloch", "src/mrtini/bloch.py", "src.mrtini.bloch")
        load("sequence", "src/sequence/sequence.py", "src.sequence.sequence")
        load("slice", "src/sequence/slice_select.py", "src.sequence.slice_select")
        load("main", "src/mrtini/mrtini_main.py", "src.mrtini.mrtini_main")
        load("bridge", "src/mrtini/pre_pass_rust_call.py", "src.mrtini.pre_pass_rust_call")
        load("executor", "src/mrtini/execute_graph.py", "src.mrtini.execute_graph")
        loaded["native"] = native
        yield SimpleNamespace(**loaded)
    finally:
        mp.undo()

@pytest.fixture
def rf(modules): return modules.rf

@pytest.fixture
def bloch(modules): return modules.bloch

@pytest.fixture
def seqmod(modules): return modules.sequence

@pytest.fixture
def sm(modules): return modules.slice

@pytest.fixture
def main(modules): return modules.main

@pytest.fixture
def zgrid(modules):
    return torch.arange(-4, 5, dtype=modules.config.DTYPE) * 1e-3

@pytest.fixture
def bank(rf, zgrid):
    ax = rf.make_axes([-10., 10.], [0.8, 1.2], device="cpu")
    angles = torch.tensor([0., 0.3, 0.6, 0.9], dtype=rf.DTYPE)
    mats = torch.tensor(Rotation.from_rotvec(np.column_stack([angles.numpy(), np.zeros((4,2))])).as_matrix(), dtype=rf.DTYPE)
    flat = mats[:, None].expand(-1, len(zgrid), -1, -1).clone()
    b = rf.RFKernel6BankSO3.from_Rxyz_flat(flat, ax, zgrid, device="cpu")
    b.attach_kgrid(rf.build_kgrid_projection(z_m=zgrid, device="cpu"))
    return b

@pytest.fixture
def repetition(seqmod):
    r = seqmod.Repetition.zero(3)
    r.event_time[:] = torch.tensor([0.01,0.02,0.03])
    r.gradm[:] = torch.tensor([[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
    r.adc_usage[:] = torch.tensor([0,1,2])
    r.adc_phase[:] = torch.tensor([0.,0.2,0.4])
    return r
