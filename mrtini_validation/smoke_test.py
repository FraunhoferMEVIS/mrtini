from pathlib import Path
import numpy as np
import validation as v
from result_cache import load_result

cases = [c for c in v.SEQUENCES if c["path"]]
if not cases: raise RuntimeError("No sequence path configured")
v.SEQUENCES = [next((c for c in cases if "flash" in c["name"].lower()), cases[0])]

v.OUTPUT = "output_smoke"
v.ISO_SPIN_COUNT = 131_072
v.MAX_STATES = 800
v.EPSILONS = (1e-6,)
v.STATE_CAPS = (800,)
v.RF_GRID_SWEEP = ()
v.MIN_STATE_MAG_SWEEP = (v.MIN_STATE_MAG,)
v.CACHE_TAG = "smoke_v1"
# Keep HARD_RF_CONTROL enabled if you also want to exercise that branch.
if __name__ == "__main__":
    v.run_all()

    folders = list((Path(v.HERE) / v.OUTPUT / "raw").rglob("plot_inputs.json"))
    assert folders, "Missing plot inputs"
    for index in folders:
        folder = index.parent
        ref = load_result(folder / "reference.npz")["signal"]
        assert np.isfinite(ref).all(), "Nonfinite ISO signal"
        runs = list(folder.glob("mrtini_*.npz"))
        assert runs, "Missing saved MRTini runs"
        for file in runs:
            result = load_result(file)
            assert result["signal"].shape == ref.shape, file
            assert np.iscomplexobj(result["signal"]), file
            assert np.isfinite(result["signal"]).all(), file
        load_result(folder / "acquisition.npz")

    print("SMOKE PASSED: simulation, figures and saved arrays completed.")