"""Edit paths and sequence designs here, then run: python validation.py"""

# This directory lives at REPOSITORY_ROOT/figures/mrtini_validation_simple/.
# The runner locates REPOSITORY_ROOT by finding src/mrtini/mrtini_main.py.
import pathlib
MRTINI_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = "output"

SEQ_PATH = pathlib.Path(__file__).resolve().parent.parent / "src" / "sequence" / "gstar_seqs"

PHANTOM = {
    "path": str(MRTINI_ROOT / "data" / "subject04_3T-highres-fat_custom_dense.npz"),  # e.g. data/subject04_3T-highres-fat_custom_dense.npz
    "kind": "brainweb",  # or "simdata" for an MRzero SimData.load file
    "shape": (1, 1, 30),
    "slice_index": 14,
    "slice_thickness_m": 0.005,
}

# Supply GammaSTAR JSON files. Each refocusing RF is a separate MRtini rep.
# Keep phantom, echo timing, ADC and flip schedule matched across paired designs.
SEQUENCES = [
    {"name": "rare_tbw52", "path": str(SEQ_PATH / "RARE_sequence_raw.json"), "tbw": 5.2},
    # Primary: >=8 slice-selective refocusing RF reps, ADC at each echo,
    # imperfect 180-degree refocusing to generate stimulated echoes.
    # {"name": "rare_tbw8", "path": "", "tbw": 8.0},
    # # Matched RARE with TBW=8; adjust slice Gz consistently.
    # {"name": "rare_long_rf", "path": "", "tbw": 5.2},
    # # Matched RARE with longer RF duration.
    # {"name": "rare_strong_spoiler", "path": "", "tbw": 5.2},
    # # Matched RARE with stronger interecho spoiler moments.
    # {"name": "cpmg_variable_flip", "path": "", "tbw": 5.2},
    # # 8-16 refocusing pulses with variable flip schedule and each echo sampled.
    {"name": "flash_control", "path": str(SEQ_PATH / "FLASH_sequence_raw.json"), "tbw": 5.2},
    # Slice-selective FLASH, spoiled transverse carryover.
    # {"name": "multiband_optional", "path": "", "tbw": 5.2},
    # Optional genuine multiband RF waveform, only if that study is in scope.
]

# Explicit-z reference convergence is deferred; this is a fixed candidate.
ISO_SPIN_COUNT = 131072
EPSILONS = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6)
STATE_CAPS = (50, 100, 150, 200, 300, 400, 600, 800)
MIN_STATE_MAG_SWEEP = (1e-4, 5e-4, 1e-5, 5e-6, 1e-6)  # enough for now
RF_GRID_SWEEP = ()
# MIN_STATE_MAG_SWEEP = (1e-4,)
DEVICE = "cuda"  # "cpu" if your MRtini checkout supports its Rust/PyTorch path

# Set True only after removing the fat_mask zeroing line in execute_graph_analyze.py.
FAT_REPLAY_FIXED = True
HARD_RF_CONTROL = True  # Same MRtini graph, but Gz=0 during each RF solve.
MIN_STATE_MAG = 1e-4
RF_GRID_OVERSAMPLING = 1
MAX_STATES = 400

##  for result caching
#===========================================

CACHE_TAG = "v2_rand"
ISO_SPIN_DIST = "rand"  # or "rand"
ISO_SEED = 0