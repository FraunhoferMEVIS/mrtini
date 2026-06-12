# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2026 Fraunhofer MEVIS, Germany
#
# This file is part of MRTini.
import torch

GAMMA_HZ = 42.577478e6  # Hz/T ~ 267.5e6 rad/s/T

SLICE_THICKNESS = 0.005 # in meters
PAD=2 # slice z grid padding factor
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"
DTYPE = torch.float32
Tissue_properties = {
    "3T": {
      "gm": {
        "PD": 0.8,
        "T1": 1.56,
        "T2": 0.083,
        "T2'": 0.32,
        "D": 0.83
      },
      "wm": {
        "PD": 0.7,
        "T1": 0.83,
        "T2": 0.075,
        "T2'": 0.18,
        "D": 0.65
      },
      "csf": {
        "PD": 1,
        "T1": 4.16,
        "T2": 1.65,
        "T2'": 0.059,
        "D": 3.19
      },
      "fat": {
        "PD": 1,
        "T1": 0.37,
        "T2": 0.125,
        "T2'": 0.012,
        "D": 0.1
      }
    }
}

      