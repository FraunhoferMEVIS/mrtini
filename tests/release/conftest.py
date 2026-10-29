# SPDX-License-Identifier: AGPL-3.0-or-later
import torch

def pytest_sessionstart(session):
    torch.set_num_threads(2)
