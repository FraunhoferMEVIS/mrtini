# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native MRtini engine, distributed in the main MRtini wheel.

Retains the historical import for existing prepass callers.
"""
from pkgutil import extend_path
# A checkout can coexist with an installed release wheel without hiding its
# extension behind this source-only package directory. Editable builds also
# work: maturin places the extension alongside this file.
__path__=extend_path(__path__,__name__)
from . import mrtini_prepass_rs as _engine
from .mrtini_prepass_rs import *
__native_file__=_engine.__file__
