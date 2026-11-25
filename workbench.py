"""Launch from the MRTini repository root: python workbench.py"""
import runpy, sys
from pathlib import Path
root=Path(__file__).resolve().parent
pkg=root/'mrtini_workbench'
sys.path.insert(0,str(pkg))
if '--mrtini-root' not in sys.argv:sys.argv += ['--mrtini-root',str(root)]
runpy.run_path(str(pkg/'server.py'),run_name='__main__')
