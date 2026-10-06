#!/usr/bin/env python3
import runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
runpy.run_path(str(root/'tools/verify_release.py'),run_name='__main__')
