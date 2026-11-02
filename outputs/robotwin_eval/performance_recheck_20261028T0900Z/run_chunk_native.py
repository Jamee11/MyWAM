"""Independent native pilot entry; original expert/render/action loop remains."""
from pathlib import Path
import runpy
import sys
ROOT=Path('/remote-home/jinminghao/OpenWAM')
OUT=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z'
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks/robotwin'),str(OUT)]
from chunk_transport import install_client
install_client()
runpy.run_path(str(OUT/'run_native.py'),run_name='__main__')
