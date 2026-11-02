"""Independent service entry: reuse canonical checkpoint gate and shared-engine lock."""
from pathlib import Path
import runpy
import sys
ROOT=Path('/remote-home/jinminghao/OpenWAM')
sys.path.insert(0,str(ROOT))
from chunk_transport import install_server
install_server()
runpy.run_path(str(ROOT/'outputs/robotwin_eval/fullbackward32000_complete100_deploy_20261028T035728Z/serve_eager_cached.py'),run_name='__main__')
