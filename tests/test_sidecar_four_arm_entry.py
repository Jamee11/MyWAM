"""Control input fingerprints and verified stop guards are part of the screen contract."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.supervise_sidecar_four_arm import AdoptedControl, stop_predecessor
from scripts.train_sidecar_four_arm import batch_fingerprint


def test_numeric_and_video_fingerprints_separate_changes():
    sample = dict(action=np.zeros((32, 80), np.float32), proprio=np.zeros((1, 80), np.float32),
                  prompt='same task', video=[Image.fromarray(np.zeros((4, 4, 3), np.uint8))])
    before, images = batch_fingerprint([sample]), batch_fingerprint([sample], video=True)
    sample['video'] = [Image.fromarray(np.ones((4, 4, 3), np.uint8))]
    assert batch_fingerprint([sample]) == before
    assert batch_fingerprint([sample], video=True) != images
    sample['action'][0, 0] = 1
    assert batch_fingerprint([sample]) != before


def test_wrong_process_identity_is_never_signalled(tmp_path):
    record = tmp_path / 'launch.json'
    record.write_text(json.dumps(dict(pid=os.getpid(), start_ticks='not_this_process')))
    with pytest.raises(RuntimeError, match='reused'):
        stop_predecessor(record, tmp_path)
    record.write_text(json.dumps(dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21])))
    with pytest.raises(RuntimeError, match='unique training output'):
        stop_predecessor(record, tmp_path)


@pytest.mark.parametrize('code', [0, 7])
def test_adopt_preserves_the_actual_unreaped_launcher_exit_status(tmp_path, code):
    child = subprocess.Popen([sys.executable, '-c', f'import time; time.sleep(.2); raise SystemExit({code})',
                              'torch.distributed.run', str(tmp_path)])
    try:
        (tmp_path / 'launch.json').write_text(json.dumps(dict(pid=child.pid,
            start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])))
        adopted = AdoptedControl(tmp_path)
        deadline = time.monotonic() + 10
        while adopted.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert adopted.returncode == code
        assert child.wait(timeout=5) == code
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
