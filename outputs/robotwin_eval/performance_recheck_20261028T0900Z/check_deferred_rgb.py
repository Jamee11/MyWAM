"""Check deferred pixels, no-read deepcopy, caching and expired-frame rejection."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sim_hotpath import DeferredRGB

class Camera:
    def __init__(self):
        self.calls=0
        self.value=np.arange(4*5*4,dtype=np.float32).reshape(4,5,4)/79
    def get_picture(self,name):
        assert name=='Color';self.calls+=1;return self.value
owner=SimpleNamespace(_deferred_frame=1);camera=Camera()
value=DeferredRGB(camera,owner,1)
assert copy.deepcopy(value) is value and camera.calls==0
expected=(camera.value*255).clip(0,255).astype('uint8')[:,:,:3]
assert np.array_equal(np.asarray(value),expected) and camera.calls==1
assert value.tobytes()==expected.tobytes() and camera.calls==1
expired=DeferredRGB(camera,owner,1);owner._deferred_frame=2
try:np.asarray(expired)
except RuntimeError:pass
else:raise AssertionError('Expired frame was silently read')
assert np.array_equal(np.asarray(value),expected)
report=dict(passed=True,pixels_exact=True,no_read_on_deepcopy=True,resolved_frame_cached=True,expired_unresolved_frame_rejected=True)
Path(__file__).with_name('deferred_rgb_cpu_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
