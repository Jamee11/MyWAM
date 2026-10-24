import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.real_robot import infer_openwam as simple


def test_absolute_pose_with_noncommuting_base_and_tcp(monkeypatch):
    base=np.eye(4);base[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];base[:3,3]=[.1,.2,.3]
    tool=np.eye(4);tool[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]];tool[2,3]=.1034
    link=np.eye(4);link[:3,3]=[.4,-.2,.1]
    monkeypatch.setattr(simple,'T_MODEL_FROM_REAL_BASE',base)
    monkeypatch.setattr(simple,'T_LINK_FROM_TCP',tool)
    state=simple.current_state(link,1)
    predicted_link,predicted_tcp=simple.target_link(state)
    np.testing.assert_allclose(predicted_link,link,atol=1e-7)
    np.testing.assert_allclose(predicted_tcp,link@tool,atol=1e-7)
    with pytest.raises(ValueError):simple.action_to_tcp(np.zeros(7))
    with pytest.raises(ValueError):simple.action_to_tcp(np.zeros(10))


def test_execute_cannot_skip_calibration(monkeypatch):
    monkeypatch.setattr(simple,'API_TOKEN','fixture-not-a-real-token')
    monkeypatch.setattr(simple,'EXECUTE',True)
    monkeypatch.setattr(simple,'CALIBRATION_CONFIRMED',False)
    monkeypatch.setattr(simple.requests,'get',lambda *a,**k:pytest.fail('No HTTP/robot action before validation'))
    with pytest.raises(ValueError,match='calibration'):simple.main()


def test_single_file_reuses_original_infer_and_rejects_failed_prefix(monkeypatch):
    publications=[]
    class OriginalInfer:
        def __init__(self):
            self.base_frame='base_link';self._lock=threading.Lock()
            self.fk_arm=SimpleNamespace(robot_model=SimpleNamespace(base_link=SimpleNamespace(name='base_link')))
            self.pub=self.gripper_pub=SimpleNamespace(publish=lambda x:publications.append(x))
        def create_subscription(self,*a,**kw):return object()
    monkeypatch.setitem(sys.modules,'infer',SimpleNamespace(Infer=OriginalInfer))
    monkeypatch.setitem(sys.modules,'sensor_msgs.msg',SimpleNamespace(JointState=object))
    monkeypatch.setitem(sys.modules,'rclpy.qos',SimpleNamespace(qos_profile_sensor_data=object()))
    monkeypatch.setitem(sys.modules,'trajectory_msgs.msg',SimpleNamespace(JointTrajectory=object,JointTrajectoryPoint=object))
    monkeypatch.setitem(sys.modules,'std_msgs.msg',SimpleNamespace(Float32=object))
    monkeypatch.setattr(simple,'WORKSPACE_MIN',[-1]*3);monkeypatch.setattr(simple,'WORKSPACE_MAX',[1]*3)
    monkeypatch.setattr(simple,'EXECUTE_STEPS',2)
    node=simple.build_node();assert isinstance(node,OriginalInfer)
    node.observed_q=np.zeros(7);node.observed_link=np.eye(4)
    calls=[]
    def solve(*a):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('IK failed')
        return np.zeros(7)
    node.ik_fun=solve
    a=np.array([0,0,0,1,0,0,0,1,0,1],dtype=float)
    with pytest.raises(RuntimeError,match='IK failed'):node.execute_absolute(np.tile(a,(32,1)))
    assert len(calls)==2 and not publications


