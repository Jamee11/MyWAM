"""Offline controller tests: no ROS installation, network, or robot required."""
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.real_robot import infer_openwam_clean as client


class Point:
    def __init__(self):
        self.time_from_start = NS(sec=0, nanosec=0)


class Trajectory:
    def __init__(self):
        self.points = []


class Pose:
    def __init__(self):
        self.header = NS(frame_id='', stamp=None)
        self.pose = NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.))


class Request:
    def __init__(self):
        self.ik_request = NS(timeout=NS(nanosec=0), robot_state=NS(joint_state=NS()))


class Future:
    def __init__(self, result):
        self.value = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self.value


class Model:
    def __init__(self):
        self.link_map = {n: n for n in (client.ARM_BASE_FRAME, 'link8', client.TCP_FRAME)}
        self.root_from_arm = np.eye(4)
        self.root_from_arm[:3, 3] = [.212713696, -.002069014, .776600015]
        self.tool = np.eye(4)
        self.tool[:3, :3] = Rotation.from_euler('z', -45, degrees=True).as_matrix()
        self.tool[2, 3] = .1054

    def link_fk(self, cfg):
        q = np.asarray(list(cfg.values()))
        link = np.eye(4)
        link[:3, 3] = q[:3]
        link[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        root_link = self.root_from_arm @ link
        return {client.ARM_BASE_FRAME: self.root_from_arm, 'link8': root_link,
                client.TCP_FRAME: root_link @ self.tool}


@pytest.fixture
def node(monkeypatch):
    clock = NS(now=0., node=None)

    def sleep(seconds):
        clock.now += seconds
        if clock.node is not None:
            clock.node._joint_time = clock.now  # A live feedback publisher.

    monkeypatch.setattr(client.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(client.time, 'sleep', sleep)

    class IK:
        def __init__(self):
            self.calls, self.fail_at, self.wrong_rotation = [], None, False

        def call_async(self, req):
            self.calls.append(req)
            p = req.ik_request.pose_stamped.pose
            quat = p.orientation
            rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_rotvec()
            q = [p.position.x, p.position.y, p.position.z, *rot, 0.]
            if self.wrong_rotation:
                q[3:6] = [0., 0., 0.]
            state = NS(name=list(reversed(clock.node.joint_names)), position=list(reversed(q)))
            code = -31 if len(self.calls) == self.fail_at else 1
            return Future(NS(error_code=NS(val=code, SUCCESS=1), solution=NS(joint_state=state)))

    class Infer:
        def __init__(self):
            self.joint_names = [f'j{i}' for i in range(7)]
            self.ik_link_name, self.group_name = 'link8', 'arm'
            self.fk_arm = NS(robot_model=Model())
            self.client = IK()
            self.trajectories, self.grips = [], []
            self.pub = NS(publish=self.publish)
            self.gripper_pub = NS(publish=lambda msg: self.grips.append((clock.now, msg.data)))

        def create_subscription(self, *args):
            return NS()

        def get_clock(self):
            return NS(now=lambda: NS(to_msg=lambda: NS()))

        def publish(self, trajectory):
            self.trajectories.append(trajectory)
            self._joint_cb(NS(name=self.joint_names, position=trajectory.points[-1].positions))

    for name, attrs in {
        'infer': {'Infer': Infer}, 'geometry_msgs.msg': {'PoseStamped': Pose},
        'moveit_msgs.srv': {'GetPositionIK': NS(Request=Request)},
        'sensor_msgs.msg': {'JointState': NS}, 'std_msgs.msg': {'Float32': NS},
        'trajectory_msgs.msg': {'JointTrajectory': Trajectory, 'JointTrajectoryPoint': Point},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    value = client.build_node()
    clock.node = value
    value._joint_cb(NS(name=value.joint_names, position=[.5, 0., .2, 0., 0., 0., 0.]))
    yield value, clock
    value.session.close()


def actions(node):
    _, link = node.robot_pose()
    tcp = link @ node.link_from_tcp
    a = np.tile(np.r_[tcp[:3, 3], tcp[:3, 0], tcp[:3, 1], 1.], (32, 1))
    a[:, 0] += np.arange(1, 33) * .0005
    a[16:, 9] = 0.
    return a


def test_reset_uses_model_tcp_and_explicit_arm_base(node):
    n, _ = node
    n.reset()
    _, link = n.robot_pose()
    np.testing.assert_allclose(link @ n.link_from_tcp, client.action_to_tcp(client.RESET_EEF), atol=1e-7)
    assert n.client.calls[-1].ik_request.pose_stamped.header.frame_id == client.ARM_BASE_FRAME
    assert n.base_frame == client.ARM_BASE_FRAME
    assert n.grips[0][1] == 1.
    assert n.trajectories[0].points[-1].time_from_start.sec >= 3


def test_full_prefix_is_one_continuous_trajectory_with_timed_gripper(node):
    n, _ = node
    a = actions(n)
    n.execute_chunk(a)
    assert len(n.client.calls) == 32 and len(n.trajectories) == 1
    points = n.trajectories[0].points
    assert len(points) == 33  # Current state at t=0, then 32 predicted targets.
    ts = [p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in points]
    np.testing.assert_allclose(ts, np.arange(33) * client.ACTION_DT, atol=1e-8)
    assert client.ACTION_DT == .5
    assert n.grips == [(pytest.approx(16 * client.ACTION_DT), 0.)]
    assert pose_is_equal(n.robot_pose()[1] @ n.link_from_tcp, client.action_to_tcp(a[-1]))


def pose_is_equal(a, b):
    pos, angle = client.pose_error(a, b)
    return pos < 1e-7 and angle < 1e-7


def test_ik_failure_sends_no_partial_prefix(node):
    n, _ = node
    n.client.fail_at = 10
    with pytest.raises(RuntimeError, match='IK failed'):
        n.execute_chunk(actions(n))
    assert n.trajectories == [] and n.grips == []


def test_rotation_only_ik_mismatch_is_detected(node):
    n, _ = node
    q, target = n.robot_pose()
    target[:3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    n.client.wrong_rotation = True
    with pytest.raises(RuntimeError, match='IK/FK mismatch'):
        n.solve_ik(target, q)
    assert n.trajectories == []


def test_speed_retiming_and_hold(node):
    n, _ = node
    q = np.zeros(7)
    targets = np.array([q.copy(), q.copy()])
    targets[:, 0] = [.1, .2]
    np.testing.assert_allclose(client.joint_times(q, targets), [.5, 1.])
    targets[:, 0] = [.4, .8]
    np.testing.assert_allclose(client.joint_times(q, targets), [.8, 1.6])
    before = n.robot_pose()[0].copy()
    n.hold()
    np.testing.assert_allclose(n.trajectories[-1].points[-1].positions, before)


def test_pose_and_task_list_match_requested_contract():
    import json
    from pathlib import Path
    ref = json.loads((Path(__file__).resolve().parents[2] /
        'tests/fixtures/maniskill20872_reset_pose.json').read_text())
    np.testing.assert_array_equal(client.RESET_EEF, ref['eef10'])
    assert client.CHUNK_STEPS == 32
    assert len(client.TASKS) == 8
    assert client.TASKS[5] == 'place the red sphere in the bin'
    assert client.TASKS[-1] == 'stand the peg upright on the table'
