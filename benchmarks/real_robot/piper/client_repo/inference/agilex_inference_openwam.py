"""ROS1 Piper client. No reset/enable commands; dry-run unless --execute is supplied."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from contract import SCHEMA, Frames, action_chunk, guard_targets
from temporal import StreamActionBuffer


class Policy:
    def __init__(self, host, port, token):
        self.url = f'http://{host}:{port}'
        self.session = requests.Session()
        self.session.trust_env = False  # Robot traffic must not take the shell's Clash proxy.
        self.session.headers['Authorization'] = 'Bearer ' + token

    def infer(self, images, state, prompt):
        from PIL import Image
        query = dict(schema=SCHEMA, state=state.tolist(), prompt=prompt)
        files = {'json': ('request.json', json.dumps(query).encode(), 'application/json')}
        for name, array in zip(('front', 'left', 'right'), images):
            stream = io.BytesIO()
            Image.fromarray(array).save(stream, format='PNG')
            files[name] = (name+'.png', stream.getvalue(), 'image/png')
        response = self.session.post(self.url+'/api/inference', files=files, timeout=60)
        response.raise_for_status()
        result = response.json()
        if result.get('schema') != SCHEMA:
            raise ValueError('Server is not an OpenWAM Piper EEF20 service')
        return action_chunk(result['actions'])


def pose_matrix(message):
    p, q = message.pose.position, message.pose.orientation
    matrix = np.eye(4)
    matrix[:3, 3] = [p.x, p.y, p.z]
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5902)
    p.add_argument('--prompt', required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).parents[1]/'robot.json')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--publish_rate', type=float, default=30)
    p.add_argument('--inference_rate', type=float, default=2)
    p.add_argument('--chunk_size', type=int, default=32)
    p.add_argument('--use_temporal_smoothing', action='store_true')
    p.add_argument('--latency_k', type=int, default=0)
    p.add_argument('--min_smooth_steps', type=int, default=12)
    p.add_argument('--steps', type=int, default=0, help='Inference requests; 0 means unlimited')
    args = p.parse_args()
    if not 1 <= args.chunk_size <= 32 or args.publish_rate <= 0 or args.inference_rate <= 0:
        p.error('chunk_size must be 1..32; rates must be positive')
    cfg = json.loads(args.config.read_text())
    frames = Frames(cfg)
    if args.execute and not cfg['geometry_verified']:
        p.error('First verify base/tool alignment and workspace in robot.json using a known pose')
    token = os.environ.get('OPENWAM_API_TOKEN', '').strip()
    if len(token) < 16:
        p.error('Set OPENWAM_API_TOKEN to the same value as the server token')
    policy = Policy(args.host, args.port, token)

    import message_filters
    import rospy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from piper_msgs.msg import PosCmd
    from sensor_msgs.msg import Image, JointState

    rospy.init_node('openwam_piper_client', anonymous=True)
    bridge = CvBridge()
    latest = [None]
    names = ('front', 'left_image', 'right_image', 'left_pose', 'right_pose', 'left_state', 'right_state')
    types = (Image, Image, Image, PoseStamped, PoseStamped, JointState, JointState)
    subs = [message_filters.Subscriber(cfg['topics'][name], kind) for name, kind in zip(names, types)]
    sync = message_filters.ApproximateTimeSynchronizer(subs, 20, .15)

    def receive(front, left, right, lp, rp, lj, rj):
        poses = [pose_matrix(lp), pose_matrix(rp)]
        grip = []
        for message in (lj, rj):
            # The existing driver names its seventh position joint6; the array order is documented.
            if len(message.position) != 7:
                raise ValueError('Need 6 joints + measured gripper opening in metres')
            grip.append(message.position[6])
        state = frames.state(poses, grip)
        images = [bridge.imgmsg_to_cv2(im, 'rgb8') for im in (front, left, right)]
        stamp = min(m.header.stamp.to_sec() for m in (front, left, right, lp, rp, lj, rj))
        latest[0] = (images, state, stamp)

    sync.registerCallback(receive)
    publishers = [rospy.Publisher(cfg['topics'][side+'_command'], PosCmd, queue_size=1)
                  for side in ('left', 'right')] if args.execute else []
    print('EXECUTE' if args.execute else 'DRY RUN: no control publishers', flush=True)
    buffer = StreamActionBuffer(state_dim=20)
    rate = rospy.Rate(args.publish_rate)
    future, previous, last_request, completed = None, None, 0., 0
    # One inference in flight, as in Kai0; ROS callbacks continue during HTTP calls.
    with ThreadPoolExecutor(max_workers=1) as pool:
        while not rospy.is_shutdown():
            observation = latest[0]
            if observation is None:
                rate.sleep()
                continue
            if rospy.Time.now().to_sec()-observation[2] > 1.:
                if args.execute:
                    raise RuntimeError('Robot observations stopped; stopping command publication')
                rate.sleep()
                continue
            if future is not None and future.done():
                actions = future.result()[:args.chunk_size]
                completed += 1
                future = None
                print(f'chunk={completed}: {actions.shape}; first left/right xyz={actions[0,[0,1,2,10,11,12]]}', flush=True)
                if args.execute:
                    if args.use_temporal_smoothing:
                        buffer.integrate_new_chunk(actions, max_k=args.latency_k, min_m=args.min_smooth_steps)
                    else:
                        buffer.cur_chunk = deque(actions)
                        buffer.k, buffer.last_action = 0, None
                elif args.steps and completed >= args.steps:
                    break
            if args.execute:
                action = buffer.pop_next_action()
                if action is not None:
                    target = frames.targets(action)
                    measured = frames.targets(observation[1])
                    guard_targets(measured, target, cfg)
                    if previous is not None:
                        guard_targets(previous, target, cfg)
                    for i, publisher in enumerate(publishers):
                        msg = PosCmd()
                        msg.x, msg.y, msg.z = target[i, :3]
                        msg.roll, msg.pitch, msg.yaw = target[i, 3:6]
                        msg.gripper = target[i, 6]
                        publisher.publish(msg)
                    previous = target
                elif args.steps and completed >= args.steps:
                    break
            if (future is None and (not args.steps or completed < args.steps) and
                    time.monotonic()-last_request >= 1/args.inference_rate):
                future = pool.submit(policy.infer, observation[0], observation[1], args.prompt)
                last_request = time.monotonic()
            rate.sleep()


if __name__ == '__main__':
    main()
