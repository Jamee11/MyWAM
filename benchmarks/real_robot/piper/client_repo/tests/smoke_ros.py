"""Private ROS master + synthetic observations + HTTP hold policy; never uses a real driver."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:11411'
os.environ['ROS_LOG_DIR'] = str(ROOT/'logs/smoke_ros')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


class HoldPolicy(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        data = BytesParser(policy=default).parsebytes(
            ('Content-Type: '+self.headers['Content-Type']+'\r\nMIME-Version: 1.0\r\n\r\n').encode()+body)
        parts = {p.get_param('name',header='content-disposition'):p.get_payload(decode=True) for p in data.iter_parts()}
        assert set(parts) == {'json','front','left','right'}
        query = json.loads(parts['json'])
        assert len(query['state']) == 20
        response = json.dumps(dict(schema=query['schema'],actions=[query['state']]*32)).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main():
    # Refuse to share an existing master, even if someone happens to use this test port.
    try:
        xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/openwam_probe')
    except OSError:
        pass
    else:
        raise RuntimeError('Test port 11411 already has a ROS master')
    log = ROOT/'logs/smoke_ros'
    log.mkdir(parents=True,exist_ok=True)
    with (log/'roscore.log').open('w') as out:
        master = subprocess.Popen(['roscore','-p','11411'],stdout=out,stderr=subprocess.STDOUT)
        server = ThreadingHTTPServer(('127.0.0.1',0),HoldPolicy)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        client = None
        try:
            for _ in range(100):
                try:
                    xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/openwam_probe')
                    break
                except OSError:
                    time.sleep(.1)
            import rospy
            from sensor_msgs.msg import Image, JointState
            from geometry_msgs.msg import PoseStamped
            rospy.init_node('openwam_synthetic_observations',anonymous=True,disable_signals=True)
            config=json.loads((ROOT/'robot.json').read_text())
            keys=['front','left_image','right_image','left_pose','right_pose','left_state','right_state']
            kinds=[Image]*3+[PoseStamped]*2+[JointState]*2
            publishers=[rospy.Publisher(config['topics'][k],t,queue_size=1) for k,t in zip(keys,kinds)]
            env=dict(os.environ,OPENWAM_API_TOKEN='test-token-0123456789')
            with (log/'client.log').open('w') as output:
                client=subprocess.Popen([sys.executable,str(ROOT/'inference/agilex_inference_openwam.py'),
                                         '--port',str(server.server_port),'--prompt','hold both arms','--steps','2'],
                                        env=env,stdout=output,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+25
                while client.poll() is None and time.monotonic()<deadline:
                    for publisher,kind in zip(publishers,kinds):
                        message=kind(); message.header.stamp=rospy.Time.now()
                        if kind is Image:
                            message.height=16; message.width=16; message.encoding='rgb8'
                            message.step=48; message.data=bytes([80,120,160])*256
                        elif kind is PoseStamped:
                            message.pose.position.x=.2;message.pose.position.z=.2;message.pose.orientation.w=1
                        else:
                            message.position=[0.]*6+[.035]
                        publisher.publish(message)
                    time.sleep(.05)
                if client.poll() is None:
                    raise RuntimeError('Client smoke test timed out')
                assert client.returncode == 0, (log/'client.log').read_text()
            publishers_state=xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getSystemState('/openwam_probe')[2][0]
            assert not any(name.startswith(('/control/','/master/')) for name,_ in publishers_state)
            assert 'chunk=2:' in (log/'client.log').read_text()
            print('PASS: real ROS1 client, 2 HTTP requests, EEF20/three images; 0 control publishers')
        finally:
            if client is not None and client.poll() is None:
                client.terminate();client.wait(timeout=10)
            server.shutdown();server.server_close()
            master.terminate();master.wait(timeout=10)


if __name__=='__main__':
    main()
