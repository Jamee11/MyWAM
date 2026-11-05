"""Piper EEF20 Flask adapter; does not connect to or move hardware."""
import argparse
import hmac
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.real_robot.piper.client_repo.inference.contract import SCHEMA, state20, action_chunk


def create_app(infer, token, metadata, record_dir):
    from flask import Flask, jsonify, request
    from PIL import Image
    if len(token) < 16:
        raise ValueError('Use a token of at least 16 characters')
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    lock = threading.Lock()
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @app.before_request
    def auth():
        if not hmac.compare_digest(request.headers.get('Authorization', ''), 'Bearer '+token):
            return jsonify(error='Unauthorized'), 401

    @app.get('/health')
    def health():
        return jsonify(schema=SCHEMA, horizon=32, action_dim=20, **metadata)

    @app.post('/api/inference')
    def predict():
        if not lock.acquire(blocking=False):
            return jsonify(error='Model busy'), 409
        try:
            if set(request.files) != {'json', 'front', 'left', 'right'}:
                raise ValueError('Send multipart json, front, left, right')
            raw = request.files['json'].read()
            query = json.loads(raw)
            if query.get('schema') != SCHEMA or not isinstance(query.get('prompt'), str) or not query['prompt'].strip():
                raise ValueError('Expected EEF20 schema and nonempty prompt')
            state = state20(query['state'])
            images, raw_images = [], []
            for name in ('front', 'left', 'right'):
                content = request.files[name].read()
                with Image.open(io.BytesIO(content)) as image:
                    if image.width * image.height > 8000000:
                        raise ValueError('Image too large')
                    images.append(image.convert('RGB'))
                raw_images.append(content)
            folder = record_dir / uuid.uuid4().hex
            folder.mkdir(mode=0o700)
            (folder/'request.json').write_bytes(raw)
            for name, content in zip(('front', 'left', 'right'), raw_images):
                (folder/(name+'.png')).write_bytes(content)
            started = time.monotonic()
            actions = action_chunk(infer(images, state, query['prompt']))
            result = dict(schema=SCHEMA, actions=actions.tolist(), **metadata,
                          latency_ms=1000*(time.monotonic()-started))
            (folder/'response.json').write_text(json.dumps(result)+'\n')
            return jsonify(result)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception('Inference failed')
            return jsonify(error='Inference failed; do not replay old actions'), 500
        finally:
            lock.release()
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5902)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--record-dir', type=Path, default=ROOT/'outputs/real_robot/piper12000/inference_records')
    args = p.parse_args()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.server import build_server_from_config
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    from benchmarks.real_robot.serve_maniskill_flask import sha256

    checkpoint = args.checkpoint.resolve(strict=True)
    cfg = OmegaConf.load(checkpoint.parent/'config.yaml')
    contract = json.loads((checkpoint.parent/'relational_sidecar_contract.json').read_text())
    if (cfg.dataloader.type != 'piper_archive' or cfg.dataloader.action_mode != 'eef' or
            list(cfg.dataloader.unify_action_map) != ['0-9', '34-43'] or
            contract['embodiment'] != 'robotwin_piper' or contract['arm_present'] != [1, 1]):
        raise ValueError('Need a dual-arm absolute EEF20 Piper archive checkpoint')
    token = args.token_file.read_text().strip()
    if len(token) < 16:
        p.error('Token must be at least 16 characters')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose one free GPU using CUDA_VISIBLE_DEVICES')
    if torch.cuda.mem_get_info()[0] < 35 * 2**30:
        raise RuntimeError('Need at least 35GiB free; do not stop other jobs')
    deploy = OmegaConf.load(ROOT/'configs/deploy.yaml')
    deploy.optimization.compile.enabled = False
    deploy.optimization.dit_cache.enabled = False
    deploy.optimization.decode_video = False
    OmegaConf.update(deploy, 'inference.cfg_scale', 1.0)
    deploy.inference.inference_horizon = 32
    model = build_server_from_config(deploy, str(checkpoint.parent), device='cpu', ckpt_name=checkpoint.name)
    if torch.cuda.mem_get_info()[0] < 35 * 2**30:
        raise RuntimeError('GPU capacity changed during CPU loading')
    model.engine.architecture.set_dtype_device(model.engine.architecture.dtype, torch.device('cuda:0'))
    if not model.engine.architecture.sidecar_enabled:
        raise ValueError('Sidecar must be enabled')
    model._init_policy()
    # Archive configs lack multiview; explicitly match the training reader, not its single-view default.
    preprocess = ObsPreprocessor(multiview=True, camera_layout=['head_camera', 'left_camera', 'right_camera'],
                                img_height=384, img_width=320, requires_proprio=True)
    normalizer = model.engine.architecture.normalizer
    if not np.array_equal(normalizer._dst_index, list(range(10))+list(range(34,44))):
        raise ValueError('Unexpected 20D -> 80D action slots')

    def infer(images, state, prompt):
        model.reset()
        obs = preprocess.preprocess(dict(images=dict(head_camera=images[0], left_wrist_camera=images[1],
                                                     right_wrist_camera=images[2]),
                                         state=state, prompt=format_prompt_for_inference(prompt)))
        with torch.inference_mode():
            return model.engine.generate(model._policy._build_conditions(obs))['actions']

    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    normalization_sha256=sha256(checkpoint.parent/'normalization_stats.npy'), pid=os.getpid())
    app = create_app(infer, token, metadata, args.record_dir)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    main()
