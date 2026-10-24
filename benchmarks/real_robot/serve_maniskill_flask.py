"""Flask transport over the existing checkpoint loader and chunk inference engine."""
import argparse
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.real_robot.openwam_real_contract import CONTRACT, action_chunk, eef_to_matrix


def create_app(infer, metadata, token, record_dir=None):
    from flask import Flask, jsonify, request
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    if len(token) < 16:
        raise ValueError('Use an API token of at least16 characters')
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
    lock = threading.Lock()
    if record_dir is not None:
        record_dir = Path(record_dir)
        record_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    @app.before_request
    def authorize():
        if not hmac.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + token):
            return jsonify(error='Unauthorized'), 401

    @app.get('/health')
    def health():
        return jsonify(status='ready', contract=CONTRACT, **metadata)

    @app.post('/api/inference')
    def inference():
        if not lock.acquire(blocking=False):
            return jsonify(error='Model busy; do not queue stale robot observations'), 409
        try:
            if set(request.files) != {'json', 'main_images', 'wrist_images'}:
                raise ValueError('Send multipart json, main_images and wrist_images')
            raw_query = request.files['json'].read()
            query = json.loads(raw_query)
            required = {'task_description', 'state', 'exp_id', 'request_id', 'schema', 'gripper_state_source'}
            if not isinstance(query, dict) or set(query) != required or query['schema'] != CONTRACT['schema']:
                raise ValueError('Wrong request schema; legacy delta-EE requests are not supported')
            for key in ('task_description', 'exp_id', 'request_id'):
                if not isinstance(query[key], str) or not query[key].strip() or len(query[key]) > 2048:
                    raise ValueError(f'Invalid {key}')
            if query['gripper_state_source'] not in ('measured', 'command_estimate'):
                raise ValueError('Declare whether gripper proprio is measured or estimated')
            state = np.asarray(query['state'], dtype=np.float32)
            eef_to_matrix(state)
            if not 0 <= state[9] <= 1:
                raise ValueError('Proprio gripper must be in[0,1]')
            images = []
            raw_images = []
            image_suffixes = []
            for key in ('main_images', 'wrist_images'):
                raw_image = request.files[key].read()
                with Image.open(io.BytesIO(raw_image)) as im:
                    if im.width * im.height > 8000000:
                        raise ValueError('Image too large')
                    image_suffixes.append('.jpg' if im.format == 'JPEG' else '.png' if im.format == 'PNG' else '.img')
                    images.append(im.convert('RGB'))
                raw_images.append(raw_image)
            if record_dir is not None:
                case_dir = record_dir / (time.strftime('%Y%m%dT%H%M%SZ_', time.gmtime()) + uuid.uuid4().hex)
                case_dir.mkdir(mode=0o700)
                (case_dir / 'request.json').write_bytes(raw_query)
                for name, raw_image, suffix in zip(('front', 'wrist'), raw_images, image_suffixes):
                    (case_dir / f'{name}{suffix}').write_bytes(raw_image)
            start = time.monotonic()
            actions = action_chunk(infer(images[0], images[1], state, query['task_description']))
            response = jsonify(actions=actions.tolist(), contract=CONTRACT, **metadata,
                           request_id=query['request_id'], exp_id=query['exp_id'],
                           gripper_state_source=query['gripper_state_source'],
                           latency_ms=1000*(time.monotonic()-start),
                           gripper_out_of_range=int(np.sum((actions[:,9]<0)|(actions[:,9]>1))))
            if record_dir is not None:
                (case_dir / 'response.json').write_bytes(response.get_data())
            return response
        except (ValueError, TypeError, KeyError, UnidentifiedImageError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception('Inference failed; no action response returned')
            return jsonify(error='Inference failed; do not execute or replay old actions'), 500
        finally:
            lock.release()

    return app


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--policy', choices=('sidecar', 'baseline'), default='sidecar')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5901)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--min-free-gib', type=float, default=35)
    p.add_argument('--max-memory-gib', type=float, default=34)
    args = p.parse_args()
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.server import build_server_from_config
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference

    token = args.token_file.read_text().strip()
    if len(token) < 16:
        p.error('Token must be at least16 characters')
    checkpoint = args.checkpoint.resolve(strict=True)
    cfg = OmegaConf.load(checkpoint.parent/'config.yaml')
    expected_type = 'maniskill_sidecar' if args.policy == 'sidecar' else 'maniskill_absolute'
    if (cfg.dataloader.type != expected_type or cfg.dataloader.action_mode != 'eef'
            or list(cfg.dataloader.unify_action_map) != ['0-9']
            or list(cfg.dataloader.camera_layout) != ['front_camera', 'hand_camera', None]):
        raise ValueError('Checkpoint is not the expected single-arm absolute ManiSkill EEF10 contract')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one selected GPU with CUDA_VISIBLE_DEVICES')
    free, total = torch.cuda.mem_get_info(0)
    if free < args.min_free_gib * 2**30:
        raise RuntimeError(f'Insufficient free memory: {free/2**30:.1f}GiB; do not stop unrelated jobs')
    cap = min(args.max_memory_gib*2**30, free-4*2**30)
    if cap <= 0:
        raise ValueError('Invalid allocator memory budget')
    torch.cuda.set_per_process_memory_fraction(cap/total, device=0)
    deploy = OmegaConf.load(ROOT/'configs/deploy.yaml')
    deploy.optimization.compile.enabled = False
    deploy.optimization.dit_cache.enabled = False
    deploy.optimization.decode_video = False
    OmegaConf.update(deploy, 'inference.cfg_scale', 1.)
    deploy.inference.inference_horizon = 32
    model = build_server_from_config(deploy, str(checkpoint.parent), device='cpu', ckpt_name=checkpoint.name)
    # Recheck after slow CPU loading; do not consume another job's newly allocated memory.
    if torch.cuda.mem_get_info(0)[0] < args.min_free_gib*2**30:
        raise RuntimeError('GPU capacity changed while loading; restart when space is available')
    model.engine.architecture.set_dtype_device(model.engine.architecture.dtype, torch.device('cuda:0'))
    model._init_policy()
    arch = model.engine.architecture
    if bool(getattr(arch, 'sidecar_enabled', False)) != (args.policy == 'sidecar'):
        raise ValueError('Loaded architecture does not match selected policy')
    metadata = dict(policy=args.policy, checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint),
                    normalization_sha256=sha256(checkpoint.parent/'normalization_stats.npy'),
                    pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    denoise_steps=10, cfg_scale=1, compile=False, dit_cache=False)

    def infer(front, wrist, state, task):
        model.reset()
        payload = dict(images=dict(head_camera=front, left_wrist_camera=wrist, right_wrist_camera=None),
                       state=state, prompt=format_prompt_for_inference(task))
        obs = model._obs_preprocessor.preprocess(payload)
        with torch.inference_mode():
            return model.engine.generate(model._policy._build_conditions(obs))['actions']

    app = create_app(infer, metadata, token, args.manifest.parent/'inference_records')
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(dict(status='model_loaded', contract=CONTRACT, **metadata,
        host=args.host, port=args.port, allocator_cap_gib=cap/2**30), indent=2))
    logging.info('MODEL_LOADED %s; no robot connection or motion is performed here', args.policy)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
