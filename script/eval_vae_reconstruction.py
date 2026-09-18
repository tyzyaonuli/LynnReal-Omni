"""Deterministic, streaming video autoencoder round trips; never loads a DiT."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_h3_vae_ab import sha256, write_json

TAE_SHA = 'af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba'


def padded_length(n):
    # Native encode drops three tokens; decode returns 17*k - 12 frames.
    return max(34, math.ceil((n + 12) / 17) * 17)


def geometry(w, h, short_edge):
    scale = min(1, short_edge / min(w, h)) if short_edge else 1
    # Resize once, then replicate-pad spatially instead of cropping/stretching.
    return max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)


def read_frames(path, width, height):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for frame in container.decode(stream):
            yield frame.reformat(width=width, height=height, format='rgb24').to_ndarray()


def ssim_rgb(prediction, reference, device='cuda'):
    """Per-frame RGB SSIM, 11x11 uniform valid windows, population covariance."""
    import torch
    import torch.nn.functional as F
    x = prediction.permute(0, 3, 1, 2).to(device, torch.float32)
    y = torch.from_numpy(reference).permute(0, 3, 1, 2).to(device, torch.float32) / 255
    pool = lambda z: F.avg_pool2d(z, 11, stride=1)
    ux, uy = pool(x), pool(y)
    vx, vy, cov = pool(x*x)-ux*ux, pool(y*y)-uy*uy, pool(x*y)-ux*uy
    score = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return score.mean((1,2,3)).cpu().tolist()


class Writer:
    """Identical browser encodes, preserving each source presentation timestamp."""
    def __init__(self, path, case):
        import av
        self.av = av
        self.container = av.open(str(path), 'w', options={'movflags': '+faststart'})
        self.stream = self.container.add_stream('libx264', rate=Fraction(case['fps']))
        self.stream.width, self.stream.height = case['width'], case['height']
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '16', 'preset': 'fast'}
        self.tb = Fraction(case['time_base'])
        self.stream.time_base = self.tb
        self.pts = case['pts']
        self.index = 0

    def write(self, pixels):
        for rgb in pixels:
            frame = self.av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts = self.pts[self.index] - self.pts[0]
            frame.time_base = self.tb
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.index += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def encode_blocks(model, variant, frames, case, measure):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from model.third_party.taehv import apply_model_with_memblocks
    total = padded_length(case['frames'])
    latents, last, count = [], None, 0
    digest = hashlib.sha256()
    for offset in range(0, total, 17):
        batch = []
        for _ in range(17):
            if count < case['frames']:
                last = next(frames)
                digest.update(last.tobytes())
                count += 1
            batch.append(last)
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).unsqueeze(0).to('cuda', torch.float32) / 255
        x = F.pad(x.flatten(0, 1), (0, -case['width'] % 16, 0, -case['height'] % 16), mode='replicate').unflatten(0, (1, 17))
        def encode():
            with torch.autocast('cuda', dtype=torch.float16):
                if variant == 'tae':
                    prepared = model.preprocess_input_frames(F.pad(x, (0, 0, 0, 0, 0, 0, 3, 0)))
                    return apply_model_with_memblocks(model.encoder, prepared, False, False).permute(0, 2, 1, 3, 4)
                mean = x.new_tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
                std = x.new_tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
                core = model.core if variant == 'light' else model
                moments = core._encode_clip(((x - mean) / std).permute(0, 2, 1, 3, 4))
                return moments.chunk(2, dim=1)[0]  # posterior.mode(), never sample
        z = measure('encoder', encode)
        assert z.shape[2] == 5 and torch.isfinite(z).all()
        latents.append(z.cpu())
        if offset % 1700 == 0:
            print('encoded', variant, case['id'], min(offset + 17, case['frames']), '/', case['frames'], flush=True)
    assert count == case['frames'] and next(frames, None) is None
    return torch.cat(latents, dim=2)[:, :, :-3].contiguous(), digest.hexdigest()


def decode_blocks(model, variant, z, measure):
    """Exact native temporal protocol, moving completed RGB chunks off device."""
    import torch
    if variant == 'tae':
        from model.third_party.taehv import apply_model_with_memblocks_sequential_single_step, TWorkItem
        memory = [None] * len(model.decoder)
        pending = []
        for t in range(z.shape[2]):
            token = z[:, :, t].to('cuda')
            def step():
                queue, out = [TWorkItem(token, 0)], []
                with torch.autocast('cuda', dtype=torch.float16):
                    while queue:
                        frame = apply_model_with_memblocks_sequential_single_step(model.decoder, memory, queue)
                        if frame is not None:
                            out.append(frame)
                return torch.cat(out, dim=1)
            pending.append(measure('decoder', step))
            if (t + 1) % 5 == 0 or t + 1 == z.shape[2]:
                raw = torch.cat(pending, dim=1)
                # Full groups: 20->17; final two tokens: 8->5. This is the
                # native zero-pad/trim/tail-drop, without materializing zeros.
                rgb = model.postprocess_output_frames(raw[:, 3:]).float()
                yield rgb[0].permute(0, 2, 3, 1).cpu()
                pending = []
        return
    core = model.core if variant == 'light' else model
    assert z.shape[2] % 5 == 2
    overlap = None
    for start in range(0, z.shape[2] - 2, 5):
        latent = z[:, :, start:start + 7].to('cuda', dtype=torch.float16)
        def step():
            with torch.autocast('cuda', dtype=torch.float16):
                return model._decode_clip(latent)
        clip = measure('decoder', step)
        chunk = clip[:, :, 3:20]
        if overlap is not None:
            chunk = core._blend(overlap, chunk, core.frame_overlap, dim=-3)
        overlap = clip[:, :, 23:]
        chunk = chunk.float()
        mean = chunk.new_tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
        std = chunk.new_tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
        yield (chunk * std + mean).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu()
    yield (overlap.float() * std + mean).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--h3', type=Path)
    p.add_argument('--light-vae', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--case')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    cases = [c for c in manifest['cases'] if not a.case or c['id'] == a.case]
    assert cases and len({c['id'] for c in cases}) == len(cases)
    for c in cases:
        assert c['frames'] == len(c['pts']) > 0
        assert all(x < y for x, y in zip(c['pts'], c['pts'][1:]))
        assert padded_length(c['frames']) - 12 >= c['frames']
    if a.dry_run:
        print(json.dumps({'cases': len(cases), 'frames': sum(c['frames'] for c in cases), 'dit_nfe': 0, 'gpu_count': 1}))
        return
    import numpy as np
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from model.light_vae import LightVAE
    from model.third_party.taehv import TAEHV
    assert torch.cuda.device_count() == 1
    assert sha256(a.checkpoint) == TAE_SHA
    torch.set_grad_enabled(False)
    a.output.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': sha256(a.manifest), 'runner_sha256': sha256(Path(__file__)),
                'light_wrapper_sha256': sha256(ROOT / 'model/light_vae.py'),
                'tae_implementation_sha256': sha256(ROOT / 'model/third_party/taehv.py'),
                'vae_weights': 'fp32', 'autocast': 'fp16', 'dit_nfe': 0,
                'posterior': 'mode', 'protocol': 'native-streaming-v1'}
    for key,path,revision in [('h3',a.h3,manifest['h3_revision']),('light',a.light_vae,manifest['light_revision'])]:
        marker_path=path/'H3_VAE_AB_READY.json'
        marker=json.loads(marker_path.read_text())
        assert marker['complete'] and marker['revision']==revision
        for entry in marker['files']:
            assert entry['verified'] and (path/entry['path']).stat().st_size==entry['size']
        identity[key+'_weights_marker_sha256']=sha256(marker_path)
        write_json(a.output/(key+'-weights.json'),marker)
    old = a.output / 'identity.json'
    if old.exists():
        assert json.loads(old.read_text()) == identity, 'resume identity mismatch'
    write_json(old, identity)
    write_json(a.output / 'manifest.json', manifest)
    write_json(a.output / 'run.json', dict(identity, job_id=os.environ.get('DLC_JOB_ID'),
               code_sha=os.environ.get('LYNNREAL_RELEASE_SHA'), gpu_count=1,
               gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
               torch=torch.__version__, audio='silent visual comparison; source audio unchanged at source URL'))
    for c in cases:
        assert sha256(a.sources / (c['id'] + '.mp4')) == c['sha256']
    for variant in ('baseline', 'light', 'tae'):
        def complete(c):
            folder = a.output / c['id']
            try:
                r = json.loads((folder / (variant + '.json')).read_text())
                return r['identity'] == identity and r['status'] == 'complete' and sha256(folder / (variant + '-web.mp4')) == r['preview_sha256']
            except (OSError, ValueError, KeyError):
                return False
        pending = [c for c in cases if not complete(c)]
        if not pending:
            continue
        started = time.perf_counter()
        if variant == 'baseline':
            model, loading = AutoencoderKLMiniMaxH3.from_pretrained(str(a.h3 / 'vae'), torch_dtype=torch.float32, local_files_only=True, output_loading_info=True)
            assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
        elif variant == 'light':
            light_core, loading = AutoencoderKLMiniMaxH3.from_pretrained(str(a.light_vae), torch_dtype=torch.float32, local_files_only=True, output_loading_info=True)
            assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
            model = LightVAE(light_core, json.loads((a.light_vae/'decode_config.json').read_text()), tile_batch=1, tile_layout='native')
            del light_core
        else:
            model = TAEHV(str(a.checkpoint), arch_name='taeh3')
        model.eval().requires_grad_(False).to('cuda', dtype=torch.float32)
        core = model.core if variant == 'light' else model
        if variant != 'tae':
            core.set_attention_backend('native')
            assert core.use_tiling and core.tile_sample_min_height == 256
        encoder_hash = hashlib.sha256()
        for name, value in core.encoder.state_dict().items():
            encoder_hash.update(name.encode())
            encoder_hash.update(value.cpu().contiguous().numpy().tobytes())
        write_json(a.output / (variant + '-model.json'), {'load_seconds': time.perf_counter()-started,
                   'encoder_sha256': encoder_hash.hexdigest(), 'checkpoint_sha256': TAE_SHA if variant == 'tae' else None})
        # Validate the streaming implementation against the native full methods
        # on 51 input frames (39 reconstructed), before any production case.
        smoke = np.stack(list(__import__('itertools').islice(read_frames(a.sources / (pending[0]['id']+'.mp4'), 64, 64), 51)))
        if len(smoke)<51:
            smoke=np.concatenate([smoke,np.repeat(smoke[-1:],51-len(smoke),axis=0)])
        smoke_case = dict(pending[0], frames=39, width=64, height=64)
        stream_z, _ = encode_blocks(model, variant, iter(smoke[:39]), smoke_case, lambda _, fn: fn())
        smoke[39:] = smoke[38]
        x = torch.from_numpy(smoke).permute(0, 3, 1, 2).unsqueeze(0).to('cuda', torch.float32) / 255
        with torch.autocast('cuda', dtype=torch.float16):
            if variant == 'tae':
                native_z = model.encode_video(x, parallel=False, show_progress_bar=False)
                native_rgb = model.decode_video(native_z, parallel=False, show_progress_bar=False)[0].permute(0, 2, 3, 1)
                native_z = native_z.permute(0, 2, 1, 3, 4)
            else:
                mean = x.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
                std = x.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
                native_z = model.encode((x.permute(0,2,1,3,4)-mean)/std).latent_dist.mode()
                native_rgb = (model.decode(native_z.to(torch.float16)).sample.float()*std+mean).clamp(0,1)[0].permute(1,2,3,0)
        actual_rgb = torch.cat(list(decode_blocks(model, variant, stream_z, lambda _, fn: fn())))
        z_error = (stream_z.float()-native_z.float().cpu()).abs().max().item()
        rgb_error = (actual_rgb-native_rgb.float().cpu()).abs().max().item()
        assert z_error < .002 and rgb_error < .002, (variant, z_error, rgb_error)
        write_json(a.output / (variant+'-stream-equivalence.json'), {'passed': True, 'input_frames':51, 'output_frames':39, 'size':64, 'latent_max_abs_error':z_error, 'rgb_max_abs_error':rgb_error})
        del x, native_z, native_rgb, actual_rgb, stream_z
        for c in pending:
            folder = a.output / c['id']; folder.mkdir(parents=True, exist_ok=True)
            source = a.sources / (c['id'] + '.mp4')
            timings = {}
            def measure(name, fn):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated(); t = time.perf_counter()
                result = fn(); torch.cuda.synchronize()
                m = timings.setdefault(name, {'wall_seconds':0, 'calls':0, 'peak_allocated_bytes':0, 'peak_reserved_bytes':0, 'incremental_peak_allocated_bytes':0})
                m['wall_seconds'] += time.perf_counter()-t; m['calls'] += 1
                for key, value in [('peak_allocated_bytes',torch.cuda.max_memory_allocated()), ('peak_reserved_bytes',torch.cuda.max_memory_reserved()), ('incremental_peak_allocated_bytes',torch.cuda.max_memory_allocated()-before)]:
                    m[key] = max(m[key], value)
                return result
            started = time.perf_counter()
            z, input_hash = encode_blocks(model, variant, iter(read_frames(source,c['width'],c['height'])), c, measure)
            for other in ('baseline','light','tae'):
                previous = folder/(other+'.json')
                if previous.exists():
                    assert json.loads(previous.read_text())['input_rgb_sha256'] == input_hash
            z_path = folder / (variant+'-latents.pt'); torch.save(z, z_path)
            preview = folder / (variant+'-web.mp4')
            writer = Writer(preview,c)
            reference_path = folder/'original-web.mp4'
            ref_writer = Writer(reference_path,c) if variant == 'baseline' or not reference_path.exists() else None
            reference = iter(read_frames(source,c['width'],c['height']))
            squared_error, samples, frame_metrics, frames = 0., 0, [], 0
            recon_hash = hashlib.sha256()
            try:
                for rgb in decode_blocks(model, variant, z, measure):
                    rgb = rgb[:c['frames']-frames, :c['height'], :c['width']]
                    if not len(rgb): break
                    assert torch.isfinite(rgb).all()
                    original = np.stack([next(reference) for _ in range(len(rgb))])
                    pred = rgb.numpy()
                    ssim = ssim_rgb(rgb, original)
                    for idx in range(len(pred)):
                        mse = float(np.mean((pred[idx]-original[idx].astype(np.float32)/255)**2, dtype=np.float64))
                        squared_error += mse * pred[idx].size; samples += pred[idx].size
                        frame_metrics.append({'frame':frames+idx,'mse_rgb':mse,'psnr_rgb_db':-10*math.log10(max(mse,1e-20)), 'ssim_rgb':ssim[idx]})
                    pixels = (pred*255).round().astype(np.uint8)
                    recon_hash.update(pixels.tobytes()); writer.write(pixels)
                    if ref_writer: ref_writer.write(original)
                    frames += len(pred)
                    if frames % 1700 == 0: print('decoded',variant,c['id'],frames,'/',c['frames'],flush=True)
            finally:
                writer.close()
                if ref_writer: ref_writer.close()
            assert frames == c['frames'] and next(reference,None) is None
            write_json(folder/(variant+'-frame-metrics.json'),frame_metrics)
            record = dict(identity=identity, status='complete',variant=variant,case_id=c['id'],
                          input_rgb_sha256=input_hash,output_rgb_sha256=recon_hash.hexdigest(),
                          source_sha256=c['sha256'],preview_sha256=sha256(preview),latent_sha256=sha256(z_path),
                          frames=frames,width=c['width'],height=c['height'],fps=c['fps'],
                          temporal_input_padding=padded_length(frames)-frames,spatial_padding=[-c['width']%16,-c['height']%16],
                          encoder=timings['encoder'],decoder=timings['decoder'],
                          wall_seconds=time.perf_counter()-started,psnr_rgb_db=-10*math.log10(max(squared_error/samples,1e-20)),
                          ssim_rgb=sum(f['ssim_rgb'] for f in frame_metrics)/frames,
                          ssim_definition='11x11 uniform valid-window RGB, population covariance, C1=.01^2 C2=.03^2',
                          metrics_domain='float RGB [0,1] before preview encoding versus common resized source')
            write_json(folder/(variant+'.json'),record)
            print('RECON_CASE_COMPLETE',variant,c['id'],json.dumps(record),flush=True)
            del z; gc.collect(); torch.cuda.empty_cache()
        del model,core; gc.collect(); torch.cuda.empty_cache()
    write_json(a.output/'summary.json', {'complete':True,'cases':len(cases),'variants':3,'dit_nfe':0})
    print('VAE_RECONSTRUCTION_PASS',flush=True)


if __name__ == '__main__':
    main()
