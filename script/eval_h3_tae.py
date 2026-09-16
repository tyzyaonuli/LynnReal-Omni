"""Decode pinned H3 latents with TAE only, without resampling or changing A/B."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_h3_vae_ab import sha256, write_json, resolve

WEIGHT_SHA = 'af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba'
REVISION = '011dfc2112197741c540e0bdd5b7b67bcc930771'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--case')
    a = p.parse_args()
    assert a.source.resolve() != a.output.resolve()
    assert sha256(a.checkpoint) == WEIGHT_SHA
    manifest_path = ROOT / 'eval/h3_vae_ab/manifest.json'
    manifest, cases = resolve(manifest_path, a.case)
    source_run = json.loads((a.source / 'run.json').read_text())
    assert source_run['manifest_sha256'] == sha256(manifest_path)
    assert source_run['gpu_count'] == 1 and source_run['precision']['dit'] == 'bf16'
    a.output.mkdir(parents=True, exist_ok=True)
    import torch
    import av
    import imageio_ffmpeg
    from PIL import Image
    from model.third_party.taehv import TAEHV
    from model.output import encode_video
    from diffusers.modular_pipelines.minimax_h3.packing import unpatchify_video_tokens
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability()[0] >= 10
    torch.set_grad_enabled(False)
    timings = {}
    def measure(name, fn):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        samples = [torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]]
        stop = threading.Event()
        def sample():
            while not stop.wait(.1):
                free, total = torch.cuda.mem_get_info(); samples.append(total-free)
        thread = threading.Thread(target=sample, daemon=True); thread.start()
        started = time.perf_counter()
        try:
            result = fn(); torch.cuda.synchronize()
        finally:
            stop.set(); thread.join()
        m = {'wall_seconds': time.perf_counter()-started, 'allocated_before_bytes': before,
             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
             'incremental_peak_allocated_bytes': torch.cuda.max_memory_allocated()-before,
             'device_used_sampled_peak_bytes': max(samples), 'memory_sample_interval_seconds': .1}
        timings[name] = m
        write_json(a.output/'timings.json', timings)
        print(name, json.dumps(m), flush=True)
        return result
    def load():
        model = TAEHV(str(a.checkpoint), arch_name='taeh3').eval()
        # Only decoder parameters are needed; retain FP32 weights and FP16 autocast.
        del model.encoder
        return model.to('cuda', dtype=torch.float32)
    model = measure('tae_model_load', load)
    provenance = {'job_id': os.environ.get('DLC_JOB_ID'), 'code_sha': os.environ.get('LYNNREAL_RELEASE_SHA'),
                  'source_job': source_run['job_id'], 'manifest_sha256': sha256(manifest_path),
                  'taehv_revision': REVISION, 'checkpoint_sha256': WEIGHT_SHA,
                  'source_sha256': sha256(ROOT/'model/third_party/taehv.py'),
                  'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                  'torch': torch.__version__, 'gpu_count': 1, 'weights_dtype': 'float32',
                  'decode_autocast': 'float16', 'parallel': False, 'spatial_tiling': False,
                  'latent_normalization': 'diffusion latents directly; no VAE mean/std reversal',
                  'sampling_reused': True, 'audio_reused': True}
    write_json(a.output/'run.json', provenance)
    work = [{'id': '_warmup'}, *cases]
    for case in work:
        folder = a.source/case['id']
        original = json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
        latent_hash, audio_hash = sha256(folder/'latents.pt'), sha256(folder/'audio.pt')
        assert latent_hash == original['latent_sha256'] and audio_hash == original['audio_sha256']
        if case['id'] != '_warmup': assert original['case'] == case
        state = torch.load(folder/'latents.pt', map_location='cuda', weights_only=True)
        z = unpatchify_video_tokens(state['latents'], state['num_latent_frames'],
                                   state['latent_height'], state['latent_width'], 24, (1,2,2))
        z = z.permute(0,2,1,3,4).contiguous()
        assert z.shape[2] == 24 and torch.isfinite(z).all()
        def decode():
            with torch.autocast('cuda', dtype=torch.float16):
                return model.decode_video(z, parallel=False, show_progress_bar=False)
        rgb = measure(case['id']+'/tae_decoder', decode)
        assert rgb.shape == (1,124,3,768,1344), tuple(rgb.shape)
        assert torch.isfinite(rgb).all()
        if case['id'] == '_warmup':
            del rgb,z,state; gc.collect(); torch.cuda.empty_cache(); continue
        output = a.output/case['id']; output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        pixels = rgb.float().clamp(0,1).mul(255).round_().to(torch.uint8).permute(0,1,3,4,2).cpu().numpy()[0]
        frames = [Image.fromarray(frame) for frame in pixels]
        post = time.perf_counter()-started
        audio = torch.load(folder/'audio.pt', map_location='cpu', weights_only=True)
        started = time.perf_counter()
        video = output/'tae.mp4'; preview = output/'tae-web.mp4'
        encode_video(frames,24,video,audio=audio['audio'][0],audio_sample_rate=audio['sampling_rate'])
        encode_seconds = time.perf_counter()-started
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(video),'-c:v','libx264',
                        '-pix_fmt','yuv420p','-crf','16','-movflags','+faststart','-c:a','copy',str(preview)],check=True)
        digest = hashlib.sha256()
        with av.open(str(video)) as container:
            for frame in container.decode(audio=0): digest.update(frame.to_ndarray().tobytes())
        assert digest.hexdigest() == original['decoded_audio_sha256']
        write_json(output/'tae.json',dict(provenance,case=case,variant='tae',status='complete',
                   latent_sha256=latent_hash,audio_sha256=audio_hash,decoded_audio_sha256=digest.hexdigest(),
                   output_sha256=sha256(video),preview_sha256=sha256(preview),video='tae.mp4',
                   decoder=timings[case['id']+'/tae_decoder'],postprocess_seconds=post,encode_seconds=encode_seconds,
                   frames=124,width=1344,height=768,fps=24))
        del rgb,z,state,pixels,frames,audio; gc.collect(); torch.cuda.empty_cache()
    write_json(a.output/'summary.json',{'complete':True,'cases':len(cases),'source_job':source_run['job_id'],
               'tae_decode_mean_seconds':sum(timings[c['id']+'/tae_decoder']['wall_seconds'] for c in cases)/len(cases)})
    print('H3_TAE_ONLY_PASS',flush=True)


if __name__ == '__main__':
    main()
