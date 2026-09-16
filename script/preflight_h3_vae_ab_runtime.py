import hashlib,json,os,time,torch
from pathlib import Path
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
from diffusers import MiniMaxH3Scheduler
from model.light_vae import LightVAE
root=Path('/cpfs/world-model/lynnreal-omni')
archive=root/'cache/env/py312-torch2121-cu130-lynnreal-0384eca3d5d7482d.tar.gz'
h=hashlib.sha256()
with archive.open('rb') as stream:
 for b in iter(lambda:stream.read(16<<20),b''):h.update(b)
model=root/'models/h3-bfc8ed0353f5a9733be73e6b2c98ec0948195b86'
pipe=MiniMaxH3Blocks().init_pipeline(str(model))
assert pipe._component_specs['transformer'].subfolder=='transformer'
m=json.loads(Path('eval/h3_vae_ab/manifest.json').read_text())
for name,shift in [('video',12),('audio',3)]:
 s=MiniMaxH3Scheduler(shift=shift);s.set_timesteps(6)
 assert s.sigmas.tolist()==m['sampling']['sigma_schedule_fp32'][name]
from eval_h3_vae_ab import light_weight_view
vae=LightVAE.from_pretrained(light_weight_view(root/'models/light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a'),tile_layout='native')
result={'status':'passed','gpu_used':False,'torch':torch.__version__,'archive':str(archive),'archive_sha256':h.hexdigest(),'archive_bytes':archive.stat().st_size,'transformer_subfolder':pipe._component_specs['transformer'].subfolder,'light_parameters':sum(p.numel() for p in vae.parameters()),'tiles':vae._geometry(37,768,1344),'sigma_schedule_verified':True}
Path(os.environ['PROBE_RESULT']).write_text(json.dumps(result,indent=2))
marker=archive.parent/'h3-vae-ab-env-ready.json';tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,marker)
print(json.dumps(result),flush=True)
