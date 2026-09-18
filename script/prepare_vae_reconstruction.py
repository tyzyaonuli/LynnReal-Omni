"""Freeze local source hashes, geometry and original packet timestamps (CPU only)."""
import argparse
import json
from pathlib import Path
import av
from eval_h3_vae_ab import sha256, write_json
from eval_vae_reconstruction import geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--short-edge',type=int,default=768,help='0 preserves original resolution')
    p.add_argument('--storage-prefix',default='world-model/lynnreal-omni/inputs/vae-reconstruction-20260918',help='object prefix in the configured Thailand bucket')
    a=p.parse_args()
    m=json.loads(a.manifest.read_text(encoding='utf-8'))
    assert a.storage_prefix.startswith('world-model/lynnreal-omni/inputs/') and '..' not in a.storage_prefix.split('/')
    m.update(mode='vae-encode-decode',dit_nfe=0,short_edge=a.short_edge,storage_prefix=a.storage_prefix,
             resize='libswscale bilinear RGB24, preserve aspect ratio, even size; edge pad to16 then crop',
             temporal='full source duration; original presentation timestamps; repeat end for native17/5 protocol',
             h3_revision='bfc8ed0353f5a9733be73e6b2c98ec0948195b86',
             light_revision='e453444c1a52b73a0c5eb0023d208c473c2bb26a',
             taehv_revision='011dfc2112197741c540e0bdd5b7b67bcc930771')
    for c in m['cases']:
        path=a.sources/(c['id']+'.mp4')
        assert path.stat().st_size==c['size']
        c['sha256']=sha256(path)
        with av.open(str(path)) as container:
            s=container.streams.video[0]
            pts=sorted((packet.pts,packet.duration) for packet in container.demux(s) if packet.pts is not None)
            assert pts and len({p for p,_ in pts})==len(pts)
            assert not s.frames or s.frames==len(pts),(c['id'],s.frames,len(pts))
            w,h=geometry(s.width,s.height,a.short_edge)
            c.update(source_width=s.width,source_height=s.height,width=w,height=h,
                     fps=str(s.average_rate),time_base=str(s.time_base),pts=[p for p,_ in pts],
                     frames=len(pts),duration_seconds=float((pts[-1][0]+pts[-1][1]-pts[0][0])*s.time_base))
        print(c['id'],c['frames'],c['duration_seconds'],[w,h],flush=True)
    write_json(a.output,m)
    print('CPU_MANIFEST_READY',len(m['cases']),sum(c['duration_seconds'] for c in m['cases']),sha256(a.output))


if __name__=='__main__': main()
