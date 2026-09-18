"""Build a portable, randomized three-VAE reconstruction viewer plus reference."""
import argparse
import json
from pathlib import Path
from eval_h3_vae_ab import write_json


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>三种 VAE · 原视频重建</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10141d;color:#e6ebf5;font:15px system-ui,sans-serif}
main{max-width:1920px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 10px}p{color:#aebbd0;line-height:1.6}
button,select,input{font:inherit}button,select{background:#263248;color:#fff;border:1px solid #455571;border-radius:7px;padding:9px 13px;cursor:pointer}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.card{background:#1a2231;border:1px solid #35435c;border-radius:10px;padding:12px;min-width:0}.label{font-weight:650;min-height:26px}
video{width:100%;background:#000;aspect-ratio:16/9}.meta{color:#aebbd0;font-size:13px;line-height:1.6;min-height:65px}.hidden{visibility:hidden}
#seek{flex:1;min-width:200px}a{color:#93c5fd}#status{color:#ffce86}#case{max-width:560px}details{margin-top:22px;color:#aebbd0}summary{cursor:pointer}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){main{padding:12px}.grid{grid-template-columns:1fr}}
</style><main><h1>三种 VAE · 原视频重建</h1>
<p>原视频 → 各自 encoder → 各自 decoder。无 DiT，NFE = 0，确定性编码。左侧为统一预处理的原视频参照，后三路每组随机排列。先看画面，再揭晓模型名称。</p>
<div class="bar"><button id="prev">上一条</button><select id="case"></select><button id="next">下一条</button><button id="reveal">显示 VAE 名称与指标</button></div>
<div id="description"></div><div class="bar"><button id="play">同步播放</button><button id="pause">暂停</button><button id="back">−1 帧</button><button id="forward">+1 帧</button><input id="seek" type="range" min="0" max="1000" value="0"><span id="time">0.00 s</span></div>
<div id="status" role="status"></div><div class="grid" id="grid"></div>
<details><summary>实验配置与指标口径</summary><p>短边最多 768、不放大小视频；保留整段视频和原始时间戳。空间补边后裁回，尾帧重复补齐后裁回原帧数。FP32 权重、FP16 计算；H3 / Light 原生空间分块，TAE 原生顺序计算。音频未参与比较。</p>
<p>PSNR / SSIM 对统一预处理后的原视频计算，位于网页压缩之前。SSIM 使用 RGB、11×11 均匀窗口。网页四路均为 H.264 CRF16；细微差别仍可能受压缩影响。计时为 CUDA 同步后的原生编码/解码调用总和；显存为 PyTorch 分配峰值。指标不等同于主观观感。</p>
<p><a href="report.json">下载结果清单</a> · <a href="manifest.json">输入与时间戳</a></p></details></main>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id);let index=0,revealed=false,videos=[],order=[];
const permutations=DATA.map(()=>{const a=['baseline','light','tae'];for(let i=2;i>0;i--){let n=new Uint32Array(1);crypto.getRandomValues(n);const j=n[0]%(i+1);[a[i],a[j]]=[a[j],a[i]]}return a});
const names={baseline:'H3 默认 VAE',light:'Light VAE',tae:'TAE'};
DATA.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1} / ${DATA.length} · ${c.id}`;$('case').append(o)});
function labels(){document.querySelectorAll('.variant').forEach((el,i)=>{el.textContent=revealed?names[order[i]]:`候选 ${String.fromCharCode(65+i)}`});document.querySelectorAll('.metrics').forEach(el=>el.classList.toggle('hidden',!revealed));$('reveal').textContent=revealed?'隐藏 VAE 名称与指标':'显示 VAE 名称与指标'}
function render(){videos.forEach(v=>v.pause());const c=DATA[index];order=permutations[index];$('case').value=index;$('status').textContent='';$('grid').replaceChildren();
$('description').textContent=`${c.name} · ${c.width}×${c.height} · ${c.frames} 帧 · ${c.duration_seconds.toFixed(2)} 秒`;
['original',...order].forEach((key,i)=>{const card=document.createElement('div');card.className='card';const label=document.createElement('div');label.className='label'+(i?' variant':'');label.textContent='原视频参照';const video=document.createElement('video');video.src=c.id+'/'+key+'-web.mp4';video.controls=true;video.muted=true;video.playsInline=true;video.preload='metadata';video.addEventListener('error',()=>{$('status').textContent='视频加载失败，请确认内网/VPN连接后刷新。'});const meta=document.createElement('div');meta.className='meta'+(i?' metrics hidden':'');if(i){const m=c.results[key];meta.textContent=`PSNR ${m.psnr_rgb_db.toFixed(2)} dB · SSIM ${m.ssim_rgb.toFixed(4)}\n编码 ${m.encoder.wall_seconds.toFixed(2)} s / 解码 ${m.decoder.wall_seconds.toFixed(2)} s\n显存峰值：编码 ${(m.encoder.peak_allocated_bytes/2**30).toFixed(2)} / 解码 ${(m.decoder.peak_allocated_bytes/2**30).toFixed(2)} GiB`;meta.style.whiteSpace='pre-line'}else{const link=document.createElement('a');link.href=c.url;link.textContent='查看原始文件（含原音轨）';link.target='_blank';link.rel='noopener';meta.append(link)}card.append(label,video,meta);$('grid').append(card)});videos=[...document.querySelectorAll('video')];labels();}
function seek(t){videos.forEach(v=>{if(v.readyState)v.currentTime=Math.max(0,Math.min(t,v.duration||t))})}
$('case').onchange=()=>{index=+$('case').value;render()};$('prev').onclick=()=>{index=(index-1+DATA.length)%DATA.length;render()};$('next').onclick=()=>{index=(index+1)%DATA.length;render()};$('reveal').onclick=()=>{revealed=!revealed;labels()};$('pause').onclick=()=>step(0);
$('play').onclick=async()=>{const t=videos[0].currentTime;seek(t);const result=await Promise.allSettled(videos.map(v=>v.play()));if(result.some(r=>r.status==='rejected'))$('status').textContent='部分视频尚未就绪，请稍后再次同步播放。'};
$('seek').oninput=()=>seek($('seek').value/1000*DATA[index].duration_seconds);
function step(d){videos.forEach(v=>v.pause());const c=DATA[index],tb=c.time_base.split('/').map(Number),times=c.pts.map(p=>(p-c.pts[0])*tb[0]/tb[1]),t=videos[0].currentTime;let n=times.findIndex(x=>x>=t-.001);if(n<0)n=times.length-1;seek(times[Math.max(0,Math.min(times.length-1,n+d))]);}
$('back').onclick=()=>step(-1);$('forward').onclick=()=>step(1);
setInterval(()=>{if(!videos.length)return;const v=videos[0];$('time').textContent=v.currentTime.toFixed(2)+' s';$('seek').value=v.currentTime/DATA[index].duration_seconds*1000;if(!v.paused)videos.slice(1).forEach(x=>{if(x.readyState>=2&&Math.abs(x.currentTime-v.currentTime)>.08)x.currentTime=v.currentTime})},100);render();
</script></html>'''


def build(root):
    manifest=json.loads((root/'manifest.json').read_text())
    report=[]
    for c in manifest['cases']:
        records={}
        for v in ('baseline','light','tae'):
            p=root/c['id']/(v+'.json')
            if not p.exists(): break
            r=json.loads(p.read_text())
            if r['status']!='complete': break
            records[v]=r
        if len(records)!=3: continue
        assert len({r['input_rgb_sha256'] for r in records.values()})==1
        report.append(dict(c,results=records))
    assert report,'No complete three-VAE cases'
    write_json(root/'report.json',report)
    (root/'player.html').write_text(HTML.replace('__DATA__',json.dumps(report,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();print('REPORT_READY',len(build(a.results)))
