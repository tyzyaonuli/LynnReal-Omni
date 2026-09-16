"""Create a portable synchronized A/B player from verified per-case results."""
import argparse
import json
from pathlib import Path


def build(results, manifest_path=None):
    cases = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
    expected = {c["id"]: c for c in manifest["cases"]} if manifest else None
    for path in sorted(results.glob("*/baseline.json")):
        if path.parent.name.startswith("_"):
            continue
        other = path.with_name("light.json")
        if not other.exists():
            continue
        a, b = json.loads(path.read_text(encoding="utf-8")), json.loads(other.read_text(encoding="utf-8"))
        if a["case"] != b["case"]:
            raise ValueError(f"A/B case mismatch: {path.parent.name}")
        if expected is not None and expected.get(path.parent.name) != a["case"]:
            raise ValueError(f"Manifest/result mismatch: {path.parent.name}")
        if any(a[key] != b[key] for key in ("latent_sha256", "audio_sha256", "decoded_audio_sha256")):
            raise ValueError(f"A/B input mismatch: {path.parent.name}")
        cases.append({"id": path.parent.name, "title": a["case"]["title"], "prompt": a["case"]["request"]["prompt"],
                      "seed": a["case"]["request"]["seed"], "a": path.parent.name + "/baseline-web.mp4",
                      "b": path.parent.name + "/light-web.mp4", "a_metrics": a["decoder"], "b_metrics": b["decoder"],
                      "latent_sha256": a["latent_sha256"]})
    if not cases:
        raise ValueError("No complete A/B pairs")
    if expected is not None:
        order = {key: index for index, key in enumerate(expected)}
        cases.sort(key=lambda case: order[case["id"]])
    data = json.dumps(cases, ensure_ascii=False).replace("<", "\\u003c")
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>H3 Default VAE / LynnReal Light VAE</title><style>
body{background:#10131a;color:#edf1f7;font:16px system-ui;margin:24px auto;max-width:1500px;padding:0 20px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:black}button,select{padding:10px;margin:6px;background:#24334b;color:white;border:0;border-radius:6px}pre{white-space:pre-wrap;color:#b3c5dc}#prompt{line-height:1.5}@media(max-width:700px){.pair{grid-template-columns:1fr}}
</style><h1>官方 H3：默认 VAE / Light VAE</h1><p>单张 B300 · BF16 H3 · 实际 NFE 5 · 1344×768 · 124 帧 · 24 fps · 同一份 latent 和音频</p>
<button id="prev">上一组</button><button id="next">下一组</button><button id="play">播放</button><button id="pause">暂停</button><button id="all">顺序播放全部</button>
<label>音轨 <select id="audio"><option value="a">默认 VAE</option><option value="b">Light VAE</option><option value="mute">静音</option></select></label>
<h2 id="title"></h2><p id="prompt"></p><p id="info"></p><div class="pair"><section><h3>A · H3 默认 VAE</h3><video id="a" playsinline controls preload="metadata"></video><pre id="ma"></pre></section><section><h3>B · LynnReal Light VAE</h3><video id="b" playsinline controls preload="metadata"></video><pre id="mb"></pre></section></div>
<p>解码耗时与显存均为独立阶段测量；单卡结果不能直接与历史 8×H100 的总耗时比较。</p><script>
const cases=__DATA__;let i=0,auto=false;const a=document.querySelector('#a'),b=document.querySelector('#b');
function sound(){const value=document.querySelector('#audio').value;a.muted=value!=='a';b.muted=value!=='b'}
function metrics(m){return `解码：${(m.wall_seconds*1000).toFixed(1)} ms\\n峰值 allocated：${(m.peak_allocated_bytes/2**30).toFixed(2)} GiB\\n解码增量：${(m.incremental_peak_allocated_bytes/2**30).toFixed(2)} GiB`}
function load(n){i=(n+cases.length)%cases.length;a.pause();b.pause();const c=cases[i];document.querySelector('#title').textContent=`${i+1}/${cases.length} ${c.title}`;document.querySelector('#prompt').textContent=c.prompt;document.querySelector('#info').textContent=`seed ${c.seed} · ${c.id}`;a.src=c.a;b.src=c.b;document.querySelector('#ma').textContent=metrics(c.a_metrics);document.querySelector('#mb').textContent=metrics(c.b_metrics);sound()}
async function play(){await Promise.all([a.play(),b.play()])}
document.querySelector('#prev').onclick=()=>{auto=false;load(i-1)};document.querySelector('#next').onclick=()=>{auto=false;load(i+1)};
document.querySelector('#play').onclick=play;document.querySelector('#pause').onclick=()=>{auto=false;a.pause();b.pause()};
document.querySelector('#all').onclick=()=>{auto=true;load(0);play()};document.querySelector('#audio').onchange=sound;
a.onended=()=>{b.pause();if(auto&&i+1<cases.length){load(i+1);play()}};
a.onseeking=()=>{if(Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime};
setInterval(()=>{if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.12)b.currentTime=a.currentTime},250);load(0);
</script></html>'''
    (results / "player.html").write_text(template.replace("__DATA__", data), encoding="utf-8")
    (results / "report.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "eval/h3_vae_ab/manifest.json")
    args = p.parse_args()
    print(f"Generated {len(build(args.results, args.manifest))} A/B pairs")
