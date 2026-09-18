# 15-chip VAE reconstruction results

Job `dlch59moaqg34mkh` reconstructed 15 source clips with each VAE's own encoder and decoder on one GPU. No DiT, prompt conditioning, scheduler, or posterior sampling was used (NFE=0). The earlier 20-long-video job was stopped at the user's request and is excluded.

[Comparison player](https://review-cdn.loopit.com.cn/world-model/results/lynnreal-omni/dlch59moaqg34mkh/eval/player.html) (office network / company VPN).

## Inputs and execution

- 15 clips selected from26 candidates before inspecting reconstruction outputs; source-frame contact sheets at20%,50%,80% guided face/eyes/hands, small/distant people, motion, dark/underwater and texture coverage. This is a curated test set, not a random population sample.
- 5,211 frames per VAE; 211.657 seconds in total; individual clips10.667–16.083 seconds. Full durations and source timestamps retained.
- Short edge capped at768, no upscaling; replicate spatial padding to16 then crop. Original reference uses the same resize. FP32 weights, FP16 autocast, native temporal protocol with tail padding.
- Platform single-B300 environment; CUDA reports `NVIDIA L20D`, capability10.3, Torch2.12.1+cu130. No second GPU was used.
- Code `ff372a3411dfba1d0c660e09d5931fb138f7fa3f`; H3 `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`; Light `e453444c1a52b73a0c5eb0023d208c473c2bb26a`; TAEHV `011dfc2112197741c540e0bdd5b7b67bcc930771`.
- [Exact input manifest](https://review-cdn.loopit.com.cn/world-model/results/lynnreal-omni/dlch59moaqg34mkh/eval/input-manifest.json) SHA256 `e6facae453cd4425c8a7cdfb279425cf8e0c13cb10ee79184474d9435bd88801`. Output `manifest.json` has identical JSON data with Linux line endings; the exact input is retained separately.

## Objective reconstruction metrics

PSNR and SSIM below are equally weighted means over15 clips, measured in float RGB before browser compression. SSIM uses11×11 uniform valid windows and population covariance. Differences in these metrics do not establish a subjective-quality ranking.

| VAE | Mean PSNR (dB) | Mean SSIM | Encoder total (s) | Decoder total (s) | Encoder / decoder peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|
| H3 default | 36.724 | 0.96185 | 376.197 | 233.946 | 12.949 / 10.798 |
| Light | 36.799 | 0.96331 | 375.208 | 169.671 | 10.448 / 8.297 |
| TAE | 30.258 | 0.91192 | 10.217 | 6.279 | 0.750 / 0.729 |

Light is close to H3 default on these metrics (+0.075dB average PSNR); Light's native decoder calls take27.5% less total time. TAE has lower reconstruction fidelity by these metrics and much faster native encoder/decoder calls. Timing excludes CPU video decode, transfers, metric computation and preview encoding; memory values are PyTorch allocated peaks, not total device memory.

## Per-clip PSNR

| Case | Selection focus | H3 default | Light | TAE |
|---|---|---:|---:|---:|
| chip-03-5e9a44a4 | 运动中的人脸、护目镜、头发与闪光特效 | 33.718 | 33.834 | 28.334 |
| chip-04-7332d93c | 正面眼睛与嘴部、摘护目镜时的手指 | 36.288 | 36.514 | 30.551 |
| chip-05-c32820bc | 近景人脸转俯视远景、小人物与复杂废墟纹理 | 33.752 | 33.906 | 28.655 |
| chip-06-cadc9fbc | 手指托举小物件、刻字细节与渐变背景 | 38.047 | 38.044 | 30.534 |
| chip-07-d03da49c | 暗光男性侧脸、发丝与高反差背景 | 41.260 | 41.469 | 35.979 |
| chip-08-out_bv11 | 森林中中远景人物、小脸与细碎植物 | 39.193 | 39.286 | 33.448 |
| chip-10-out_bv11 | 奔跑人物、小脸、细肢体与高频草木 | 33.666 | 33.707 | 27.284 |
| chip-11-out_bv11 | 眼睛近景、眉毛睫毛、动物毛发 | 36.890 | 36.864 | 29.562 |
| chip-12-out_bv11 | 快速手臂动作、衣袖、兽类口齿和运动模糊 | 33.901 | 33.961 | 27.027 |
| chip-14-out_bv11 | 多人远景与俯视小脸、姿态和背景细纹理 | 34.371 | 34.412 | 27.929 |
| chip-16-out_bv11 | 手部与衣饰纹样、远景小人物、近景侧脸 | 38.847 | 38.886 | 31.565 |
| chip-19-out_bv11 | 水下低对比人脸、发丝漂动与强光变化 | 40.680 | 40.742 | 32.368 |
| chip-21-out_bv11 | 空中多人和小脸、手势、盔甲与云层 | 38.232 | 38.233 | 31.635 |
| chip-24-out_bv11 | 女性侧脸、头饰发丝、远景动作与高亮特效 | 35.982 | 36.017 | 29.496 |
| chip-25-out_bv11 | 双人中景表情、手臂动作、远景人物和复杂特效 | 36.037 | 36.109 | 29.510 |

## Validation

- Native-vs-streaming probes: all three models have zero latent and RGB max absolute difference on the51-frame64×64 probe (39 reconstructed frames).
- All45 result records match the run identity/source hashes; all15 groups share exactly the same common resized input RGB hash. Per-frame metrics and aggregates independently recomputed.
- All60 MP4s have the expected frame count, dimensions and source-relative PTS (maximum PTS error0). All45 reconstruction byte hashes match the producer records; hashes were recorded for the15 original-reference previews.
- Edge browser checks passed for all15 groups /60 videos: each played beyond3 seconds, no page errors, randomized model ordering, hidden/revealed names and metrics, aligned pause and forward frame stepping.
- Evidence: `verification-metric.json`, `verification-media.json`, `verification-browser.json` under the same result prefix. GPU job succeeded and ended; monitoring is paused after validation.

Browser previews use the same H.264 CRF16/yuv420p settings and are silent; original-source links retain audio. The reference is fixed on the left, three VAE positions are randomized per case, names/metrics are hidden initially, and the player supports synchronized pause/frame stepping.
