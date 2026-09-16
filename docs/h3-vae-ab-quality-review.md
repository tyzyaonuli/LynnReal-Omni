# Preliminary visual review, 2026-09-16

The ten-case job `dlc1xe41e2xvq1cp` produced all twenty paired previews.
Automated Edge playback checks passed beyond four seconds for each preview.
An AI review inspected thirty paired browser screenshots: each case at
0.25, 2.5 and 4.75 seconds. This is a preliminary static review, not independent
human approval or a continuous temporal-quality assessment.

Across these samples, subject placement, pose and broad scene structure align.
Light VAE generally softens fine texture. The ribbon, sea spray and rock surfaces
make this most apparent. This supports a speed/memory versus detail tradeoff;
it does not establish perceptual equivalence or justify a blanket quality pass.

| Case | Observed in sampled paired frames |
| --- | --- |
| 01 Rally drift | Car, lighting and signs align; road and distant vegetation detail is smoother with Light. |
| 02 Motocross | Rider pose and stadium layout align; dirt and small bike details are softer with Light. Unusual bike structure appears in both middle samples. |
| 03 Parkour | Runner and architecture align; wall, railing and ground detail is softer with Light. |
| 04 Canyon | Rock layout and reflections align; fine rock ridges are smoother with Light. Strong motion blur appears in both end samples. |
| 05 Lighthouse | Lighthouse and large wave shapes align; foam and spray detail is reduced with Light. |
| 06 Ribbon dancer | Pose and ribbon silhouette align; ribbon folds, striations and thin trailing strands are visibly smoother with Light. |
| 07 FPS | Corridor, weapon and flash positions align; small highlights and repeated edges are softer with Light. Both outputs have smeared edge structure. |
| 08 Futuristic racing | Vehicles and signs align; neon, facade and wet-road microtexture is smoother with Light. |
| 09 Voxel world | Character, blocks and colors align; small edges and water detail are smoother with Light. |
| 10 Dragon | Wing poses align; membrane ridges and mountain detail are softer with Light. |

## Evidence limits and remaining acceptance

Screenshots show scaled CRF16 browser previews, not full-resolution lossless
outputs. Scaling and preview encoding can affect subtle differences. Initial
screenshots for cases 03, 09 and 10 retain a browser loading indicator; obscured
regions were excluded from assessment. Loading overlays are not model defects.

Three samples cannot rule out flicker, temporal discontinuities or brief artifacts.
The browser playback check establishes playability, not visual quality. Continuous
paired playback and independent human assessment remain pending. In particular,
review the ribbon motion, foam and moving rock texture before deciding whether
the observed smoothing is acceptable for deployment.

Per-case latent, source-audio and decoded-AAC hashes were checked separately;
matching audio hashes establish paired identity, not perceptual audio quality.

The saved review evidence is under the experiment journal's
`assets/lynnreal-issue2-inputs-20260916/ten-dlc1xe41e2xvq1cp/visual-review/`:
`captures.json`, thirty PNGs and `observations-*.json`. These local artifacts are
not bundled in the repository. The shareable previews remain available in the
[OSS player](https://leap-worldmodel-thailand.oss-ap-southeast-7.aliyuncs.com/world-model/public/lynnreal-h3-vae-ab/dlc1xe41e2xvq1cp/player.html?v=2);
the default endpoint downloads the HTML, which can then be opened locally.
