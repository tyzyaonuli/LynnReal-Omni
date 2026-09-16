# Controlled application-cache startup comparison, 2026-09-16

Reusing the completed application cache reduced CPU preparation from 1,307.17 s
to 26.34 s, saving 1,280.83 s (21.35 minutes). GPU job submission to the first
complete A/B pair was essentially unchanged: 370.58 s versus 368.55 s.
This is preparation reuse, not a comparable reduction in inference latency.

Both runs used source snapshot `18ee427ef5c699f112243c39734ec69e88ff3d3a`, one GPU,
case `01-rally-drift`, seed 41001 and a separate seed-40999 warmup. Official model,
Light VAE, precision, attention, tiling and actual NFE were identical to the
versioned manifest. Both GPU workers were scheduled on the same node. Each ran
fresh inference; neither resumed previous outputs. Latent and both MP4 SHA256
values matched exactly across runs.

| Stage (seconds) | Empty application cache | Reused application cache |
| --- | ---: | ---: |
| CPU environment preparation/validation | 332.59 | 26.01 |
| CPU weight copying+hash verification / marker+size validation | 974.50 | 0.20 |
| CPU preparation total | 1307.17 | 26.34 |
| GPU job created to Running, including provisioning | 72.00 | 80.00 |
| Text model load | 57.73 | 56.85 |
| DiT model load | 55.74 | 52.74 |
| Audio VAE load | 0.83 | 0.82 |
| Default visual VAE load | 8.47 | 8.41 |
| Light VAE load | 6.59 | 6.44 |
| Runner start to first baseline case complete | 234.99 | 230.12 |
| Runner start to first A/B pair complete | 258.81 | 253.68 |
| GPU job created to first A/B pair complete | 370.58 | 368.55 |

The CPU cold stage copied and verified 151,780,990,295 bytes into the new namespace
`/cpfs/world-model/lynnreal-omni/cache/startup-ab/h3-startup-20260916-r1`.
The warm stage copied zero bytes. GPU processes consumed that completed cache in
both cases; environment extraction had already happened before GPU allocation.
GPU environment selection was below the one-second timestamp resolution in both
runs; it must not be interpreted as literally zero-cost initialization.

Text/sampling/default-decoder warmup took approximately 27.3 / 25.5 / 15.5 s in
both runs. Compilation was disabled. No persistent compile-cache benefit is
claimed. Source CPFS data and host page/image caches were not cleared; this is
not a cold-machine or first-ever internet-download benchmark. Image events show
the runtime image already cached on the worker. Only one pair of runs was made;
small model-loading differences should not be treated as statistically robust.

CPU preparation plus GPU submission-to-pair is a **stage sum**, approximately
27.96 minutes cold versus 6.58 minutes warm. It excludes CPU job provisioning,
manual/monitor handoff gaps and output upload, and is not a directly measured
continuous request latency.

## Job evidence and upload failure

| Job | ID | Outcome |
| --- | --- | --- |
| Cold CPU preparation | `dlc1syvlsavgjcy9` | Succeeded |
| Cold GPU inference | `dlc1pn8wlmwbcvzh` | Inference passed; job Failed during upload |
| Warm CPU validation | `dlc1wb4s22vp0ez0` | Succeeded |
| Warm GPU inference | `dlc1a42o0aa04vow` | Succeeded |

The cold job's final `cp` to the OSS mount failed to close
`eval/_warmup/latents.pt` with `Input/output error`. Its completed inference
timings, summary, case records and bootstrap timestamps were recovered from OSS.
This post-measurement failure does not invalidate the recorded time to first
result, but the cold cloud job must remain labelled Failed and its upload must
not be claimed complete. Recovery of the missing artifact is tracked separately.

GPU evidence is under
`oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/<job-id>/`.
CPU receipts are under the same results prefix with directory names
`h3-startup-h3-startup-20260916-r1-cold` and
`h3-startup-h3-startup-20260916-r1-warm`.
See [the runbook](h3-vae-ab.md#controlled-application-cache-coldwarm-comparison)
for the reusable submission commands.
