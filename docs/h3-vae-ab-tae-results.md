# Paired TAE-only B300 run, 2026-09-16

TAE smoke `dlc1e17c3uv222dn` and ten-case job `dlc7l1o968ignln0` both succeeded.
Only TAE was executed. Both consumed the saved latents and audio from the original
ten-case job `dlc1xe41e2xvq1cp`; default and Light VAE results were not rerun.
Every TAE record matches its baseline's case, latent SHA256, source-audio SHA256
and decoded AAC SHA256, and contains 124 frames at 1344×768 and 24 fps.

The source snapshot was `59fcc391af5275c3f14880688e88432bf7ca50cc` for the ten-case
run and `164509d5b7f07ce30ca0c5a8dcd13d5d115e3f3f` for smoke; the TAE runner was
unchanged. The pinned model and conversion contract are described in the runbook.
The same Thailand B300 profile, Torch 2.12.1+cu130, FP32 weights and FP16 decode
autocast were used, with one GPU at a time. Reported device name is NVIDIA L20D,
capability 10.3, as in the original evaluation; retain this actual runtime identity.

After a separate saved warmup case, the ten-case mean measured TAE decoder time
was **0.115662 seconds**, with peak allocated **2,001,889,792 bytes (1.86 GiB)**.
These are decoder-stage measurements, excluding RGB conversion, encoding and
loading. They are not request latency. TAE uses native sequential temporal
decoding without spatial tiling; default/Light use their native tiled decoders.
This records the implementations' different memory strategies rather than
claiming identical kernels or scheduling. No speedup implies quality equivalence.

JSON records and browser previews are at
`oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/dlc7l1o968ignln0/eval/`.
Lossless TAE masters remain on CPFS under the same job's result directory.
The smoke preview played successfully in Edge and was visually inspected for
basic layout/color plausibility; independent full temporal quality review remains
separate from technical generation acceptance.

The public randomized three-column player now uses the new TAE previews from
Thailand OSS. No historical S3 TAE URLs remain in that page. Default/Light
previews are unchanged. Names and metrics remain hidden until revealed, and
each case has an independently randomized order.
