# Video VAE reconstruction (no DiT)

This experiment measures each model's own encoder → decoder round trip on real
source videos. It does not load the H3 transformer, prompt encoder, scheduler,
or audio model. NFE is zero. It is a different question from decoding a shared
diffusion latent, and the two rankings must not be conflated.

## Reproduce

1. Freeze the source selection in JSON (`cases`: id, name, size, URL), and download
   each source as `<id>.mp4`. Keep the selection and source page provenance.
2. Run `script/prepare_vae_reconstruction.py --manifest selection.json --sources
   sources --output manifest.json --short-edge 768 --storage-prefix
   world-model/lynnreal-omni/inputs/EXPERIMENT`. This CPU step records source SHA256,
   dimensions, every original PTS, frame count and duration. A zero short edge
   keeps source resolution. Upload those files to the configured Thailand bucket
   under that prefix before allocating a GPU.
3. Use the existing submitter with `--submit --vae-reconstruction manifest.json
   --preflight-receipt weights.json --runtime-receipt runtime.json`. It requires
   exactly one GPU and the same pinned, verified weight/runtime caches as the H3
   A/B runner. `--case ID` runs a smoke case. `--resume-job JOB` restores completed
   results only when the runner, wrapper and manifest identities match.

The local runner also works directly with `--manifest`, `--sources`, `--output`,
`--h3`, `--light-vae`, and `--checkpoint`. `--dry-run` checks the manifest without
importing Torch or loading models.

## Comparison contract

- Full source duration and source timestamps, including variable frame timing.
  Common bilinear resize, no upscaling; replicate-pad to16 for the networks and
  crop back to the displayed size. The reference shows that same common resize.
- H3 default and Light use ImageNet pixel normalization and posterior **mode**.
  TAE uses its own deterministic encoder with RGB [0,1]. No diffusion latent
  mean/std transformation is inserted into these self-contained round trips.
- FP32 weights, FP16 autocast; H3/Light native256 spatial tiles, overlap64,
  tile batch1. TAE retains its released non-tiled sequential temporal protocol.
- Encode native17-frame blocks, concatenate five latent tokens per block, then
  drop the three global tail tokens. Repeat source tail frames until the decoded
  sequence covers every source frame, then trim only the excess. H3 decode keeps
  its native temporal overlap/blending; TAE carries decoder memory across the
  entire video. Streaming introduces no independent windows or seam resets.
- Before production, each model must match its native full encode/decode on a
  51-frame64x64 probe (39 reconstructed frames), maximum RGB/latent absolute
  error below.002. This checks temporal protocol, not all resolutions or inputs.

## Evidence and interpretation

Each result stores source and common-input hashes, latent hashes (latent files
stay on CPFS), RGB/output preview hashes, per-frame PSNR/SSIM and model identities.
PSNR is computed from aggregate float RGB MSE. SSIM is RGB, uniform11x11 valid
windows with population covariance; do not compare numerically to a different
SSIM convention. Both are measured **before** browser compression. They measure
reconstruction fidelity, not perceptual preference or generation ability.

Encoder and decoder native calls are separately CUDA-synchronized and timed.
Peak allocated/reserved memory and peak incremental allocated memory are
recorded per stage. These are PyTorch counters, not total device process usage.
Full wall time includes transfers, source decode, metrics and browser encoding.
The small equivalence probe warms up the model; it is not a rigorous throughput
benchmark. Compare visual quality first, and label these timing boundaries.

Browser previews use identical H.264 CRF16/yuv420p settings, silent to focus on
visual reconstruction. Audio is available from the original source URL and is
not evaluated. Preview compression can obscure tiny differences; raw-pixel
metrics and retained latents provide separate evidence.
