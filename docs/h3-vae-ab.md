# Official H3 / Light VAE comparison

The ten reference inputs are in `eval/h3_vae_ab/manifest.json`. H3 is the official
FL2VA checkpoint at `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`; this experiment
does not use the LynnReal Standard or Flash denoiser. Five actual denoiser calls
require six scheduler points. The reference source was reconstructed; the old
NFE-5 job's exact runtime identity remains unavailable.

This entry uses **one B300**, BF16 DiT/text, FP32 VAE weights, FP16 VAE decode,
native attention and no model compilation. It loads text, DiT, audio VAE and
each video VAE in separate phases. It preserves the exact saved sampling result
for both decoders, including one audio result. It does not reproduce historical
TP8 timing on eight H100s. The runtime is pinned Diffusers
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`, with LynnReal's LightVAE wrapper.
Both decoders use native 256×256 tiles, 64-pixel minimum overlaps and one tile at
a time. LightVAE's default all-tiles batching is explicitly disabled so the
performance/VRAM comparison does not mix a batching change with a decoder change.

## CPU checks before GPU allocation

```bash
python -m unittest discover -s test -p test_h3_vae_ab.py -v
python script/eval_h3_vae_ab.py --dry-run --output /tmp/h3-ab-plan
```

Use the existing CPU model-cache launcher to stage the pinned transformer through
the working ModelScope / Hangzhou OSS / Thailand replication path:

```bash
bash script/aliyun_thailand_model_cache_submit.sh --stage --submit --h3-fl2va
# Wait for the stage marker AND all matching objects in Thailand OSS, then:
bash script/aliyun_thailand_model_cache_submit.sh --sync --submit --h3-fl2va
```

The Thai CPFS VPC lane cannot directly reach external model sites. The sync runs
`script/prepare_h3_vae_ab.py` on CPU, checks each byte against the immutable HF tree and reuses matching cached
components through hard links. Only a complete model gets `H3_VAE_AB_READY.json`.
Partial downloads are never consumed by the GPU runner. An existing Ref2VA or
distilled transformer is not substituted for the official FL2VA transformer.

`script/preflight_h3_vae_ab_runtime.py` runs in the same environment image on CPU.
It verifies the environment archive, actual scheduler sigmas, FL2VA component
selection and successful LightVAE loading. Set `PROBE_RESULT` to its JSON receipt
path. The existing environment archive is reused, without installing dependencies
on the B300.

## Single-case smoke, then all ten

Use the existing Thailand submitter, which retains its region/workspace/quota,
role injection and exactly-one-GPU constraint:

```bash
ALIYUN_PROFILE=<profile> PAI_MAX_MINUTES=45 \
bash script/aliyun_thailand_b300_submit.sh --submit --h3-vae-ab \
  --preflight-receipt /path/weights-preflight.json \
  --runtime-receipt /path/runtime-preflight.json --case 01-rally-drift
```

Omit `--case` for the ten-case run; replace `--submit` with `--dry-run` to render.
Only submit the full run after the smoke's latent/audio hashes, geometry and
videos pass inspection. All authored code must be committed before submission.

The underlying runner supports `--variant baseline|light|both`, `--case`, and
resuming the same `--output` directory. Its manifest, source and weight marker
identity must match; completed phases are reused. New configuration or source
requires a fresh directory. A failed case can be resumed by its ID.

The cloud entry exposes the same selection with `--h3-variant baseline|light|both`.
To retry a failed case, submit the **same committed runtime** with
`--resume-job <previous-dlc-job-id> --case <case-id>` and the same receipts.
The bootstrap copies the previous evaluation directory to the new job directory,
records `resumed-from-job.txt`, and leaves the original evidence untouched.
Completed decoder results are skipped only after checking latent, audio, MP4 and
preview SHA256 hashes. Missing or damaged videos are regenerated. Model execution
failures are appended to `failures.json` with stage, error, time and job ID.
Do not resume from a job that is still writing results. Runtime code changes
require a fresh run; a resume is not a way to mix implementations in one comparison.

## Acceptance evidence (required before calling the evaluation complete)

- Run the committed repository entry for both smoke and ten-case evaluation;
  retain command, source revision, receipts, job ID and output URI.
- Pass the CPU manifest, report, integrity/resume and no-GPU dry-run tests.
- Inspect the smoke's latent/audio identities, dimensions and paired videos.
- Exercise same-revision cloud resume and a single-case retry; verify that completed
  cases retain their output hashes and original job provenance.
- Rebuild the HTML from persisted results without GPU access.
- Record cross-job cold/warm timing separately from within-job decoder warmup.
- Review every case, publish the static report, and provide a PR plus a tested
  copyable full-run command. Local source commits alone do not satisfy PR delivery.

These are acceptance gates, not a claim that every live-cloud check has passed.

## Evidence and interpretation

Each case retains `conditioning.pt`, `latents.pt`, `latent.json`, `audio.pt`,
lossless `baseline.mp4` / `light.mp4`, browser previews and per-decoder JSON.
The player refuses mismatched latent, source audio or decoded AAC hashes.
Warmup seed 40999 is stored separately and excluded from the ten-case report.

Record each stage's synchronized wall time, allocated/reserved CUDA peaks,
resident allocation before the stage, incremental allocation and sampled device
usage (100 ms, not a guaranteed hardware peak). The decoder timer wraps actual
VAE decoding; unpacking/RGB conversion and MP4 encoding have separate timings.
The phased end-to-end sum shares conditioning/sampling/audio time across A/B;
it is not an independently measured request latency. Loading and warmup are
reported separately and are not included in this warm-stage sum.

`run.json`, `timings.json`, `pip-freeze.txt` and bootstrap timestamps preserve
job/code/model identities, actual GPU details, precision, scheduling and loading
evidence. Record PAI scheduling/image-pull timestamps from job events separately.
Repeat the identical run on a subsequent job to measure environment/cache reuse;
do not label within-job warmup as a cross-job cold/warm comparison.

Results are copied to the established OSS results prefix under the PAI job ID.
Generate or rebuild the portable player with:

```bash
python script/report_h3_vae_ab.py --results /path/to/results/eval
```

The HTML references only adjacent preview videos. Quality judgments require
watching the paired clips; pixel differences alone are not a quality verdict.
