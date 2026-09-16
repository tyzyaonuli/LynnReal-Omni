# Missing-output recovery acceptance, 2026-09-16

Job `dlcl5lgy25hif5jj` passed the isolated missing-preview recovery check.
It copied the completed ten-case evaluation from `dlc1xe41e2xvq1cp`, preserved
the selected Light preview outside the evaluation directory, and removed that
preview from the copied case `01-rally-drift`. The normal runner then loaded
only Light VAE and regenerated that case's Light outputs.

The persisted acceptance receipt confirms all 77 video/tensor file hashes
match the pre-injection values and 32 other per-case JSON records are unchanged.
The regenerated Light record identifies the new job and source snapshot
`6912350ec5c286e0fe79670e827e635606a5898c`. The original ten-case directory was
never altered. This validates missing-output detection and regeneration;
it does not simulate a CUDA crash or establish that every possible failed stage
is recoverable.

The regenerated decoder took 18.73 seconds including its first invocation in
this process. Its inherited warmup had already been marked complete. Do not mix
this recovery timing into the original warmed ten-case performance comparison.

## Upload failure and evidence rescue

The cloud job did not complete successfully: final copying through the OSS mount
reported an I/O error and a connection timeout on warmup video files. The job
subsequently reached `Stopped`. Preserve that cloud status separately from the
successful inference/recovery check.

CPU job `dlc1pe70uii0grl0` read the CPFS JSON/text metadata and returned a compressed
archive through job logs, whose SHA256 was verified locally before extraction.
The first CPU rescue, `dlc1bipx564fue89`, exposed the API's 2048-byte log-line
truncation: archive hash validation rejected its payload. The reusable helper
now emits 1024-byte chunks. Neither rescue reran GPU inference.

The verified acceptance receipt is independently retained at
`oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/dlcl5lgy25hif5jj-recovery/retry-acceptance.json`.
Storage repair uses the rescued metadata and server-side copies of original
artifacts whose equality was established by the recovery receipt. The repair
operation log records each source and target separately. The public ten-case
player continues to use the original successful benchmark job.

Repair completed with 125 successful copy/upload operations. All 77 artifact
sizes match; 61 ETags match directly, and the other 16 multipart objects have
matching CRC64 values despite changed multipart boundaries. `upload-repair.json`
and `oss-repair-operations.json` beside the repaired results record verification
and provenance. The HTML was rebuilt from rescued records without GPU access.

For CPU metadata rescue through the existing launcher:

```bash
export H3_RECOVER_JOB=dlcl5lgy25hif5jj
bash script/aliyun_thailand_model_cache_submit.sh --sync --submit --h3-recover
```

Join the `H3_METADATA_CHUNK=` lines, base64-decode, verify against
`H3_METADATA_SHA256=`, then extract only safe relative regular-file paths.
An `H3_METADATA_END` line alone is not an integrity check. This helper rescues
small metadata; it does not replace a general reliable object-upload backend.
