# Human-detail extension (2026-09-17)

Twelve additional cases extend the original ten to 22 three-decoder comparisons
(66 previews). Cases 11–14 provide face, eye, teeth and finger closeup controls;
cases 15–22 request medium/wide framing with smaller faces and multiple people.
Prompt proportions are targets, not measured face sizes.

AB job `dlcr1jlzezv9xhdi` and subsequent TAE-only job `dlc36o5cazoj3m4c`
both succeeded at revision `a1a7b15fa9c4df32e328b8eaed9ab96a2fb6b33c`.
Both used `--case human-details`; jobs ran sequentially with one B300-profile GPU.
TAE used the AB job's saved latents and audio. Original ten outputs were preserved.
All 22 triples passed manifest, latent, source-audio and decoded-AAC identity checks.
Full masters and tensors remain on CPFS; JSON evidence and web previews are on OSS.

| New 12 cases | Mean decoder seconds | Peak allocated GiB |
| --- | ---: | ---: |
| H3 default | 4.5943 | 15.9113 |
| Light | 3.8609 | 9.4181 |
| TAEHV | 0.1153 | 1.8644 |

These measure the decoder phase, not end-to-end generation. The full VAEs use
native spatial tiling; TAE uses its native sequential temporal implementation.
Retained allocations and decoder strategies differ; see the TAE results runbook.

The merged player uses the existing object prefix
`world-model/public/lynnreal-h3-vae-ab/dlc1xe41e2xvq1cp/`; that prefix is a stable
presentation location, not a claim that all results came from the original job.
New baseline/Light previews come from `dlcr1jlzezv9xhdi`, new TAE previews from
`dlc36o5cazoj3m4c`. Original TAE previews come from `dlc7l1o968ignln0`.

[Internal comparison player](https://review-cdn.loopit.com.cn/world-model/public/lynnreal-h3-vae-ab/dlc1xe41e2xvq1cp/player.html?v=6)
requires office network or company VPN. The verified HTML returns `200 text/html`
without an attachment header. The publisher now defaults to this delivery origin
and does not set public ACLs. Browser playback and framing review are tracked
separately from continuous human quality acceptance.

## Browser and framing checks

Edge played all 66 internal-CDN videos beyond three seconds with 1344x768 decoded
geometry and no media error. Reveal/hide and reshuffle checks passed. Twelve
three-lane captures at approximately three seconds were visually inspected.
Cases 11–14 contain the intended closeup controls. Cases 17, 20 and 22 clearly
contain small faces in wide/full-body compositions. Cases 15, 16, 18, 19 and 21
provide medium-sized faces; their framing is tighter than the requested fractions
and they must not be counted as strict tiny-face tests. Case 18 also has smaller
background faces. Case 21's waving hand is blurred in the captured frame.
These are static framing observations, not full temporal quality acceptance;
video-control overlays obscure the lower strip and are excluded from review.
