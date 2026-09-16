#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 INPUT.mp4 OUTPUT.mp4" >&2
  exit 2
fi

readonly input="$1"
readonly output="$2"
if [[ ! -f "$input" ]]; then
  echo "Input does not exist: $input" >&2
  exit 2
fi
if [[ "$input" == "$output" ]]; then
  echo "Input and output must be different files" >&2
  exit 2
fi

ffmpeg_bin="${FFMPEG_BIN:-}"
if [[ -z "$ffmpeg_bin" ]]; then
  ffmpeg_bin="$(command -v ffmpeg || true)"
fi
if [[ -z "$ffmpeg_bin" ]]; then
  ffmpeg_bin="$(python3 - <<'PY'
try:
    import imageio_ffmpeg
except ImportError:
    pass
else:
    print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
fi
if [[ ! -x "$ffmpeg_bin" ]]; then
  echo "ffmpeg not found; install ffmpeg or set FFMPEG_BIN" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$output")"
"$ffmpeg_bin" -v error -y -i "$input" \
  -map 0:v:0 -map '0:a?' \
  -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=auto:out_range=tv:out_color_matrix=bt709,format=yuv420p,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709' \
  -c:v libx264 -profile:v high -pix_fmt yuv420p \
  -preset "${WEB_VIDEO_PRESET:-medium}" -crf "${WEB_VIDEO_CRF:-18}" \
  -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a "${WEB_VIDEO_AUDIO_BITRATE:-192k}" \
  -movflags +faststart \
  "$output"

echo "Wrote browser-compatible MP4: $output"
