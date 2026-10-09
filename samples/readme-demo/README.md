# README demo frames

Side-by-side stills in the root README:

| File | Source |
|------|--------|
| `frame_input.jpg` | One frame from `datasets/privacy_redaction/nyc_times_square_45s/frames/nyc_0015.png` |
| `frame_output.jpg` | Same frame after `open-redact` (`FACE_BLUR_DETECTOR=max`, opaque face boxes) |

Regenerate:

```bash
cd ../..  # repo root
source .venv/bin/activate
IMG=datasets/privacy_redaction/nyc_times_square_45s/frames/nyc_0015.png
ffmpeg -y -loop 1 -i "$IMG" -t 2 -r 10 -pix_fmt yuv420p samples/readme-demo/input.mp4
export FACE_BLUR_DETECTOR=max FACE_BLUR_STYLE=blackout FACE_BLUR_BYTE_TRACK=1
open-redact samples/readme-demo/input.mp4 -o samples/readme-demo/output.redacted.mp4
ffmpeg -ss 0.5 -i samples/readme-demo/input.mp4 -frames:v 1 /tmp/in.png
ffmpeg -ss 0.5 -i samples/readme-demo/output.redacted.mp4 -frames:v 1 /tmp/out.png
ffmpeg -i /tmp/in.png -vf scale=800:-1 samples/readme-demo/frame_input.jpg -q:v 3
ffmpeg -i /tmp/out.png -vf scale=800:-1 samples/readme-demo/frame_output.jpg -q:v 3
```

Video caches stay gitignored; only the two JPEGs are committed for GitHub rendering.
