# README demo frames

Side-by-side stills in the root README:

| File | Source |
|------|--------|
| `frame_input.jpg` | Same still with three manually placed opaque face masks for the public README preview |
| `frame_output.jpg` | Full source frame after `open-redact` (`FACE_BLUR_DETECTOR=max`, opaque face boxes on all detections) |

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
ffmpeg -i /tmp/out.png -vf scale=800:-1 samples/readme-demo/frame_output.jpg -q:v 3
# Three fixed masks on the 800x450 preview. The last covers the woman in red.
ffmpeg -hide_banner -loglevel error -y -i "$IMG" \
  -vf 'scale=800:450,drawbox=158:161:62:104:black:fill,drawbox=531:91:72:114:black:fill,drawbox=597:145:37:63:black:fill' \
  -frames:v 1 -map_metadata -1 -q:v 3 -update 1 samples/readme-demo/frame_input.jpg
```

Video caches stay gitignored; only the two JPEGs are committed for GitHub rendering.
