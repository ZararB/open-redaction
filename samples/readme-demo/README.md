# README demo frames

Side-by-side stills in the root README:

| File | Source |
|------|--------|
| `frame_input.jpg` | Same still with manually placed opaque masks over visible faces for the public README preview |
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
filter='scale=800:450'
# Coordinates are x:y:width:height on the 800x450 README preview.
for box in 0:197:55:89 51:203:53:66 108:181:41:55 158:161:62:104 \
           222:185:38:52 283:125:44:85 335:151:39:58 393:173:39:49 \
           432:141:63:92 494:173:40:48 531:91:72:114 593:140:43:74 \
           628:151:39:64 659:152:30:66 680:136:38:73 765:155:35:73; do
  IFS=: read -r x y w h <<EOF
$box
EOF
  filter="$filter,drawbox=x=$x:y=$y:w=$w:h=$h:color=black:t=fill"
done
ffmpeg -hide_banner -loglevel error -y -i "$IMG" -vf "$filter" \
  -frames:v 1 -map_metadata -1 -q:v 3 -update 1 samples/readme-demo/frame_input.jpg
```

Video caches stay gitignored; only the two JPEGs are committed for GitHub rendering.
