# README demo frames

Side-by-side stills in the root README:

| File | Source |
|------|--------|
| `frame_input.jpg` | Same still, but **only the two largest YuNet face boxes** are manually blacked out for public README (not a real pipeline input) |
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
python3 - <<'PY'
# Public input preview: black out the two largest face detections only.
import cv2
from pathlib import Path
from agents.ml.face_blur.factory import build_detector
src = Path("datasets/privacy_redaction/nyc_times_square_45s/frames/nyc_0015.png")
frame = cv2.imread(str(src))
boxes = sorted(build_detector("opencv", 0.3).detect(frame), key=lambda b: int(b[2])*int(b[3]), reverse=True)
for x,y,w,h in (tuple(int(v) for v in b[:4]) for b in boxes[:2]):
    cv2.rectangle(frame, (max(0,x-int(w*0.15)), max(0,y-int(h*0.2))), (min(frame.shape[1],x+w+int(w*0.15)), min(frame.shape[0],y+h+int(h*0.2))), (0,0,0), -1)
h,w=frame.shape[:2]; frame=cv2.resize(frame,(800,int(h*800/w)))
cv2.imwrite("samples/readme-demo/frame_input.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),88])
PY
```

Video caches stay gitignored; only the two JPEGs are committed for GitHub rendering.
