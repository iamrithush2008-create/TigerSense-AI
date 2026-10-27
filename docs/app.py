from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import numpy as np, cv2

app = FastAPI()
model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), 1)
    r = model(img, conf=0.4)[0]
    return {"count": len(r.boxes),
            "detections": [{"label": model.names[int(b.cls)],
                            "conf": float(b.conf)} for b in r.boxes]}