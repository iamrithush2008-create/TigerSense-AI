from ultralytics import YOLO
model = YOLO("best.pt")
r = model("sample_data/tiger1.jpg")[0]
print(len(r.boxes), "tiger(s) detected")
assert len(r.boxes) >= 1