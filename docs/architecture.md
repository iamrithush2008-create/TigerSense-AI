# Architecture
Camera trap / UAV video -> YOLOv8 detection -> per-frame count
-> (planned) health check on tiger crop -> FastAPI /detect
-> TigerSense AI dashboard -> alerts to rangers and vets

Tech: Python, Ultralytics YOLOv8, Roboflow, FastAPI, Google Colab (T4).