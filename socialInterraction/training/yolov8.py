import comet_ml
import sys
from ultralytics import YOLO



model = YOLO('yolov8m.pt') 

results = model.train(data='../training/data.yaml', epochs=50, imgsz=640, name="First-Run", batch=8, save_period=10,device='mps', rect=True, workers=0)