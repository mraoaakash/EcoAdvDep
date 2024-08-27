import comet_ml
import sys
from ultralytics import YOLO





# export COMET_API_KEY=NiIALx9i86acjJL53ck558RCD
# export COMET_WORKSPACE=zebrafish2
# export COMET_PROJECT_NAME=zebrafish-tracking





model = YOLO('yolov8m.pt') 

results = model.train(data='/Users/mraoaakash/Documents/research/research-arpi/Social-interraction/algorithms/training/data.yaml', epochs=50, imgsz=640, name="First-Run", batch=8, save_period=10,device='mps', rect=True, workers=0)