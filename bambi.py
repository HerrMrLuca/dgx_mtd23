

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/last.pt')

results = model.train(data='bambi.yaml', epochs=50000, batch=8, imgsz=640, patience=20000, save_period=50, resume=True)

#model = YOLO('yolov8n.pt')

#results = model.train(data='bambi.yaml', epochs=50000, batch=8, imgsz=640, patience=20000, save_period=500)