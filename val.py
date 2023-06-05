import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('runs/detect/train6/weights/best.pt')

model.val("bambi.yaml", imgsz=640)