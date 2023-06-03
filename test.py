import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('runs/detect/train6/weights/best.pt')

model.predict('datasets/test', save=True, imgsz=640, conf=0.3, line_width=1)