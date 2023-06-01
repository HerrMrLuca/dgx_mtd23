import ultralytics
ultralytics.checks()

# start train
# yolo task=detect mode=train epochs=20 data=bambi.yaml model=yolov8n.pt imgsz=640 batch=8 patience=150