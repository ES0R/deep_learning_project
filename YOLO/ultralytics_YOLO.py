from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='coco8.yaml', epochs=10, imgsz=256)