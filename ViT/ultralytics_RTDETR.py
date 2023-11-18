from ultralytics import RTDETR
import pandas as pd
from PIL import Image


# Load a COCO-pretrained RT-DETR-l model
model = RTDETR('rtdetr-l.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset
results = model.train(data='coco8.yaml', epochs=1, imgsz=640, iou=0.9)

# Forward pass
results = model('best_electric_luxury_car_bmw_i7.jpg')

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save('results.jpg')

