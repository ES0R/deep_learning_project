# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:15:03 2023

@author: hujo8
"""

import pathlib
import shutil
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import selectivesearch



data_dir_path = "data3"

data_dir = pathlib.Path(data_dir_path).with_suffix('')

image_files = list(data_dir.glob('*.jpg'))
print(len(image_files))


#%%

sample_img_path = str(image_files[103])
print(sample_img_path)
sample_img = cv.imread(sample_img_path, cv.IMREAD_COLOR)
plt.imshow(cv.cvtColor(sample_img, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

sample_annotation_path = sample_img_path[0:-4] + '.txt'
with open(sample_annotation_path) as f:
    print(f.readlines())


#%%

import os, random

def list_files(full_data_path = "data3/", image_ext = '.jpg', split_percentage = [70, 20]):
    

    files = []

    discarded = 0
    masked_instance = 0

    for r, d, f in os.walk(full_data_path):
        for file in f:
            if file.endswith(".txt"):

                with open(full_data_path + "/" + file, 'r') as fp: 
                    lines = fp.readlines()
                    if len(lines) > 1:
                        discarded += 1
                        continue
                    

                strip = file[0:len(file) - len(".txt")]  
                image_path = full_data_path + "/" + strip + image_ext
                if os.path.isfile(image_path):
                    if lines[0][0] == '0':
                        masked_instance += 1
                    files.append(strip)

    size = len(files)   
    
    random.shuffle(files)

    split_training = int(split_percentage[0] * size / 100)
    split_validation = split_training + int(split_percentage[1] * size / 100)

    return files[0:split_training], files[split_training:split_validation], files[split_validation:]

training_files, validation_files, test_files = list_files()

print(str(len(training_files)) + " training files")
print(str(len(validation_files)) + " validation files")
print(str(len(test_files)) + " test files")


#%%

# Load the image and define the bounding box

temp_img = cv.imread("data3/000000000094.jpg", cv.IMREAD_GRAYSCALE)
temp_box = [int(356.23), int(275.22), int(42.87), int(38.46)]

   
# Convert the image to color
temp_color_img = cv.cvtColor(temp_img, cv.COLOR_GRAY2RGB)

# Draw the original bounding box on the image
cv.rectangle(temp_color_img, (temp_box[0], temp_box[1]), (temp_box[0] + temp_box[2], temp_box[1] + temp_box[3]), (0, 255, 0), 2)

# Display the original image with the bounding box
plt.imshow(temp_color_img)
plt.axis("off")
plt.title("Original Image with Bounding Box")
plt.show()


input_size = 256

def format_image(img, box):
    height, width = img.shape 
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    x, y, w, h = box[0], box[1], box[2], box[3]
    
    x = (x/width)+0.5*w/width
    y = (y/height)+0.5*h/height
    w = w/width
    h = h/height
    
    print(x, y, w, h)

    new_box = [int((x - 0.5*w)* width / r), int((y - 0.5*h) * height / r), int(w*width / r), int(h*height / r)]
    return new_image, new_box


temp_img = cv.imread("data3/000000000094.jpg", cv.IMREAD_GRAYSCALE)
temp_box = [int(356.23), int(275.22),int(42.87), int(38.46)]

temp_img_formated, box = format_image(temp_img, temp_box)

temp_color_img = cv.cvtColor(temp_img_formated, cv.COLOR_GRAY2RGB)

cv.rectangle(temp_color_img, box, (0, 255, 0), 2)

plt.imshow(temp_color_img)
plt.axis("off")
plt.show()


#%%

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging

# uncomment the following line if you want to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.__version__)


#%%


def data_load(files, full_data_path = "data3/", image_ext = ".jpg"):
    X = []
    Y = []

    for file in files:
        img_path = os.path.join(full_data_path, file + image_ext)

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        k = 0

        with open(full_data_path + "/" + file + ".txt", 'r') as fp: 
            line = fp.readlines()[0]
            k = line[0]

            box = np.array(line[1:].split(), dtype=float)

        img, box = format_image(img, box)
        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / input_size
        label = np.append(box, k)

        X.append(img)
        Y.append(label)

    X = np.array(X)
    
    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    result = tf.data.Dataset.from_tensor_slices((X, Y))

    return result

#%%

raw_train_ds = data_load(training_files)
raw_validation_ds = data_load(validation_files)
raw_test_ds = data_load(test_files)

#%%

CLASSES = 32

def format_instance(image, label):
    return image, (tf.one_hot(int(label[4]), CLASSES), [label[0], label[1], label[2], label[3]])

BATCH_SIZE = 32

# see https://www.tensorflow.org/guide/data_performance

def tune_training_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat() # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = tune_training_ds(raw_train_ds)

def tune_validation_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(len(validation_files) // 4)
    dataset = dataset.repeat()
    return dataset

validation_ds = tune_validation_ds(raw_validation_ds)

#%%


plt.figure(figsize=(20, 10))
for images, labels in train_ds.take(1):
    for i in range(BATCH_SIZE):
        ax = plt.subplot(4, BATCH_SIZE//4, i + 1)
        label = labels[0][i]
        box = (labels[1][i] * input_size)
        box = tf.cast(box, tf.int32)

        image = images[i].numpy().astype("float") * 255.0
        image = image.astype(np.uint8)
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        color = (0, 0, 255)
        if label[0] > 0:
            color = (0, 255, 0)

        cv.rectangle(image_color, box.numpy(), color, 2)

        plt.imshow(image_color)
        plt.axis("off")


#%%

DROPOUT_FACTOR = 0.5

def selective_search(image):
    img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=10)
    candidates = set()

    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 1000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    return candidates


def build_resnet_block(x, filters, kernel_size=3, stride=1):
    # Shortcut
    shortcut = x

    # First convolution layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second convolution layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x

def build_feature_extractor(inputs):
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu', input_shape=(input_size, input_size, 1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # ResNet blocks
    x = build_resnet_block(x, 64)
    x = build_resnet_block(x, 64)
    x = build_resnet_block(x, 128, stride=2)
    x = build_resnet_block(x, 128)
    x = build_resnet_block(x, 256, stride=2)
    x = build_resnet_block(x, 256)
    x = build_resnet_block(x, 512, stride=2)
    x = build_resnet_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    return x

def build_model_adaptor(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return x

def build_classifier_head(inputs):
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputs)
    return tf.keras.layers.Dense(CLASSES, activation='softmax', name = 'classifier_head')(x)

def build_regressor_head(inputs):
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputs)
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    return tf.keras.layers.Dense(units = '4', name = 'regressor_head')(inputs)

def build_model(inputs):
    
    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(model_adaptor)

    model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss = {'classifier_head' : 'categorical_crossentropy', 'regressor_head' : 'mse' }, 
              metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })

    return model


model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))

model.summary()


import json

#%%

EPOCHS = 100

history = model.fit(train_ds,
                    steps_per_epoch=(len(training_files) // BATCH_SIZE),
                    validation_data=validation_ds, validation_steps=1, 
                    epochs=EPOCHS)

model.save('object_detection_model.h5')




# Save history to a JSON file
with open('training_history.json', 'w') as json_file:
    json.dump(history.history, json_file)



#%%

"""
model = tf.keras.models.load_model('object_detection_model2.h5')

with open('training_history2.json', 'r') as json_file:
    history = json.load(json_file)

"""

#%%

plt.plot(history['classifier_head_accuracy'])
plt.plot(history['val_classifier_head_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#%%
plt.plot(history['classifier_head_loss'])
plt.plot(history['val_classifier_head_loss'])
plt.title('Classification Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%%

plt.plot(history['regressor_head_loss'])
plt.plot(history['val_regressor_head_loss'])
plt.title('Bounding Box Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#%%

n = 10


# adapted from: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def tune_test_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    dataset = dataset.repeat()
    return dataset

test_ds = tune_test_ds(raw_test_ds)


import matplotlib.pyplot as plt

def plot_predictions(model, test_ds):
    # Take one batch from the test dataset
    for images, labels in test_ds.take(n):
        image = images[0].numpy().astype("float") * 255.0
        image = image.astype(np.uint8)
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        # Make predictions using the model
        predictions = model.predict(images)

        # Extract bounding box information from predictions
        predicted_box = predictions[1][0] * input_size
        predicted_box = tf.cast(predicted_box, tf.int32)

        # Extract ground truth bounding box information
        true_box = labels[1][0] * input_size
        true_box = tf.cast(true_box, tf.int32)

        # Draw predicted bounding box in red
        color = (255, 0, 0)
        cv.rectangle(image_color, predicted_box.numpy(), color, 2)

        # Draw true bounding box in green
        color = (0, 255, 0)
        print(predicted_box.numpy())
        cv.rectangle(image_color, true_box.numpy(), color, 2)

        # Display the image with predictions and true bounding box
        plt.imshow(image_color)
        plt.axis("off")
        plt.title("Predicted and True Bounding Boxes")
        plt.show()

# Call the function to plot predictions
plot_predictions(model, test_ds)




