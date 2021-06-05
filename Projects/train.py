import os, sys, random, warnings, math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from tqdm.auto import tqdm, trange
from itertools import chain

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

from tensorflow.keras import models, Input, callbacks
from model import build_model
from utils import iou_metric_batch

class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = 'competition_data/train/'
    path_test = 'competition_data/test/'

train_ids = next(os.walk(config.path_train+"images"))[2]
test_ids = next(os.walk(config.path_test+"images"))[2]

X = np.zeros((len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8)
Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    x = img_to_array(load_img(config.path_train + '/images/' + id_, color_mode="grayscale"))
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X[n] = x
    mask = img_to_array(load_img(config.path_train + '/masks/' + id_, color_mode="grayscale"))
    Y[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

print('Done!')
print('X shape:', X.shape)
print('Y shape:', Y.shape)

X_train = X[:int(0.9*len(X))]
Y_train = Y[:int(0.9*len(X))]
X_eval  = X[int(0.9*len(X)):]
Y_eval  = Y[int(0.9*len(X)):]

X_train = np.append(X_train, [np.fliplr(x) for x in X], axis=0)
Y_train = np.append(Y_train, [np.fliplr(x) for x in Y], axis=0)
X_train = np.append(X_train, [np.flipud(x) for x in X], axis=0)
Y_train = np.append(Y_train, [np.flipud(x) for x in Y], axis=0)

print('X train shape:', X_train.shape, 'X eval shape:', X_eval.shape)
print('Y train shape:', Y_train.shape, 'Y eval shape:', Y_eval.shape)



input_layer = Input((config.im_height, config.im_width, config.im_chan))
output_layer = build_model(input_layer, 16)


model = models.Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    X_train, Y_train, validation_data=(X_eval, Y_eval), batch_size=8, epochs=1, callbacks=[es, rlp]
)

preds_eval = model.predict(X_eval, verbose=1)

thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(Y_eval, np.int32(preds_eval > threshold)) for threshold in tqdm(thresholds)])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

print(iou_best)
print(threshold_best)
