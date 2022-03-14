#%%

import os
import time
import random
import numpy as np
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from tensorflow.keras.utils import normalize

from skimage import io

#%%

from tools.dtype import as_uint8
from core.functions import data_augmentation

#%% Augmentation parameters

AUG = False
ITER = 1000

# Define operations 
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5)
    ]
)

#%% Check GPUs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Initialize

''' 1) Get paths '''

DATA_PATH = 'data_RBCs/'
TEMP_PATH = DATA_PATH + 'temp/'
RAW_NAME = '210218_1600_RSize_REG(1-2000).tif'
MASK_NAME = '210218_1600_RSize_rawREGMask(1-2000).tif'
TEST_NAME = '210218_1600_RSize_REG(2001-3000).tif'
TEST_NAME = '210218_1600_Expl-02_RSize_REG.tif'

''' 2) Open data '''

raw = io.imread(DATA_PATH + RAW_NAME)
test = io.imread(DATA_PATH + TEST_NAME) 
mask = io.imread(DATA_PATH + MASK_NAME)

''' 3) Format data '''

# Convert to uint8
raw = as_uint8(raw, 0.999)
test = as_uint8(test, 0.999)
mask = as_uint8(mask, 0.999)

# Check for xy size
min_size = min([raw.shape[1], raw.shape[2]])
raw = raw[:,0:min_size,0:min_size]
mask = mask[:,0:min_size,0:min_size]
test = test[:,0:min_size,0:min_size]

# Normalize data
raw = normalize(raw)
test = normalize(test)
mask = mask/255

#%% Data augmentation

if AUG:

    raw, mask = data_augmentation(
        raw, mask, 
        operations, 
        iterations=1000, 
        parallel=False
        )
        
    
#%% Train model

BACKBONE = 'resnet34'

# Define model using segmentation_models
model = sm.Unet(
    BACKBONE, 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None
    )

# Compile model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Train the model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
history = model.fit(
    raw, mask, 
    validation_split=0.2,
    batch_size=8, 
    epochs=30, 
    callbacks=callbacks, 
    verbose=1)

# Plot training results
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()