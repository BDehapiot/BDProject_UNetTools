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

from skimage import io


#%%

from tools.dtype import as_uint8
from core.functions import data_augmentation

#%% Check GPUs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Initialize

''' 1) Get paths '''

DATA_PATH = 'data_GBE/'
TEMP_PATH = DATA_PATH + 'temp/'
RAW_NAME = '18-05-29_GBE_67xYW(F2)_128x128_raw.tif'
MASK_NAME = '18-05-29_GBE_67xYW(F2)_128x128_mask.tif'
TEST_NAME = '18-05-29_GBE_67xYW(F2)_128x128_test.tif'

# DATA_PATH = 'data_RICM/'
# TEMP_PATH = DATA_PATH + 'temp/'
# RAW_NAME = 'Cells_expl_01_sStack-10_sStack-10_raw.tif'
# MASK_NAME = 'Cells_expl_01_sStack-10_sStack-10_mask.tif'
# TEST_NAME = 'Cells_expl_01_sStack-10_test.tif'

''' 2) Open data '''

raw = io.imread(DATA_PATH + RAW_NAME)
test = io.imread(DATA_PATH + TEST_NAME) 
mask = io.imread(DATA_PATH + MASK_NAME)

''' 3) Convert data to uint8 '''

# raw = as_uint8(raw, 0.999)
# test = as_uint8(test, 0.999)
# mask = as_uint8(mask, 0.999)

#%% Data augmentation (albumentation)

# Define operations 
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5)
    ]
)

start = time.time()
print('Data augmentation')

raw_augmented, mask_augmented = data_augmentation(
    raw, mask, operations, iterations=256, parallel=False)
    
end = time.time()
print(f'  {(end - start):5.3f} s')  

# io.imsave(TEMP_PATH+'raw_augmented.tif', raw_augmented.astype("uint8"), check_contrast=False)  
# io.imsave(TEMP_PATH+'mask_augmented.tif', mask_augmented.astype("uint8"), check_contrast=False)   

#%%

BACKBONE = 'resnet34'

''' ........................................................................'''

# Define model
model = sm.Unet(BACKBONE, input_shape=(None, None, 1), classes=1, activation='sigmoid', encoder_weights=None)
# model.compile(optimizer='Adam', loss=sm.losses.bce_jaccard_loss, metrics=['sm.metrics.iou_score'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

''' ........................................................................'''

# Preprocess inputs
preprocess_input = sm.get_preprocessing(BACKBONE)
raw_augmented = preprocess_input(raw_augmented)
test = preprocess_input(test)

''' ........................................................................'''

# from tensorflow.keras.utils import normalize
# raw_augmented = normalize(raw_augmented)
# mask_augmented = mask_augmented/255

''' ........................................................................'''

# Train the model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
history = model.fit(
    raw_augmented, 
    mask_augmented, 
    validation_split=0.2,
    batch_size=16, 
    epochs=100, 
    callbacks=callbacks, 
    verbose=1)

# Plot
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

''' ........................................................................'''

# Predict
prediction = model.predict(test)

# Save
io.imsave(TEMP_PATH+'prediction.tif', prediction.astype("float32"), check_contrast=False)  
