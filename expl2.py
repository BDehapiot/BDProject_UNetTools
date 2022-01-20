#%%

import os
import time
import random
import numpy as np
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt

from tensorflow.keras.utils import normalize

from skimage import io

#%%

from tools.dtype import as_uint8
from core.functions import simple_unet_model
from core.functions import data_augmentation

#%% Check GPUs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Initialize

''' 1) Get paths '''

# DATA_PATH = 'data_GBE/'
# TEMP_PATH = DATA_PATH + 'temp/'
# RAW_NAME = '18-05-29_GBE_67xYW(F2)_128x128_raw.tif'
# MASK_NAME = '18-05-29_GBE_67xYW(F2)_128x128_mask.tif'
# TEST_NAME = '18-05-29_GBE_67xYW(F2)_128x128_test.tif'

# DATA_PATH = 'data_RICM/'
# TEMP_PATH = DATA_PATH + 'temp/'
# RAW_NAME = 'Cells_expl_01_sStack-10_sStack-10_raw.tif'
# MASK_NAME = 'Cells_expl_01_sStack-10_sStack-10_mask.tif'
# TEST_NAME = 'Cells_expl_01_sStack-10_test.tif'

DATA_PATH = 'data_MitoEM/'
TEMP_PATH = DATA_PATH + 'temp/'
RAW_NAME = 'MitoEM_EPFL_raw_01.tif'
MASK_NAME = 'MitoEM_EPFL_mask_01.tif'
TEST_NAME = 'MitoEM_EPFL_raw_02.tif'

''' 2) Open data '''

raw = io.imread(DATA_PATH + RAW_NAME)
test = io.imread(DATA_PATH + TEST_NAME) 
mask = io.imread(DATA_PATH + MASK_NAME)

''' 3) Format data '''

# # Convert to uint8
# raw = as_uint8(raw, 0.999)
# test = as_uint8(test, 0.999)
# mask = as_uint8(mask, 0.999)

# Check for xy size
# min_size = min([raw.shape[1], raw.shape[2]])
min_size = 128
raw = raw[:,0:min_size,0:min_size]
mask = mask[:,0:min_size,0:min_size]
test = test[:,0:min_size,0:min_size]

# Normalize data
raw = normalize(raw)
test = normalize(test)
mask = mask/255

# Augment data

# Define operations 
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    # A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.Transpose(p=0.5),
    A.GridDistortion(p=0.5)
    ]
)

raw, mask = data_augmentation(
    raw, mask, operations, iterations=1000, parallel=False)

#%%

#%%

# Define the model
model = simple_unet_model(min_size, min_size, 1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

model_path = TEMP_PATH+'weights_epoch({epoch:02d})_vloss({val_loss:.2f}).hdf5'

checkpoint = ModelCheckpoint(
    model_path, monitor='val_loss', save_best_only=True, mode='min')
early_stop = EarlyStopping(
    monitor='val_loss', patience=20, verbose=1)
log_csv = CSVLogger(
    'my_logs.csv', separator=',', append=False)

callbacks_list = [checkpoint, early_stop, log_csv]

# Train model
history = model.fit(
    raw, mask, 
    validation_split=0.3,
    batch_size=64, 
    epochs=100, 
    callbacks=callbacks_list, 
    verbose=1)

# Evaluate model
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

# Predict
prediction = model.predict(test, verbose=1)

# Save
io.imsave(TEMP_PATH+'prediction.tif', prediction.astype("float32"), check_contrast=False)  