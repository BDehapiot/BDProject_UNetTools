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

#%% Check GPUs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Parameters

''' 1) Get paths '''
DATA_PATH = 'data_RBCs/'
TEMP_PATH = DATA_PATH + 'temp/'

''' 2) Augmentation '''
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

#%% Initialize

''' 3) Open & format data '''
dirlist = os.listdir(DATA_PATH)
for name in dirlist:
    if 'mask' in name:
        
        # Open data
        temp_mask = io.imread(DATA_PATH + name)
        temp_raw = io.imread(DATA_PATH + name[0:-8]+'raw.tif')           
        
        # Format data
        temp_raw = normalize(temp_raw)
        temp_mask = temp_mask/255
                
        if 'mask' not in locals():
            mask = temp_mask
            raw = temp_raw          
        else:
            mask = np.append(mask, temp_mask, axis=0)
            raw = np.append(raw, temp_raw, axis=0)  
                       
# io.imsave(TEMP_PATH+'mask.tif', mask.astype("uint8"), check_contrast=False)          
# io.imsave(TEMP_PATH+'raw.tif', (raw*255).astype("uint8"), check_contrast=False)  

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

#%% Predict

test = io.imread(DATA_PATH + 'Expl_04_noREG_raw.tif')
test = normalize(test)
pred = model.predict(test)

# Save
io.imsave(TEMP_PATH+'pred.tif', pred.astype("float32"), check_contrast=False)  
