#%%

import time
import numpy as np

from skimage import io

import tensorflow as tf

#%%

from core.functions import UNetCompiled

#%% Check GPUs

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Initialize

''' 1) Get paths '''

DATA_PATH = 'data_epicell/'
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

''' 3) Reshape and format data '''
raw = raw.reshape((raw.shape[0], raw.shape[1], raw.shape[2], 1)) 
test = test.reshape((test.shape[0], test.shape[1], test.shape[2], 1))
mask = mask.reshape((mask.shape[0], mask.shape[1], mask.shape[2], 1))

raw = raw/65535
test = test/65535
mask = mask.astype('bool')

#%%

 
unet = UNetCompiled(input_size=(128,128,1), n_filters=16, n_classes=1)

unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]

results = unet.fit(raw, mask, validation_split=0.2, batch_size=64, epochs=1000, callbacks=callbacks)

test_mask = unet.predict(test, verbose=1)

test_mask = test_mask[:,:,:,0]

#%%

io.imsave(TEMP_PATH+'test.tif', test.astype("float32"), check_contrast=False)  
io.imsave(TEMP_PATH+'mask.tif', mask.astype("uint8"), check_contrast=False)   
io.imsave(TEMP_PATH+'test_mask.tif', test_mask.astype("float32"), check_contrast=False)   




