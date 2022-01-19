#%%

import random
import numpy as np
from joblib import Parallel, delayed 

#%% Data augmentation

def _data_augmentation(raw, mask, operations):
    
    rand = random.randint(0, raw.shape[0]-1)
    outputs = operations(image=raw[rand,:,:], mask=mask[rand,:,:])
        
    raw_aug = outputs['image']
    mask_aug = outputs['mask']
    
    return raw_aug, mask_aug

''' ........................................................................'''

def data_augmentation(raw, mask, operations, iterations=256, parallel=True):
    
    if parallel:
 
        # Run _data_augmentation (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_data_augmentation)(
                raw,
                mask,
                operations
                )
            for i in range(iterations)
            )
            
    else:
            
        # Run _data_augmentation
        output_list = [_data_augmentation(
                raw,
                mask,
                operations
                ) 
            for i in range(iterations)
            ]

    # Extract outputs
    raw_augmented = np.stack([arrays[0] for arrays in output_list], axis=0)
    mask_augmented = np.stack([arrays[1] for arrays in output_list], axis=0)
    
    return raw_augmented, mask_augmented