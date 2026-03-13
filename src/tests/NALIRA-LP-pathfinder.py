#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:16:42 2026

@author: nico
NALIRA LOW POWER / NALIRA-LP / LOW POWER NALIRA
"""
def split_array(array, n_chunks):
    rows = np.array_split(array, np.sqrt(n_chunks))#, axis=0) # split into rows
    split_arrays = [np.array_split(row_chunk, np.sqrt(n_chunks), 
                                   axis=1) for row_chunk in rows]
    chunks = [subarray for row_chunk in split_arrays for subarray in row_chunk]
    return chunks

# this pathfinder script is to validate the improved "low-power" pipeline for NALIRA. it changes the order of operations. 
# instead of conducting large and computationally expensive operations on very large matrices (entire images), it conducts those same operations on significantly smaller matrices (chunks, not entire images). 
# for example, instead of conducting omnicloudmask inference on an entire satellite image and then creating smaller labelling chunks, NALIRA-LP-pathfinder is designed to create the small labelling chunks and then apply omnicloudmask inference to the tiny chunk. 
# chunks are ~127x127 pixels and a full satellite image is ~11000x11000
# the total duration of omnicloudmask if you were to go chunk-by-chunk or on the entire image in one sweep is the same; if anything, chunk-by-chunk is probably slower. 
# however, this NALIRA-LP approach could be much faster for data labelling, where you are only looking at one chunk at a time. 
# instead of having to wait over an hour (on integrated graphics) for omnicloudmask inference on a full satellite image, you can wait a fraction of a second for the inference on a tiny labelling chunk instead. 
# this effectively makes the data labelling pipeline significantly more accessible for low-power computers. 

import numpy as np
import omnicloudmask as ocm
import torch
import time
import matplotlib.pyplot as plt
torch.set_num_threads(12) # number of threads on dell xps 9315

# %% create image array(s)
start_time = time.monotonic()

img1 = np.random.rand(3, 11000, 11000).astype(np.float32)
img2 = np.random.rand(3, 11000, 11000).astype(np.float32)
img2_60 = np.random.rand(3, 2000, 2000).astype(np.float32)
img_list = [img1, img2]

print(f"step 1 complete: {round(time.monotonic()-start_time,2)}")

# %% chunkification
start_time = time.monotonic()

img_chunks_list = []

for img in img_list:
    img_chunks_list_band = []
    for band in img:
        img_chunks_list_band.append(split_array(array=band, n_chunks=5000))
    img_chunks_list.append(img_chunks_list_band)

print(f"step 2 complete: {round(time.monotonic()-start_time,2)}")

# %% labelling loop
start_time = time.monotonic()

np.seterr(divide="ignore", invalid="ignore")
i = 0
while i < 5:
    mask = []
    ndwi = []
    ndvi = []
    for img_chunks in img_chunks_list:
        img_chunk = img_chunks[i]
        
        # ==== mask clouds ==== #
        mask.append(
            ocm.predict_from_array(
                img_chunk[i], patch_size=127, 
                patch_overlap=64, 
                inference_device="cpu", 
                inference_dtype="fp32"
                )[0]
            )
        
        # ==== mask known features ==== #
        pass
        
        # ==== calculate indices ==== #
        green, nir, red = img_chunk[i]
        ndwi = (green - nir) / (green + nir)
        ndvi = ((nir - red) / (nir + red))
        index_arrays = {"ndwi": ndwi, "ndvi": ndvi}
        
        # ==== spectral temporal metrics ==== #
        stms = [{}]
        for index_name, arrays_list in index_arrays.items():
            stack = np.stack(arrays_list)
            mean, p25, median, p75 = np.percentile(stack, [0, 25, 50, 75], axis=0)
            stms[index_name].append({
                "p25": p25, 
                "median": median, 
                "p75": p75, 
                "mean": mean
                })
    
    # ==== display chunks ==== #
    ndwi_mean = stms["ndwi"]["median"]
    plt.plot(ndwi_mean)

print(f"step 3 complete: {round(time.monotonic()-start_time,2)}")

