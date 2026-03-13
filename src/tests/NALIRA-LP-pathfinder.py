#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:16:42 2026

@author: nico
NALIRA LOW POWER / NALIRA-LP / LOW POWER NALIRA
Low-power variant of NALIRA. Opens images normally, then immediately splits
each full image into chunks. All processing (cloud masking, known feature
masking, index calculation, compositing) is done chunk-by-chunk so that
machines without discrete GPUs (e.g. laptops) can run the data labelling
portion of NALIRA without holding entire processed images in memory.
"""


import numpy as np
import omnicloudmask as ocm
import torch
import time
import matplotlib.pyplot as plt

torch.set_num_threads(12) # number of threads on dell xps 9315

def split_array(array, n_chunks):
    rows = np.array_split(array, np.sqrt(n_chunks))#, axis=0) # split into rows
    split_arrays = [np.array_split(row_chunk, np.sqrt(n_chunks), 
                                   axis=1) for row_chunk in rows]
    chunks = [subarray for row_chunk in split_arrays for subarray in row_chunk]
    return chunks

# %% create image array(s)
start_time = time.monotonic()

img1 = np.random.rand(3, 11000, 11000).astype(np.float32)
img2 = np.random.rand(3, 11000, 11000).astype(np.float32)
img2_60 = np.random.rand(3, 2000, 2000).astype(np.float32)
img_list = [img1, img2]

print(f"step 1 complete: {round(time.monotonic()-start_time,2)}")

# %% chunkification
start_time = time.monotonic()

img_chunks_list = [] # will be shape: [n_images][n_bands][n_chunks][h][w]
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
    ndwi_per_chunk = []
    ndvi_per_chunk = []
    
    chunk_stms = { "ndwi": [], "ndvi": [] }
    
    for img_chunks in img_chunks_list:
        # ==== mask known features ==== #
        pass
        
        # ==== create image stacks ==== #
        red = img_chunks[2][i]
        green = img_chunks[0][i]
        nir = img_chunks[1][i]
        
        img_chunk_stack = np.stack((
            red, 
            green, 
            nir
            ))
        
        print("bands stacked")
        
        # ==== mask clouds ==== #
        pred_mask_2d = ocm.predict_from_array(
            img_chunk_stack, 
            patch_size=157, patch_overlap=64, 
            inference_device="cpu", inference_dtype="fp32"
            )[0]
        
        combined_mask = (
            (pred_mask_2d == 1) | 
            (pred_mask_2d == 2) | 
            (pred_mask_2d == 3)
            )
        
        # float is used as it supports NaN
        red[combined_mask] = np.nan
        green[combined_mask] = np.nan
        nir[combined_mask] = np.nan
        
        print("clouds masked")
        
        # ==== calculate indices ==== #
        ndwi = (green - nir) / (green + nir)
        ndvi = ((nir - red) / (nir + red))
        
        ndwi_per_chunk.append(ndwi)
        ndvi_per_chunk.append(ndvi)
        
        print("indices calculated")
        
    index_chunk_arrays = {"ndwi": ndwi_per_chunk, "ndvi": ndvi_per_chunk}
    
    # ==== spectral temporal metrics ==== #
    for index_name, arrays_list in index_chunk_arrays.items():
        stack = np.stack(arrays_list)
        p25, median, p75 = np.nanpercentile(stack, [25, 50, 75], axis=0)
        mean = np.nanmean(stack, axis=0)
        
        chunk_stms[index_name] = ({
            "p25":      p25, 
            "median":   median, 
            "p75":      p75, 
            "mean":     mean
            })
    print("STMs complete")
    
    # ==== display chunks ==== #
    ndwi_mean = chunk_stms["ndwi"]["median"]
    plt.imshow(ndwi_mean)
    plt.show()
    i += 1
    

print(f"step 3 complete: {round(time.monotonic()-start_time,2)}")
