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

# FIX 1: imports moved to the top, before any function definitions or code
# that uses them. Previously, split_array() used np.array_split() but numpy
# had not been imported yet, causing a NameError on the first call.
import numpy as np
import omnicloudmask as ocm
import torch
import time
import matplotlib.pyplot as plt

torch.set_num_threads(12) # number of threads on dell xps 9315

N_CHUNKS = 5000


def split_array(array, n_chunks):
    """Split a 2D array into n_chunks roughly equal sub-arrays arranged in a
    grid. n_chunks should be a perfect square (or close to one); the actual
    chunk count is floor(sqrt(n_chunks))^2."""
    rows = np.array_split(array, np.sqrt(n_chunks), axis=0)
    split_arrays = [np.array_split(row_chunk, np.sqrt(n_chunks), axis=1)
                    for row_chunk in rows]
    chunks = [subarray for row_chunk in split_arrays for subarray in row_chunk]
    return chunks


# %% 1. Create image array(s)
start_time = time.monotonic()

img1 = np.random.rand(3, 11000, 11000).astype(np.float32)
img2 = np.random.rand(3, 11000, 11000).astype(np.float32)
img_list = [img1, img2]

print(f"step 1 complete: {round(time.monotonic() - start_time, 2)}")

# %% 2. Chunkification
# Each image is split into chunks immediately after loading so that the rest
# of the pipeline never has to hold a full processed image in memory at once.
# Structure of img_chunks_list:
#   img_chunks_list[img_idx]        -> one entry per image
#   img_chunks_list[img_idx][band]  -> one entry per band (green=0, nir=1, red=2)
#   img_chunks_list[img_idx][band][chunk_idx] -> a single 2D chunk array
start_time = time.monotonic()

img_chunks_list = []
for img in img_list:
    img_chunks_per_band = []
    for band in img:                        # img has shape (3, H, W)
        img_chunks_per_band.append(split_array(array=band, n_chunks=N_CHUNKS))
    img_chunks_list.append(img_chunks_per_band)

n_chunks_actual = len(img_chunks_list[0][0]) # real chunk count after splitting
print(f"step 2 complete: {round(time.monotonic() - start_time, 2)}")
print(f"  chunks per image: {n_chunks_actual}")

# %% 3. Chunk-by-chunk processing loop
# FIX 5: stms accumulator is initialised OUTSIDE the loop so results are not
# wiped on every iteration.
stms = {"ndwi": [], "ndvi": []}   # FIX 3: dict, not [{}] (a list)

start_time = time.monotonic()
np.seterr(divide="ignore", invalid="ignore")

i = 0
while i < n_chunks_actual:         # FIX 7: loop over all chunks, not just 5

    ndwi_chunk_across_imgs = []    # collect per-image ndwi for this chunk
    ndvi_chunk_across_imgs = []    # collect per-image ndvi for this chunk

    for img_chunks in img_chunks_list:
        # img_chunks[band][chunk_idx] -> 2D array for that band at chunk i
        green_chunk = img_chunks[0][i]
        nir_chunk   = img_chunks[1][i]
        red_chunk   = img_chunks[2][i]

        # ==== mask known features ==== #
        # (placeholder - same as original)
        pass

        # ==== mask clouds ==== #
        # FIX 2: ocm.predict_from_array expects a (3, H, W) array in
        # (red, green, nir) order. Previously this was img_chunk[i] which
        # re-used the outer loop counter i as a band index - wrong and will
        # crash for i >= 3.
        input_array = np.stack([red_chunk, green_chunk, nir_chunk])
        pred_mask_2d = ocm.predict_from_array(
            input_array,
            patch_size=127,
            patch_overlap=64,
            inference_device="cpu",
            inference_dtype="fp32"
        )[0]

        combined_mask = (
            (pred_mask_2d == 1) |
            (pred_mask_2d == 2) |
            (pred_mask_2d == 3)
        )

        # Apply mask (float32 already, supports NaN)
        green_chunk = green_chunk.copy(); green_chunk[combined_mask] = np.nan
        nir_chunk   = nir_chunk.copy();   nir_chunk[combined_mask]   = np.nan
        red_chunk   = red_chunk.copy();   red_chunk[combined_mask]   = np.nan

        # ==== calculate indices ==== #
        ndwi = (green_chunk - nir_chunk) / (green_chunk + nir_chunk)
        ndvi = (nir_chunk   - red_chunk) / (nir_chunk   + red_chunk)

        ndwi_chunk_across_imgs.append(ndwi)
        ndvi_chunk_across_imgs.append(ndvi)

    # ==== spectral temporal metrics (per chunk, across images) ==== #
    # FIX 4: np.stack receives a *list* of same-shaped 2D arrays, one per
    # image. Previously index_arrays held single arrays, not lists, so
    # np.stack had nothing meaningful to stack along a temporal axis.
    index_chunk_arrays = {
        "ndwi": ndwi_chunk_across_imgs,
        "ndvi": ndvi_chunk_across_imgs,
    }

    chunk_stms = {}
    for index_name, arrays_list in index_chunk_arrays.items():
        stack = np.stack(arrays_list)   # shape: (n_images, H_chunk, W_chunk)
        mean, p25, median, p75 = np.nanpercentile(
            stack, [0, 25, 50, 75], axis=0
        )
        chunk_stms[index_name] = {
            "p25":    p25,
            "median": median,
            "p75":    p75,
            "mean":   mean,
        }

    # FIX 3 (continued): stms is now a proper dict of lists; append this
    # chunk's result so the full composite can be assembled after the loop.
    stms["ndwi"].append(chunk_stms["ndwi"])
    stms["ndvi"].append(chunk_stms["ndvi"])

    i += 1  # FIX 7: increment i - previously missing, causing infinite loop

print(f"step 3 complete: {round(time.monotonic() - start_time, 2)}")

# %% 4. Display
# FIX 6: stms["ndwi"] is now a list of per-chunk dicts. Access by index first,
# then by key. Here we plot the median of the first chunk as a quick check.
# (A full composite would reassemble all chunks into a single 2D array.)
ndwi_median_chunk0 = stms["ndwi"][0]["median"]
plt.figure()
plt.imshow(ndwi_median_chunk0, cmap="RdYlBu")
plt.title("NDWI median - chunk 0")
plt.colorbar()
plt.show()
