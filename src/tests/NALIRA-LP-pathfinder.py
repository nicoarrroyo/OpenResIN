#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:16:42 2026

@author: nico
LOW POWER NALIRA
"""
import numpy as np
import omnicloudmask as ocm
import torch
torch.set_num_threads(12) # number of threads on dell xps 9315

# %% create image array(s)
img1 = np.random.rand(3, 127, 127).astype(np.float32)
img2 = np.random.rand(3, 127, 127).astype(np.float32)

# %% cloud masking



stack = np.stack(arrays_list)



