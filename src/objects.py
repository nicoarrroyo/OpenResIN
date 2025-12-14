import os
import numpy as np
import config as c
import pipeline_operations

# %% sentinel image object
class sentinel_image:
    def __init__(self, folder_name):
        """The Setup Step: Runs automatically when you create the object."""
        self.name = folder_name
        self.path = os.path.join(c.DATA_DIR, folder_name)
        
        # State: Variables that belong to THIS specific image
        self.bands = {}
        self.res = {}
        self.indices = {}
        self.chunks = []
        
        print(f"Initialized sentinel_image object for: {self.name}")

    def get_band_data(self):
        print(f"Loading data from {self.path} using HighRes={c.HIGH_RES}")
        self.bands["green"] = np.random.rand(100, 100)
        self.bands["nir"] = np.random.rand(100, 100)
        print("Data Loaded")

    def compute_indices(self):
        if "green" not in self.bands or "nir" not in self.bands:
            raise ValueError("Data not loaded! Call get_band_data() first.")
        
        print("Calculating NDWI")
        ndwi_map = pipeline_operations.calculate_ndwi(self.bands["green"], self.bands["nir"])
        
        self.indices["ndwi"] = ndwi_map

    def segment_image(self):
        """Splits the calculated index into chunks."""
        print(f"Splitting into {c.N_CHUNKS} chunks")
        self.chunks = pipeline_operations.split_into_chunks(self.indices["ndwi"], c.N_CHUNKS)

