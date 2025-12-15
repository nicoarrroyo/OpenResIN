# standard libraries
import os

# third-party libraries
import numpy as np

# local libraries
import config as c
import data_handling as data_do
import image_handling as image_do
import misc
import user_interfacing as ui

import pipeline_operations # will merge this with others (temporary)


# %% sentinel image object
class SentinelImage:
    def __init__(self, folder_name):
        """The Setup Step: Runs automatically when you create the object."""
        self.name = folder_name
        self.path = os.path.join(c.DATA_DIR, folder_name)
        
        # State: Variables that belong to THIS specific image
        self.bands = {}
        self.res = {}
        self.indices = {}
        self.chunks = []
        
        print(f"Initialised SentinelImage object for: {self.name}")
    
    def one_create_image_arrays(self):
        self.establish_paths(self)
        """Most Sentinel 2 files that come packaged in a satellite image 
        folder follow naming conventions that use information contained in the 
        title of the folder. This information can be used to easily navigate 
        through the folder's contents."""
        
        self.get_images(self)
        """This operation takes a long time because the image files are so big. 
        The difference in duration for this operation between using and not 
        using HIGH_RES is a factor of about 20, but, again, not using HIGH_RES 
        results in unusable images."""
        
        print("step 1 complete! finished at {dt.datetime.now().time()}")

    def establish_paths(self):
        # %%%% 1.1 Establishing Paths
        images_path = os.path.join(self.path, "GRANULE")
        
        # %%%%% 1.1.1 Subfolder iterative search
        """This folder has a strange naming convention that doesn't quite apply 
        to the other folders, so it's difficult to find a rule that would work 
        for any Sentinel 2 image. The easier way of finding this folder is by 
        searching for any available directories in the GRANULE folder, and if 
        there is more than one, then alert the user and exit, otherwise go into 
        that one directory because it will be the one we're looking for."""
        subdirs = [d for d in os.listdir(images_path) 
                   if os.path.isdir(os.path.join(images_path, d))]
        if len(subdirs) == 1:
            images_path = os.path.join(images_path, subdirs[0])
        else:
            print("Too many subdirectories in 'GRANULE':", len(subdirs))
            ui.confirm_continue_or_exit()
        
        # %%%%% 1.1.2 Resolution selection and file name deconstruction
        """Low resolution should only be used for troubleshooting as it does 
        not produce usable training data. High resolution uses the 10m spatial 
        resolution images but processing time is significantly longer. To save 
        some space, when splitting the folder name into its parts, some of the 
        splitting functions outputs are suppressed. In order, the variables not 
        assigned are: sentinel number, instrument and product level, 
        processing baseline number, relative orbit number, and product 
        discriminator and format."""
        if c.HIGH_RES:
            res = "10m"
            path_10 = os.path.join(images_path, "IMG_DATA", "R10m")
        else:
            res = "60m"
            path_60 = os.path.join(images_path, "IMG_DATA", "R60m")
        
        (_, _, datatake_start_sensing_time, _, _, 
         tile_number_field, _) = folder.split("_")
        
        prefix = f"{tile_number_field}_{datatake_start_sensing_time}"
        bands = get_sentinel_bands(sat_number, c.HIGH_RES)
        
        for band in bands:
            if c.HIGH_RES:
                file_paths.append(
                os.path.join(path_10, f"{prefix}_B{band}_10m.jp2")
                )
            else:
                file_paths.append(
                os.path.join(path_60, f"{prefix}_B{band}_60m.jp2")
                )

    def get_images(self):
        # %%%% 1.2 Opening and Converting Images
        try:
            with rasterio.open(file_paths[0]) as src:
                image_metadata = src.meta.copy()
        except Exception as e:
            print("failed raster metadata pull")
            print("error: ", e)
            confirm_continue_or_exit()
        
        image_arrays = image_to_array(file_paths)
        
        if c.CLOUD_MASKING:
            image_arrays_clouds = image_arrays
    
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

class DataLabeller:
    def __init__(self, sentinel_image_object):
        self.name = "this is gonna be a hugely different beast of a program"
        self.target_image = sentinel_image_object
        self.relative_chunk_index = 0
        self.current_chunk_index = 0 # to be changed
        self.user_coords = []
    
    def label_chunk(self):
        for chunk in self.target_image.chunks:
            n = ui.enumerate_rois # this function doesn't exist yet but it would be the section of asking the user to say how many reservoirs there are
            coords = ui.prompt_roi(chunk, n)
            self.save_responses(coords)
    
    def save_responses(self, coords):
        self.user_coords.append(coords) # something like this (not sure if this would work)
    


