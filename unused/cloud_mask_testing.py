import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 150_000_000


def image_to_array(file_path_s):
    """
    Convert an image or list of images to a numpy array. The image is opened 
    temporarily but not opened permanently. Note: the conversion of the image 
    to a numpy array forces its contents into the numpy.uint16 type, which 
    causes overflow errors, which then causes the index calculation to break. 
    To fix this, convert the uint16 arrays must be converted to integer type 
    when calculating the indices. 
    
    Parameters
    ----------
    file_path_s : list or string
        A list containing all the file paths 
        
    Returns
    -------
    image_arrays : list of numpy arrays
        A list containing some number of numpy arrays converted from images. 
        
    """
    if not isinstance(file_path_s, list):
        with Image.open(file_path_s) as img:
            image_array = np.array(img)
        return image_array
    else:
        image_arrays = []
        for file_path in file_path_s:
            with Image.open(file_path) as img:
                image_arrays.append(np.array(img))
        return image_arrays

def plot_indices(data, size):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(size))
    
    ax = plt.gca()
    plt.imshow(data)
    
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False, 
                   labelleft=False, labelbottom=False)
    
    print("displaying NDWI image", end="... ")
    plt.show()
    print("NDWI image display complete!")

file_paths = ["crop_test_img_green.jpg", "crop_test_img_red.jpg", "crop_test_img_nir.jpg"]

image_arrays = image_to_array(file_paths)

from omnicloudmask import predict_from_array

input_array = np.stack(
    (
    image_arrays[2], # red
    image_arrays[0], # green
    image_arrays[1]  # nir
    ))

pred_mask_2d = predict_from_array(
    input_array=input_array, 
    patch_size=150, 
    patch_overlap=100, 
    batch_size=1, 
    inference_device="cuda", 
    inference_dtype="float32"
    )[0]

combined_mask = (
    (pred_mask_2d == 1) | 
    (pred_mask_2d == 2) | 
    (pred_mask_2d == 3)
    )

for i in range(len(image_arrays)):
    # float is used as it supports NaN
    image_arrays[i] = image_arrays[i].astype(np.float32)
    image_arrays[i][combined_mask] = np.nan

green, nir, red = image_arrays
ndwi = (green - nir) / (green + nir)

plot_indices(ndwi, (3,3))

