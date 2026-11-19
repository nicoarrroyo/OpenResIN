import numpy as np


def image_to_array(file_path_s):
    """
    Read one raster file, or a list of them, into numpy arrays with rasterio.
    Values keep the file's own dtype (uint16 for Sentinel-2 band files), so
    callers convert to float before calculating indices.

    Parameters
    ----------
    file_path_s : list or string
        One file path, or a list of file paths.

    Returns
    -------
    numpy array, or list of numpy arrays
        One array per path given, in the order given. A single-band file gives
        shape (rows, cols); a multi-band file gives (rows, cols, bands).

    """
    import rasterio

    def read_one(file_path):
        with rasterio.open(file_path) as src:
            if src.count == 1:
                return src.read(1)
            return np.transpose(src.read(), (1, 2, 0))

    if not isinstance(file_path_s, list):
        return read_one(file_path_s)
    return [read_one(file_path) for file_path in file_path_s]

def get_image_bounds(image_metadata):
    transform = image_metadata["transform"]
    width = image_metadata["width"]
    height = image_metadata["height"]

    left = transform[2]
    top = transform[5]
    right = left + transform[0] * width
    bottom = top + transform[4] * height

    return (left, bottom, right, top)

def known_feature_mask(image_array, image_metadata, data_path,
                       feature_type, buffer_metres=None, fill=0):
    """
    "fill" is the value that masked pixels become. Default is 0 to preserve
    uint16 band arrays. For arrays of computed indices, pass NaN directly.
    """
    import fiona
    import geopandas as gpd
    from rasterio import features
    s2_bounds_tuple = get_image_bounds(image_metadata)
    s2_crs = image_metadata["crs"]

    with fiona.open(data_path, "r") as collection:
        feature_crs = collection.crs

    if s2_crs != feature_crs:
        from shapely.geometry import box
        bbox_gdf = gpd.GeoDataFrame(
            {"geometry": [box(*s2_bounds_tuple)]},
            crs=s2_crs)

        bbox_reprojected = bbox_gdf.to_crs(feature_crs)
        bounds_for_filtering = tuple(bbox_reprojected.total_bounds)
    else:
        bounds_for_filtering = s2_bounds_tuple
    # gdf means geospatial data fusion
    feature_gdf = gpd.read_file(data_path, bbox=bounds_for_filtering)
    # vector outline of feature (clipped to relevant bounds)

    if feature_gdf.empty:
        print(f"no {feature_type} found in this image")
        return image_array

    if feature_gdf.crs != s2_crs:
        feature_gdf = feature_gdf.to_crs(s2_crs) # step to ensure
        # shapefile and raster are using the same coordinate system

    if buffer_metres is not None and buffer_metres > 0: # expands the lines in
    # the shapefile to render them wider than one pixel
        feature_gdf["geometry"] = feature_gdf.geometry.buffer(buffer_metres)

    feature_mask_layer = features.rasterize(
        shapes=feature_gdf.geometry,
        out_shape=image_array.shape,
        fill=0,
        transform=image_metadata["transform"],
        default_value=1,
        dtype=np.uint8
        )

    if feature_type == "sea": # sea masking logic is inverted because
    # the shapefile defines land boundaries
        feature_mask = feature_mask_layer == 0
        # masks out where the layer is 0 (outside land polygons)
    else:
        feature_mask = feature_mask_layer == 1

    image_array[feature_mask] = fill

    return image_array

def mask_urban_areas(image_array, image_metadata, urban_data_path, fill=0):
    s2_bounds_tuple = get_image_bounds(image_metadata)
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    with rasterio.open(urban_data_path) as urban_src:
        window = from_bounds(*s2_bounds_tuple, transform=urban_src.transform)
        urban_data_window = urban_src.read(1, window=window)

        if urban_data_window.shape != image_array.shape:
            reprojected_urban_data = np.zeros(image_array.shape,
                                              dtype=np.uint8)

            reproject(
                source=rasterio.band(urban_src, 1),
                destination=reprojected_urban_data,
                src_transform=urban_src.transform,
                src_crs=urban_src.crs,
                dst_transform=image_metadata["transform"],
                dst_crs=image_metadata["crs"],
                resampling=Resampling.nearest
                )

            urban_mask = (reprojected_urban_data==20)|(reprojected_urban_data==21)
        else:
            urban_mask = (urban_data_window == 20) | (urban_data_window == 21)

        # Apply the mask
        image_array[urban_mask] = fill
    return image_array
