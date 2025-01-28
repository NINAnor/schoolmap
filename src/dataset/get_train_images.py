#!/usr/env/bin python3

from io import BytesIO
from pathlib import Path

import backoff
import geopandas as gpd
import hydra
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm


def create_tiff(output_path_tiff, res_img, bbox, epsg):
    # open the image in memory using rasterio
    with rasterio.open(BytesIO(res_img)) as src:
        # update the metadata to include geospatial information
        profile = src.meta
        profile.update(
            {
                "driver": "GTiff",
                "transform": from_bounds(*bbox, width=src.width, height=src.height),
                "crs": CRS.from_epsg(epsg),
            }
        )

        with rasterio.open(output_path_tiff, "w", **profile) as dst:
            dst.write(src.read())


def create_png(img_bytes, crs, bbox, output_png):
    img = Image.open(img_bytes)
    img_np = np.array(img)
    # Calculate the transform and shape for the new coordinate system
    transform, width, height = calculate_default_transform(
        crs,
        crs,
        img_np.shape[1],
        img_np.shape[0],
        bbox[0],  # minx
        bbox[1],  # miny
        bbox[2],  # maxx
        bbox[3],  # maxy
    )

    # Reproject the image data to the new CRS
    warped_img = np.empty((height, width, img_np.shape[2]), dtype=np.uint8)
    for i in range(img_np.shape[2]):  # Loop through the RGB bands
        reproject(
            source=img_np[:, :, i],
            destination=warped_img[:, :, i],
            src_transform=transform,  # Use the affine transform from before
            src_crs=crs,
            dst_transform=transform,  # The new transformation
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

    Image.fromarray(warped_img).save(output_png)

    return warped_img


@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, max_tries=5)
def get_image(gdf, item_id, wms_url, crs, output_path):
    """
    Get the image from WMS and returns a numpy array
    """

    gdf_item = gdf[gdf["id"] == item_id]
    minx, miny, maxx, maxy = gdf_item.total_bounds
    height = int(11**5 * abs(maxy - miny))
    width = int(11**5 * abs(maxx - minx))
    bbox = (minx, miny, maxx, maxy)

    # Fetch ortofoto
    wms_url = wms_url
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ortofoto",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "SRS": "EPSG:4326",
    }

    response = requests.get(wms_url, params=params)
    response.raise_for_status()
    img_res = response.content
    img_bytes = BytesIO(img_res)

    output_path_tiff = output_path / f"image_{item_id}.geotiff"
    output_png = output_path / f"image_{item_id}.png"

    create_tiff(output_path_tiff, img_res, bbox, 4326)
    create_png(img_bytes, crs, bbox, output_png)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    gdf = gpd.read_file(cfg.paths.MASK)
    gdf = gdf[~gdf["labelTekst"].isin(["1", "1."])]  # two labels that are artifacts

    img_dir = Path(cfg.paths.IMG_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    for item_id in tqdm(gdf["id"].unique()[:2], desc="Saving Images"):
        get_image(
            gdf,
            item_id,
            cfg.dataset.WMS_URL,
            cfg.dataset.CRS,
            img_dir,
        )


if __name__ == "__main__":
    main()
