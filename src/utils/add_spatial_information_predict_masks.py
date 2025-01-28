import logging
import warnings
from pathlib import Path

import geopandas
import hydra
import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds
from tqdm import tqdm


def setup_logger():
    """
    Sets up the logger to log to both a file and the terminal.
    """
    logging.basicConfig(level=logging.INFO,
                            format='[%(levelname)s] %(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler("add_spatial_information_predict_masks.log"),
                                logging.StreamHandler()
                            ])
    return logging.getLogger()



def turn_pred_masks_into_geotiffs(
    path_to_predmask, output_path_tiff, gdf, epsg, logger
):
    """
    Turn the predicted masks into geotiffs
    """
    count = 0
    # only look at the .tiff files
    files_to_process = list(path_to_predmask.glob("*.tiff"))

    # Process files with tqdm for progress tracking
    for file in tqdm(files_to_process[:2], desc="Converting to geotiff"):
        # get the name of the file
        name = file.stem

        # extract the school id from the file name
        school_id = "skl_" + name.split("_")[-1]

        # get the row in the gdf that corresponds to the file
        row = gdf[gdf["id"] == school_id]

        if not row.empty:
            try:
                minx, miny, maxx, maxy = row.total_bounds
                bbox = (minx, miny, maxx, maxy)

                output_tiff = output_path_tiff / f"predmask_{school_id}.geotiff"

                with rasterio.open(file) as src:
                    meta_data = src.meta

                    meta_data.update(
                        {
                            "driver": "GTiff",
                            "transform": from_bounds(
                                *bbox, width=src.width, height=src.height
                            ),
                            "crs": CRS.from_epsg(epsg),
                        }
                    )
                    with rasterio.open(output_tiff, "w", **meta_data) as dst:
                        dst.write(src.read())

                count += 1
            except Exception as e:
                logger.error(f"Failed to process {school_id}: {e}")

        else:
            logger.error(f"Could not find the school id for {name}")

    logger.info(f"Converted {count} predicted masks to geotiffs")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    bbox_path = Path(cfg.paths.BOUNDING_BOXES)
    predmask_path = Path(cfg.paths.PREDICTED_MASKS)
    output_path_tiff = Path(cfg.paths.PREDICTED_MASKS_TIFF)

    epsg = cfg.dataset.EPSG
    gdf = geopandas.read_file(bbox_path)

    # set up logger
    logger = setup_logger()

    turn_pred_masks_into_geotiffs(predmask_path, output_path_tiff, gdf, epsg, logger)


if __name__ == "__main__":
    main()
