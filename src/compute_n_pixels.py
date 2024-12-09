from pathlib import Path
import rasterio
import numpy as np
import pandas as pd
import hydra
from tqdm import tqdm
import json


def count_pixels_per_class(raster_folder, output_csv, class_mapping):
    # Convert the folder path to a Path object
    raster_folder = Path(raster_folder)
    results = []

    # Iterate through all .tif or .tiff files in the folder
    for raster_path in tqdm(
        raster_folder.glob("*.tif"), desc="Processing Rasters", unit="file"
    ):
        # Open the raster file
        with rasterio.open(raster_path) as src:
            data = src.read(1)  # Read the first band

        # Get unique values and their counts
        unique, counts = np.unique(data, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        # Map the numerical classes to categories
        mapped_counts = {
            class_mapping[str(k)]: counts_dict[k]
            for k in counts_dict
            if str(k) in class_mapping
        }

        # Add the filename and mapped counts to the results
        row = {"filename": raster_path.name}
        row.update(mapped_counts)  # Add the mapped counts
        results.append(row)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    numeric_columns = df.columns.difference(["filename"])
    df[numeric_columns] = df[numeric_columns].fillna(0).astype(int)

    # Save to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    with open(cfg.paths.LABEL_CORRESPONDANCE_FILE, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    raster_folder = Path(cfg.paths.PREDICTED_MASKS)
    output_csv = cfg.paths.PREDICTED_MASKS / Path("output.csv")

    count_pixels_per_class(raster_folder, output_csv, class_mapping)


if __name__ == "__main__":
    main()
