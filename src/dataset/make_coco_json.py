import json
import os

import geopandas as gpd
import yaml
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon


def create_coco_annotations(gdf, output_dir, output_json, label_column="labelTekst"):
    """
    Create a COCO-format JSON file for the dataset.

    Parameters:
    - gdf: GeoDataFrame containing polygons and metadata for segmentation.
    - output_dir: Directory where the images are stored.
    - output_json: Path to the output COCO JSON file.
    - label_column: Column in the GeoDataFrame that contains the class names.

    Returns:
    - None. Writes the COCO annotations JSON to the specified path.
    """

    # Initialize COCO structure
    coco_output = {"images": [], "annotations": [], "categories": []}

    # Prepare category mapping (label to ID)
    categories = gdf[label_column].unique().tolist()
    category_mapping = {cat: i + 1 for i, cat in enumerate(categories)}

    # Add categories to COCO output
    for category_name, category_id in category_mapping.items():
        coco_output["categories"].append({"id": category_id, "name": category_name})

    annotation_id = 1

    # Process each image and its corresponding polygons
    for image_id, (idx, row) in enumerate(gdf.iterrows()):
        # Image file path based on the 'id' field
        image_filename = f"image_{row['id']}.png"
        image_path = os.path.join(output_dir, image_filename)

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_filename} not found.")
            continue

        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size

        # Add image metadata to COCO output
        coco_output["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
            }
        )

        # Process each polygon or multipolygon in the image
        geometry = row.geometry
        polygons = []
        if isinstance(geometry, Polygon):
            polygons = [geometry]  # Single polygon, just wrap in a list
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)  # MultiPolygon, access its geometries
        else:
            continue  # Skip if not a Polygon or MultiPolygon

        # Create a segmentation annotation for each polygon
        for polygon in polygons:
            segmentation = []
            for x, y in polygon.exterior.coords:
                segmentation.extend([x, y])

            # Get bounding box (x, y, width, height)
            x_min, y_min, x_max, y_max = polygon.bounds
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Calculate the area of the polygon
            area = polygon.area

            # Add annotation to COCO output
            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_mapping[row[label_column]],
                    "segmentation": [segmentation],  # COCO expects a list of lists
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO annotations saved at: {output_json}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfgP = yaml.load(f, Loader=yaml.FullLoader)

    gdf = gpd.read_file(cfgP["MASK"])
    gdf = gdf[~gdf["labelTekst"].isin(["1", "1."])]
    create_coco_annotations(
        gdf,
        cfgP["IMG_DIR"],
        cfgP["COCO_JSON_PATH"],
        label_column="labelTekst",
    )
