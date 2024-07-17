import json
import os


def coco_to_label_studio(coco_json_path, images_dir, output_json_path):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    label_studio_data = []

    image_id_to_filename = {
        image["id"]: image["file_name"] for image in coco_data["images"]
    }

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        image_filename = image_id_to_filename[image_id]
        image_path = os.path.join(images_dir, image_filename)

        label_studio_annotation = {
            "data": {"image": os.path.abspath(image_path)},
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": annotation["bbox"][0],
                                "y": annotation["bbox"][1],
                                "width": annotation["bbox"][2],
                                "height": annotation["bbox"][3],
                                "rectanglelabels": [annotation["category_id"]],
                            },
                        }
                    ]
                }
            ],
        }

        label_studio_data.append(label_studio_annotation)

    with open(output_json_path, "w") as f:
        json.dump(label_studio_data, f, indent=4)


# Example usage
coco_json_path = "../backup/COINS-1/coco/project-SHKALIM-6-2/result.json"
images_dir = "../backup/COINS-1/coco/project-SHKALIM-6-2/images/"
output_json_path = (
    "../backup/COINS-1/coco/project-SHKALIM-6-2/label_studio_annotations.json"
)

coco_to_label_studio(coco_json_path, images_dir, output_json_path)
