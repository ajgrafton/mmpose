from meerkat.training_data import Annotator
from mmpose.datasets.pipelines.meerkat_data_loader import LoadMultipleImagesFromMeerkat
import json
import matplotlib.pyplot as plt
import os

annotator = Annotator.create("test-multi")
annotator.add_annotations(
    "/data/mk025b_torso_training_2_dir.csv", "/data/annotation/mk025b/", "mk025b"
)
annotator.set_combine_channels(True)
annotator.set_bgr(True)
annotator.set_ir(True)
annotator.set_depth(True)
annotator.set_upright(True)
annotator.check_json_export()
# annotator.export_to_json("test-multi-annotations.json", "test-multi-archives")

loader = LoadMultipleImagesFromMeerkat(color_start_indices=0)
with open("test-multi-annotations.json", "r") as f:
    data = json.load(f)
img = data["images"][30]
file_name = {"image_file": os.path.join("test-multi-archives", img["file_name"])}
loaded_image = loader(file_name)["img"]
annotation = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]][0]

plt.figure()
plt.imshow(loaded_image[:, :, :3])
plt.figure()
plt.imshow(loaded_image[:, :, 3])
plt.figure()
plt.imshow(loaded_image[:, :, 4])
plt.show()
