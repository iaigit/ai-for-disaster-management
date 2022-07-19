import os
from tqdm import tqdm
from uuid import uuid4

path_image_dir = "./dataset/day_fix/"
all_images = os.listdir(path_image_dir)

for image in tqdm(all_images):
    image_format = image.split(".")[-1]
    new_name = f"{uuid4().hex}.{image_format}"
    os.rename(
        os.path.join(path_image_dir, image), os.path.join(path_image_dir, new_name)
    )
