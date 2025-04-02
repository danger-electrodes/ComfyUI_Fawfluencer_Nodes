import folder_paths
from PIL import Image, JpegImagePlugin, ExifTags
import json
from comfy.cli_args import args
import os
import numpy as np
from .constants import PHONES, REV_TAGS
import random
from datetime import datetime, timedelta
import piexif
from piexif import TYPES, TAGS

class FawfakeAuthenticImageSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        phones_options = [phone["name"] for phone in PHONES]
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "phone": (phones_options,),
                "taken_within_the_last_n_days": ("INT",)
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Fawfulized/Influencer"
    DESCRIPTION = "Saves the input images in the output folder as if they were taken from a real phone."

    def get_brightness(self, image):
        luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        average_brightness = np.mean(luminance) / 255.0
        return average_brightness
    
    def replace_placeholders(self, phone_metadata, replacer):
        def recursive_replace(metadata, tag_group):
            for key, value in metadata.items():
                if isinstance(value, str):  # Check if it's a placeholder
                    for placeholder, replacement in replacer.items():
                        if placeholder in value:
                            value = value.replace(placeholder, replacement)

                    # Convert types correctly
                    expected_type = self.get_tag_type(tag_group, key)

                    if expected_type in {TYPES.Short, TYPES.Long, TYPES.SShort, TYPES.SLong}:
                        value = int(value)  # Convert to integer
                    elif expected_type in {TYPES.Rational, TYPES.SRational, TYPES.Float, TYPES.DFloat}:
                        value = tuple(map(int, value.split("/"))) if "/" in value else float(value)
                    
                    metadata[key] = value  # Update the metadata value

                elif isinstance(value, dict):  # If nested dict, recurse
                    recursive_replace(value, key)

        recursive_replace(phone_metadata, "0th")  # Start from the root
        
        return phone_metadata

    def get_tag_type(self, tag_group, key):
        """Fetches the expected type of an EXIF tag from the _exif.py TAGS"""
        for group_name, tag_data in TAGS.items():
            if group_name == tag_group and key in tag_data:
                return tag_data[key]["type"]
        return None  # Default to None if not found
    
    def save_image_with_metadatas(self, image_path, img, phone_properties, replacer):
        phone_metadata = phone_properties["metadatas"]
        phone_real_metadata = self.replace_placeholders(phone_metadata, replacer)
        print(phone_real_metadata)
        exif_bytes = piexif.dump(phone_real_metadata)

        # Save image with EXIF metadata
        img.save(image_path, exif=exif_bytes)

        # Read EXIF back for verification
        image = Image.open(image_path)
        img_exif = image.getexif()
        print(img_exif)

    def generate_random_date(self, taken_within_the_last_n_days):
        current_date = datetime.today()
        start_date = current_date - timedelta(days=taken_within_the_last_n_days)
        random_days = random.randint(0, taken_within_the_last_n_days)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y:%m:%d")
    
    def generate_random_time(self, brightness):
        # Define probability-based hour ranges
        if brightness < 0.3:  # Mostly nighttime
            hour = random.randint(0, 5) if random.random() < 0.7 else random.randint(18, 23)
        elif brightness < 0.6:  # Transition times (morning/evening)
            hour = random.randint(6, 9) if random.random() < 0.5 else random.randint(17, 20)
        else:  # Mostly daytime
            hour = random.randint(10, 16)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        return f"{hour:02}:{minute:02}:{second:02}"
    
    def generate_random_microseconds(self):
        return f"{random.randint(0, 999999):06d}"


    def save_images(self, images, phone, taken_within_the_last_n_days):
        width = images.shape[2]
        height = images.shape[1]
        filename_prefix = "IMG"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        phone_properties = [phone_metadata for phone_metadata in PHONES if phone_metadata["name"] == phone][0]
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            brightness = self.get_brightness(i)
            date = self.generate_random_date(taken_within_the_last_n_days)
            time = self.generate_random_time(brightness)
            ms = self.generate_random_microseconds()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_data = f"{filename_prefix}_{date.replace(":", "")}_{time.replace(":", "")}"
            file = f"{filename_with_data}.jpg"
            image_abs_path = os.path.join(full_output_folder, file)

            replacer = {
                "__WIDTH__" : str(width),
                "__HEIGHT__" : str(height),
                "__DATE__" : date,
                "__TIME__" : time,
                "__MS__" : ms
            }
            self.save_image_with_metadatas(image_abs_path, img, phone_properties, replacer)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }