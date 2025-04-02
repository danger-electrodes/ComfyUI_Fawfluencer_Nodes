class FawfulizedAddImagesToImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),  # Existing image list
                "images": ("IMAGE",),  # List of images to add
            }
        }
    
    INPUT_IS_LIST = (True, True,)
    RETURN_TYPES = ("IMAGE", "IMAGE")
    OUTPUT_IS_LIST = (True, False,)
    RETURN_NAMES = ("updated image list", "last_image")
    FUNCTION = "create"
    CATEGORY = "image/image_list"

    def create(self, image_list, images):
        new_images = images[0]

        print(len(image_list))

        
        updated_list = image_list.extend(images[0])
        last_image = updated_list[-1] if isinstance(updated_list, list) else None
        return (updated_list, last_image,)  # Return as a tuple