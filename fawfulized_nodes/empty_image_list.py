class FawfulizedEmptyImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("empty image list",)
    FUNCTION = "create"
    CATEGORY = "image/image_list"

    def create(self):
        image_list = [] 
        return (image_list,)