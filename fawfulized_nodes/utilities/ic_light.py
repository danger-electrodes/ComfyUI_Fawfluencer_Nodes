import comfy
import torch
import numpy as np
from enum import Enum

class ICLight:
    def extra_conds(self, **kwargs):
        out = {}
        
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        model_in_channels = self.model_config.unet_config['in_channels']
        input_channels = image.shape[1] + 4

        if model_in_channels != input_channels:
            raise Exception(f"Input channels {input_channels} does not match model in_channels {model_in_channels}, 'opt_background' latent input should be used with the IC-Light 'fbc' model, and only with it")
        
        if image is None:
            image = torch.zeros_like(noise)

        if image.shape[1:] != noise.shape[1:]:
            image = comfy.utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        image = comfy.utils.resize_to_batch_size(image, noise.shape[0])

        process_image_in = lambda image: image
        out['c_concat'] = comfy.conds.CONDNoiseShape(process_image_in(image))

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)
        return out
    

class LightPosition(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"

def generate_gradient_image(width:int, height:int, start_color: tuple, end_color: tuple, multiplier: float, lightPosition:LightPosition):
    """
    Generate a gradient image with a light source effect.

    Parameters:
    width (int): Width of the image.
    height (int): Height of the image.
    start_color: Starting color RGB of the gradient.
    end_color: Ending color RGB of the gradient.
    multiplier: Weight of light.
    lightPosition (LightPosition): Position of the light source.

    Returns:
    np.array: 2D gradient image array.
    """
    # Create a gradient from 0 to 1 and apply multiplier
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(0, 1, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(1, 0, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(0, 1, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(1, 0, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported position. Choose from {', '.join([member.value for member in LightPosition])}.")

    # Interpolate between start_color and end_color based on the gradient
    gradient_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    
    gradient_img = np.clip(gradient_img, 0, 255).astype(np.uint8)
    return gradient_img

def toRgb(hex):
    if hex.startswith('#') and len(hex) == 7: 
        color_rgb =tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))
    else: 
        color_rgb = tuple(int(i) for i in hex.split(','))
    return color_rgb

def toRgb01(hex):
    if hex.startswith('#') and len(hex) == 7: 
        color_rgb = tuple(int(hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    else: 
        color_rgb = tuple(int(i) / 255.0 for i in hex.split(','))
    return color_rgb
    
def get_light_source(light_position, multiplier, start_color, end_color, width, height, batch_size=1):
    lightPosition = LightPosition(light_position)
    start_color_rgb = toRgb(start_color)
    end_color_rgb = toRgb(end_color)
    image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)
    
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    image = image.repeat(batch_size, 1, 1, 1)
    return image

def color_lightmap(light_map, start_color_rgb, end_color_rgb):
    start_color = torch.tensor(start_color_rgb, dtype=torch.float32, device=light_map.device).view(1, 1, 1, 3)
    end_color = torch.tensor(end_color_rgb, dtype=torch.float32, device=light_map.device).view(1, 1, 1, 3)
    interpolated_color = start_color + (light_map * (end_color - start_color))

    return interpolated_color

def create_lightmap(normal_map, light_angle, grow_mask_by, blur_size, start_color, end_color, radians = False, threshold = 0.2):
    from .image import expand_mask, mask_to_image, image_to_mask

    device = normal_map.device

    if not radians:
        light_angle = np.radians(light_angle)

    normals = normal_map[..., :3]
    normals = (normals * 2.0) - 1.0

    light_dir = torch.tensor([np.cos(light_angle), np.sin(light_angle), 0.0], device=device)
    light_dir = light_dir / torch.norm(light_dir)

    dot_product = torch.sum(normals * light_dir, dim=-1)

    mask = (dot_product > threshold).float()

    real_mask = expand_mask(mask, grow_mask_by, True, False, blur_size, 0.0, 1.00, 1.00, False)

    light_map = mask_to_image(real_mask)

    start_color_rgb = toRgb01(start_color)
    end_color_rgb = toRgb01(end_color)

    colored_lightmap = color_lightmap(light_map, start_color_rgb, end_color_rgb)
    return colored_lightmap