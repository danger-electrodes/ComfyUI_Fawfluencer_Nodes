import folder_paths
import node_helpers
from PIL import Image, ImageOps, ImageSequence, ImageFilter
import numpy as np
import torch
import comfy.model_management
import comfy.utils
from comfy.utils import common_upscale
import random
from .sampling import sample, unify_sigmas
import os
from typing import Optional, Union, List
import sys
import subprocess
import scipy.ndimage
import math

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

MODELS_DIR =  folder_paths.models_dir

class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

# Freeze PIP modules
def packages(versions=False):
    try:
        result = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.STDOUT)
        lines = result.decode().splitlines()
        return [line if versions else line.split('==')[0] for line in lines]
    except subprocess.CalledProcessError as e:
        print("An error occurred while fetching packages:", e.output.decode())
        return []

def install_package(package, uninstall_first: Union[List[str], str] = None):
    if os.getenv("WAS_BLOCK_AUTO_INSTALL", 'False').lower() in ('true', '1', 't'):
        cstr(f"Preventing package install of '{package}' due to WAS_BLOCK_INSTALL env").msg.print()
    else:
        if uninstall_first is None:
            return

        if isinstance(uninstall_first, str):
            uninstall_first = [uninstall_first]

        cstr(f"Uninstalling {', '.join(uninstall_first)}..")
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'uninstall', *uninstall_first])
        cstr("Installing package...").msg.print()
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', '-q', 'install', package])

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor


def flip_batch(images, flip_x=False, flip_y=False):
    if flip_x:
        images = torch.flip(images, dims=[2])  
    if flip_y:
        images = torch.flip(images, dims=[1]) 

    return images

def merge_batch(batch1, batch2):
    merged = torch.cat((batch1, batch2), dim=0)
    return merged

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pilmaskgrow(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def pil2tensormaskgrow(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

if "rembg" not in packages():
    install_package("rembg")

from rembg import remove, new_session


def load_image_from_path(image_path):
    img = node_helpers.pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image, output_mask

def load_image_from_path_multiple(images_paths):
    images = []
    masks = []

    for i in range(len(images_paths)):
        path = images_paths[i]
        img, msk = load_image_from_path(path)
        images.append(img)
        masks.append(msk)

    return images, masks

def load_image(image):
    image_path = folder_paths.get_annotated_filepath(image)

    img = node_helpers.pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image, output_mask

def get_empty_latent(width, height, batch):
    samples = torch.zeros([batch, 4, height // 8, width // 8], device=comfy.model_management.intermediate_device())
    latent = {"samples":samples}
    return latent

def set_latent_noise_mask(samples, mask):
    samples["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return samples

def upscale_latent_by(samples, upscale_method, scale_by):
    s = samples.copy()
    width = round(samples["samples"].shape[-1] * scale_by)
    height = round(samples["samples"].shape[-2] * scale_by)
    s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
    return s

def upscale_image_by(image, upscale_method, scale_by):
    samples = image.movedim(-1,1)
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return s

def resize(image, width, height, upscale_method="lanczos"):
    B, H, W, C = image.shape
    
    if width == 0:
        width = W
    if height == 0:
        height = H
    
    image = image.movedim(-1,1)
    image = common_upscale(image, width, height, upscale_method, "center")
    image = image.movedim(1,-1)

    return image

def fit(image, target_width, target_height, upscale_method="lanczos"):
    B, H, W, C = image.shape 

    width_ratio = target_width / W
    height_ratio = target_height / H

    if width_ratio < height_ratio:
        new_width = target_width
        new_height = int(H * width_ratio)
    else:
        new_height = target_height
        new_width = int(W * height_ratio)
    
    resized_img = resize(image, new_width, new_height, upscale_method)

    return resized_img

def remove_background(images):
    os.environ['U2NET_HOME'] = os.path.join(MODELS_DIR, 'rembg')
    os.makedirs(os.environ['U2NET_HOME'], exist_ok=True)

    batch_tensor = []
    bgrgba = [0, 0, 0, 255]
    for image in images:
        image = tensor2pil(image)

        # Remove background
        img_no_bg = remove(
            image,
            session=new_session("u2net"),
            post_process_mask=False,
            alpha_matting=False,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            only_mask=False,
            bgcolor=bgrgba # Ensure background is black
        ).convert('RGB')

        # Convert to NumPy array for pixel manipulation
        img_array = np.array(img_no_bg)

        # Create mask: If pixel is black (0,0,0), keep it. Otherwise, make it white (255,255,255)
        mask = np.where((img_array == [0, 0, 0]).all(axis=-1, keepdims=True), 0, 255).astype(np.uint8)

        # Convert to tensor
        mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask_tensor = mask_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        batch_tensor.append(mask_tensor)

        del img_no_bg
        del mask
        del mask_tensor

    del bgrgba
    
    batch_tensor = torch.cat(batch_tensor, dim=0)  # Combine batch

    return batch_tensor  # Return final mask tensor


def expand_mask(mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pilmaskgrow(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensormaskgrow(pil_image)
            blurred = torch.cat(out, dim=0)
            return blurred
        else:
            return torch.stack(out, dim=0)


def create_rectangle_mask(latent):
    _, _, height, width = latent["samples"].shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    min_size = min(width, height) // 2
    max_size = (3 * min(width, height)) // 4
    rect_width = random.randint(min_size, max_size)
    rect_height = random.randint(min_size, max_size)
    
    x = random.randint(0, width - rect_width)
    y = random.randint(0, height - rect_height)
    
    mask[y:y+rect_height, x:x+rect_width] = 255

    # Convert NumPy array to PyTorch tensor and add batch and channel dimensions
    mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    return mask_tensor


def create_silhouette_mask(noise, guider, sampler, high, low, latent, vae):
    unified_sigmas = unify_sigmas(high, low)

    out, out_denoise = sample(noise, guider, sampler, unified_sigmas, latent)

    images = vae.decode(out_denoise["samples"])
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    mask_tensor = remove_background(images)

    del out
    del out_denoise
    del images
    del unified_sigmas

    return mask_tensor

def create_random_mask(noise, guider, sampler, high, low, latent, vae, grow_mask_by, blur_size, mask_type):
    mask_tensor = None

    match mask_type:
        case "rectangle":
            mask_tensor = create_rectangle_mask(latent)
        case "silhouette":
            mask_tensor = create_silhouette_mask(noise, guider, sampler, high, low, latent, vae)

    blurred_mask_tensor = expand_mask(mask_tensor, grow_mask_by, True, False, blur_size, 0.0, 1.00, 1.00, False)


    del mask_tensor
    return blurred_mask_tensor

def image_to_mask(image, channel):
    channels = ["red", "green", "blue", "alpha"]
    mask = image[:, :, :, channels.index(channel)]
    return mask

def mask_to_image(mask):
    image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return image
    
def combine_images_dynamic(image_list, num_columns, padding=10, bg_color=(0, 0, 0)):
    pil_image_list = [tensor2pil(img) for img in image_list]
    
    num_images = len(pil_image_list)
    num_rows = (num_images + num_columns - 1) // num_columns  # Ceiling division
    
    # Determine individual image dimensions
    max_width = max(img.width for img in pil_image_list)
    max_height = max(img.height for img in pil_image_list)
    
    # Determine total canvas size
    total_width = num_columns * max_width + (num_columns - 1) * padding + 2 * padding
    total_height = num_rows * max_height + (num_rows - 1) * padding + 2 * padding
    
    final_image = Image.new("RGB", (total_width, total_height), bg_color)

    for idx, img in enumerate(pil_image_list):
        row = idx // num_columns
        col = idx % num_columns
        
        x_offset = padding + col * (max_width + padding) + (max_width - img.width) // 2
        y_offset = padding + row * (max_height + padding) + (max_height - img.height) // 2
        
        final_image.paste(img, (x_offset, y_offset))
    
    final_image_tensor = pil2tensor(final_image)
    return final_image_tensor


def resize_image(pil_image, width, height, bg_color=(0, 0, 0)):
    resized_pil_image = Image.new("RGB", (width, height), bg_color)
    orig_width, orig_height = pil_image.size
    x_offset = (width - orig_width) // 2
    y_offset = (height - orig_height) // 2
    resized_pil_image.paste(pil_image, (x_offset, y_offset))
    return resized_pil_image

def image_list_to_batch(img_list, bg_color=(0, 0, 0), target_ratio=None):

    max_width = max(img.shape[2] for img in img_list) #shape of an image tensor is always BHWC
    max_height = max(img.shape[1] for img in img_list)

    if(target_ratio is not None):
        max_width = max(max_width, max_height)
        max_height = max_width
        max_width = int(max_width * target_ratio[0])
        max_height = int(max_height * target_ratio[1])

    resized_tensor_list = [fit(img, max_width, max_height) for img in img_list]
    pil_image_list = [tensor2pil(img) for img in resized_tensor_list]

    pil_image_list_resized = [resize_image(img, max_width, max_height, bg_color) for img in pil_image_list]

    img_list_resized = [pil2tensor(img) for img in pil_image_list_resized]
    image1 = img_list_resized[0]

    for image2 in img_list_resized[1:]:
        image1 = torch.cat((image1, image2), dim=0)

    return image1


def image_batch_to_list(img_batch):

    img_list = []
    for image_index in range(img_batch.shape[0]):

        img = img_batch[image_index].unsqueeze(0)
        img_list.append(img)

    return img_list

def resize_image_any(image, width, height, bg_color = (0, 0, 0)):
    pil_image = tensor2pil(image)
    pil_image_resized = resize_image(pil_image, width, height, bg_color)
    resized_image = pil2tensor(pil_image_resized)
    return resized_image

def get_inverse_mask(mask):
    inverse_mask = 1.0 - mask
    return inverse_mask

def set_mask_strength(mask, strength):
    new_mask = mask * strength
    return new_mask

def combine_mask(mask1, mask2):
    combined_mask = mask1 + mask2
    return combined_mask

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)

def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

def contrast_adaptive_sharpening(image, amount):
    img = T.functional.pad(image, (1, 1, 1, 1)).cpu()

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)

    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w)

    output = ((b + d + f + h)*w + e) * div
    output = torch.nan_to_num(output)
    output = output.clamp(0, 1)

    return output

def prepare_image_for_clipvision(image, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
    size = (224, 224)
    _, oh, ow, _ = image.shape
    output = image.permute([0,3,1,2])

    if crop_position == "pad":
        if oh != ow:
            if oh > ow:
                pad = (oh - ow) // 2
                pad = (pad, 0, pad, 0)
            elif ow > oh:
                pad = (ow - oh) // 2
                pad = (0, pad, 0, pad)
            output = T.functional.pad(output, pad, fill=0)
    else:
        crop_size = min(oh, ow)
        x = (ow-crop_size) // 2
        y = (oh-crop_size) // 2
        if "top" in crop_position:
            y = 0
        elif "bottom" in crop_position:
            y = oh-crop_size
        elif "left" in crop_position:
            x = 0
        elif "right" in crop_position:
            x = ow-crop_size

        x2 = x+crop_size
        y2 = y+crop_size

        output = output[:, :, y:y2, x:x2]

    imgs = []
    for img in output:
        img = T.ToPILImage()(img) # using PIL for better results
        img = img.resize(size, resample=Image.Resampling[interpolation])
        imgs.append(T.ToTensor()(img))
    output = torch.stack(imgs, dim=0)
    del imgs, img

    if sharpening > 0:
        output = contrast_adaptive_sharpening(output, sharpening)

    output = output.permute([0,2,3,1])

    return output