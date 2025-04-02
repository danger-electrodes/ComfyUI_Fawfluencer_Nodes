import torch
import numpy as np
from skimage import morphology
from pathlib import Path
import os
from .utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management
from .custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
from .custom_controlnet_aux.sam import SamDetector
from .custom_controlnet_aux.tile import TileDetector
from .custom_controlnet_aux.open_pose import OpenposeDetector
from .custom_controlnet_aux.normalbae import NormalBaeDetector
from .custom_controlnet_aux.teed import TEDDetector
from .custom_controlnet_aux.teed.ted import TED
from .custom_controlnet_aux.lineart_standard import LineartStandardDetector
from .custom_controlnet_aux.metric3d import Metric3DDetector
from ...fawfulized_nodes.utilities import model_management as model_management_utilities
#get more preprocessors from custom_nodes/comfyui_controlnet_aux/node_wrappers

def load_mteed_model():
    current_dir = Path(__file__).resolve().parent
    checkpoint_path = current_dir / "custom_controlnet_aux" / "anyline" / "checkpoints" / "Anyline" / "Anyline" / "MTEED.pth"
    model = TED()
    model.load_state_dict(torch.load(checkpoint_path, map_location=model_management.get_torch_device()))
    model.eval()
    return model

def get_intensity_mask(image_array, lower_bound, upper_bound):
    mask = image_array[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    return mask

def combine_layers(base_layer, top_layer):
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    result = base_layer * (~mask) + temp * mask
    return result

#DEPTH
def depth(image):
    model = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(model_management.get_torch_device())
    out = common_annotator_call(model, image, resolution=512, max_depth=1)
    del model
    return out

#SEGMENT
def segment(image):
    mobile_sam = SamDetector.from_pretrained().to(model_management.get_torch_device())
    out = common_annotator_call(mobile_sam, image, resolution=512)
    del mobile_sam
    return out

#TILE
def tile(image):
    out = common_annotator_call(TileDetector(), image, pyrUp_iters=3, resolution=512)
    return out

#OPENPOSE
def openpose(image):
    detect_hand = True
    detect_body = True
    detect_face = True
    scale_stick_for_xinsr_cn = False

    model = OpenposeDetector.from_pretrained().to(model_management.get_torch_device())        
    openpose_dicts = []
    def func(image, **kwargs):
        pose_img, openpose_dict = model(image, **kwargs)
        openpose_dicts.append(openpose_dict)
        return pose_img
    
    out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, xinsr_stick_scaling=scale_stick_for_xinsr_cn, resolution=512)
    del model
    return out

#NORMAL
def normal(image):
    model = NormalBaeDetector.from_pretrained().to(model_management.get_torch_device())
    out = common_annotator_call(model, image, resolution=512)
    del model
    return out

#ANYLINE
def anyline(image):
    # Process the image with MTEED model
    mteed_model = TEDDetector(model=load_mteed_model()).to(model_management.get_torch_device())
    mteed_result = common_annotator_call(mteed_model, image, resolution=1280)
    mteed_result = mteed_result.squeeze(0).numpy()


    # Process the image with the lineart standard preprocessor
    lineart_standard_detector = LineartStandardDetector()
    lineart_result  = common_annotator_call(lineart_standard_detector, image, guassian_sigma=2, intensity_threshold=3, resolution=1280).squeeze().numpy()
    lineart_result  = get_intensity_mask(lineart_result, lower_bound=0, upper_bound=1)
    cleaned = morphology.remove_small_objects(lineart_result .astype(bool), min_size=36, connectivity=1)
    lineart_result = lineart_result *cleaned

    # Combine the results
    final_result = combine_layers(mteed_result, lineart_result)

    del mteed_model
    return torch.tensor(final_result).unsqueeze(0)

def metric3Dnormalmap(image):
        model = Metric3DDetector.from_pretrained(filename="metric_depth_vit_small_800k.pth").to(model_management.get_torch_device())
        cb = lambda image, **kwargs: model(image, **kwargs)[1]
        out = common_annotator_call(cb, image, resolution=512, fx=1000, fy=1000, depth_and_normal=True)
        del model
        return out

#CALLER FUNCTION
def process_image(preprocessor, image):
    args = {"image": image}
    out = globals()[preprocessor](**args)
    return out

def try_process_image(preprocessor, image, attempts=10):
    for i in range(attempts):
        try:
            processed_image = process_image(preprocessor, image)
            break
        except torch.OutOfMemoryError:  # Catch only PyTorch OOM
            model_management_utilities.unload_models()
            if i == attempts - 1:
                raise Exception("All attempts to sample have failed, try increasing the value of 'sampling_attempt_number'.")
    return processed_image