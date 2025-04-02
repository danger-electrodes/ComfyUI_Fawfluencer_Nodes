from PIL import Image

import cv2
import numpy as np
import torch
from collections import namedtuple
from .detectors_utils import dilate_masks, tensor2pil, combine_masks, make_crop_region, crop_ndarray2, crop_image
import inspect
import logging
from .face_restoration_helper import FaceRestoreHelper
from .LivePortraitCropper.utils.cropper import CropperInsightFace
from .LivePortraitCropper.utils.crop import _transform_img_kornia
import comfy.model_management
from facexlib.parsing import init_parsing_model
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm
import gc
import os
import copy
from .grounding_dino.datasets import transforms as T
from .sams.predictor import SamPredictorHQ
orig_torch_load = torch.load
script_directory = os.path.dirname(os.path.abspath(__file__))

MAX_RESOLUTION = 16384
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        if self.tensor is None:
            self.tensor = new_tensor
        else:
            self.tensor = torch.concat((self.tensor, new_tensor), dim=0)

def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped

def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")

def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return

def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image

def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)

def filter(segs, labels):
    labels = set([label.strip() for label in labels])

    if 'all' in labels:
        res_segs = segs
        remaining_segs = (segs[0], [])
        return res_segs, remaining_segs
    else:
        res_segs = []
        remained_segs = []

        for x in segs[1]:
            if x.label in labels:
                res_segs.append(x)
            elif 'eyes' in labels and x.label in ['left_eye', 'right_eye']:
                res_segs.append(x)
            elif 'eyebrows' in labels and x.label in ['left_eyebrow', 'right_eyebrow']:
                res_segs.append(x)
            elif 'pupils' in labels and x.label in ['left_pupil', 'right_pupil']:
                res_segs.append(x)
            else:
                remained_segs.append(x)
    res_segs = (segs[0], res_segs)
    remaining_segs = (segs[0], remained_segs)
    return res_segs, remained_segs


def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1*rw), int(y1*rw), int(x2*rh), int(y2*rh)
        bbox = int(bx1*rw), int(by1*rw), int(bx2*rh), int(by2*rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)

        if isinstance(cropped_mask, torch.Tensor) and len(cropped_mask.shape) == 3:
            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0)
        else:
            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = tensor_resize(cropped_image if isinstance(cropped_image, torch.Tensor) else torch.from_numpy(cropped_image), new_w, new_h)
            cropped_image = cropped_image.numpy()

        new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return (th, tw), new_segs

class NO_BBOX_DETECTOR:
    pass


class NO_SEGM_DETECTOR:
    pass


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


# Limit the commands that can be executed through `getattr` to `ultralytics.nn.modules.head.Detect.forward`.
def restricted_getattr(obj, name, *args):
    if name != "forward":
        logging.error(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.\nIf you believe the use of this code is genuinely safe, please report it.\nhttps://github.com/ltdrdata/ComfyUI-Impact-Subpack/issues")
        raise RuntimeError(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.")
        
    return getattr(obj, name, *args)

restricted_getattr.__module__ = 'builtins'
restricted_getattr.__name__ = 'getattr'


try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.tal import TaskAlignedAssigner
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block_modules
    import torch.nn.modules as torch_modules
    import ultralytics.utils.loss as loss_modules
    import dill._dill
    from numpy.core.multiarray import scalar
    try:
        from numpy import dtype
        from numpy.dtypes import Float64DType
    except:
        logging.error("[Impact Subpack] installed 'numpy' is outdated. Please update 'numpy' to 1.26.4")
        raise Exception("[Impact Subpack] installed 'numpy' is outdated. Please update 'numpy' to 1.26.4")


    torch_whitelist = []

    # https://github.com/comfyanonymous/ComfyUI/issues/5516#issuecomment-2466152838
    def build_torch_whitelist():
        """
        For security, only a limited set of namespaces is allowed during loading.

        Since the same module may be identified by different namespaces depending on the model,
        some modules are additionally registered with aliases to ensure backward compatibility.
        """
        global torch_whitelist

        for name, obj in inspect.getmembers(modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules"

                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(block_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules.block"

                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(loss_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.utils.loss"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.yolo.utils.loss"

                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(torch_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("torch.nn.modules"):
                torch_whitelist.append(obj)

        aliasIterableSimpleNamespace = type("IterableSimpleNamespace", (IterableSimpleNamespace,), {})
        aliasIterableSimpleNamespace.__module__ = "ultralytics.yolo.utils"

        aliasTaskAlignedAssigner = type("TaskAlignedAssigner", (TaskAlignedAssigner,), {})
        aliasTaskAlignedAssigner.__module__ = "ultralytics.yolo.utils.tal"

        aliasYOLOv10DetectionModel = type("YOLOv10DetectionModel", (DetectionModel,), {})
        aliasYOLOv10DetectionModel.__module__ = "ultralytics.nn.tasks"
        aliasYOLOv10DetectionModel.__name__ = "YOLOv10DetectionModel"

        aliasv10DetectLoss = type("v10DetectLoss", (loss_modules.E2EDetectLoss,), {})
        aliasv10DetectLoss.__name__ = "v10DetectLoss"
        aliasv10DetectLoss.__module__ = "ultralytics.utils.loss"

        torch_whitelist += [DetectionModel, aliasYOLOv10DetectionModel, SegmentationModel, IterableSimpleNamespace,
                            aliasIterableSimpleNamespace, TaskAlignedAssigner, aliasTaskAlignedAssigner, aliasv10DetectLoss,
                            restricted_getattr, dill._dill._load_type, scalar, dtype, Float64DType]

    build_torch_whitelist()

except Exception as e:
    logging.error(e)
    logging.error("\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    raise e

# HOTFIX: https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues/754
# importing YOLO breaking original torch.load capabilities
def torch_wrapper(*args, **kwargs):
    # NOTE: A trick to support code based on `'weights_only' in torch.load.__code__.co_varnames`.
    if 'weights_only' in kwargs:
        weights_only = kwargs.pop('weights_only')
    else:
        weights_only = None

    if hasattr(torch.serialization, 'safe_globals'):
        if weights_only is not None:
            kwargs['weights_only'] = weights_only

        return orig_torch_load(*args, **kwargs)  # NOTE: This code simply delegates the call to torch.load, and any errors that occur here are not the responsibility of Subpack.
    else:
        if weights_only is not None:
            kwargs['weights_only'] = weights_only
        else:
            logging.warning("[Impact Subpack] Your torch version is outdated, and security features cannot be applied properly.")
            kwargs['weights_only'] = False

        return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper


def load_yolo(model_path: str):
    # https://github.com/comfyanonymous/ComfyUI/issues/5516#issuecomment-2466152838
    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(torch_whitelist):
            try:
                return YOLO(model_path)
            except ModuleNotFoundError:
                # https://github.com/ultralytics/ultralytics/issues/3856
                YOLO("yolov8n.pt")
                return YOLO(model_path)
    else:
        try:
            return YOLO(model_path)
        except ModuleNotFoundError:
            YOLO("yolov8n.pt")
            return YOLO(model_path)


def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    else:
        # Handle the grayscale image here
        # For example, you might want to convert it to a 3-channel grayscale image for consistency:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def inference_segm(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    # NOTE: masks.data will be None when n == 0
    segms = pred[0].masks.data.cpu().numpy()

    h_segms = segms.shape[1]
    w_segms = segms.shape[2]
    h_orig = image.size[1]
    w_orig = image.size[0]
    ratio_segms = h_segms / w_segms
    ratio_orig = h_orig / w_orig

    if ratio_segms == ratio_orig:
        h_gap = 0
        w_gap = 0
    elif ratio_segms > ratio_orig:
        h_gap = int((ratio_segms - ratio_orig) * h_segms)
        w_gap = 0
    else:
        h_gap = 0
        ratio_segms = w_segms / h_segms
        ratio_orig = w_orig / h_orig
        w_gap = int((ratio_segms - ratio_orig) * w_segms)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
        mask = mask[h_gap:mask.shape[0] - h_gap, w_gap:mask.shape[1] - w_gap]

        scaled_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.size[1], image.size[0]),
                                                      mode='bilinear', align_corners=False)
        scaled_mask = scaled_mask.squeeze().squeeze()

        results[2].append(scaled_mask.numpy())
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass


class UltraSegmDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass

def get_segs(bbox_detector, faces_model_spreadsheets, threshold, dilation, crop_factor, drop_size, labels=None):
    if len(faces_model_spreadsheets) > 1:
        raise Exception('ERROR: BboxDetectorForEach does not allow image batches.')

    segs = bbox_detector.detect(faces_model_spreadsheets, threshold, dilation, crop_factor, drop_size)

    if labels is not None and labels != '':
        labels = labels.split(',')
        if len(labels) > 0:
            segs, _ = filter(segs, labels)

    return segs

def get_faces_list(bbox_detector_model_name, faces_model_spreadsheets, threshold, dilation, crop_factor, drop_size, labels):
    from .model_management import get_bbox_detector

    bbox_detector = get_bbox_detector(bbox_detector_model_name)
    cropped_list = []
    segs = get_segs(bbox_detector, faces_model_spreadsheets, threshold, dilation, crop_factor, drop_size, labels)
    segs = segs_scale_match(segs, faces_model_spreadsheets.shape)
    ordered_segs = segs[1]

    for face, seg in enumerate(ordered_segs):
        cropped_image = crop_ndarray4(faces_model_spreadsheets.cpu().numpy(), seg.crop_region)
        cropped_image = to_tensor(cropped_image)      
        orig_cropped_image = cropped_image.clone()
        cropped_list.append(orig_cropped_image)

    return cropped_list

def get_landmarks(image, face_helper, normalized=False):

    face_helper.clean_all()
    face_helper.read_image(image)
    face_helper.get_face_landmarks_5(only_center_face=True)

    landmarks_5 = face_helper.all_landmarks_5

    if normalized and landmarks_5:
        from .image import image_to_tensor
        image_tensor = image_to_tensor(image)
        width, height = image_tensor.shape[1], image_tensor.shape[0] 
        landmarks_5 = [
            [[x / width, y / height] for x, y in landmark] for landmark in landmarks_5
        ]

    return landmarks_5

def get_landmarks_dists(landmarks):

    if(len(landmarks) == 0):
        return None
    # Extract the first (and only) list of landmarks
    points = np.array(landmarks[0])  # Convert to numpy array for easier calculations
    num_points = len(points)
    num_combinations = 0
    distances = {}

    # Compute pairwise distances
    for i in range(num_points):
        for j in range(i + 1, num_points):  # Avoid redundant calculations
            p1, p2 = points[i], points[j]
            dist = np.linalg.norm(p1 - p2)  # Euclidean distance
            distances[f"point_{i}_to_point_{j}"] = dist
            num_combinations += 1

    distances['num_points'] = num_points
    distances['num_combinations'] = num_combinations

    return distances


def get_landmarks_angles(landmarks, normalized=False):

    if(len(landmarks) == 0):
        return None

    # Convert to numpy array
    points = np.array(landmarks[0])  
    num_points = len(points)
    num_combinations = 0
    angles = {}

    # Define key facial landmarks
    left_eye_idx, right_eye_idx = 0, 1   # Left & right eyes
    left_mouth_idx, right_mouth_idx = 3, 4  # Left & right mouth corners

    # Compute the midline vector
    eye_midpoint = (points[left_eye_idx] + points[right_eye_idx]) / 2
    mouth_midpoint = (points[left_mouth_idx] + points[right_mouth_idx]) / 2
    midline_vector = eye_midpoint - mouth_midpoint  

    # Normalize midline vector
    midline_norm = np.linalg.norm(midline_vector)
    if midline_norm == 0:
        return {"error": "Midline vector has zero length, cannot compute angles"}
    midline_unit = midline_vector / midline_norm  # Normalize

    # Compute angles of landmark vectors relative to the midline
    for i in range(num_points):
        for j in range(i + 1, num_points): 
            num_combinations += 1 
            p1, p2 = points[i], points[j]
            vector = p2 - p1  # Vector between points
            vector_norm = np.linalg.norm(vector)

            if vector_norm == 0:
                angles[f"point_{i}_to_point_{j}"] = 0.0
                continue  

            unit_vector = vector / vector_norm  # Normalize

            # Compute the dot product and angle
            dot_product = np.dot(unit_vector, midline_unit)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180 / np.pi)  # Convert to degrees

            if(normalized):
                angles[f"point_{i}_to_point_{j}"] = angle / 180.0
            else:
                angles[f"point_{i}_to_point_{j}"] = angle

    angles['num_points'] = num_points
    angles['num_combinations'] = num_combinations
    return angles


def get_landmarks_difference_score(land_mark_processed_1, land_mark_processed_2, normalized=False):

    if(land_mark_processed_1 is None or land_mark_processed_2 is None or land_mark_processed_1['num_points'] != land_mark_processed_2['num_points']): #not the same number of points, so score is 0
        return 1 if normalized else 1000

    num_points = land_mark_processed_1['num_points']
    num_combinations = land_mark_processed_1['num_combinations']

    if num_combinations == 0:
        return 1 if normalized else 1000
    
    score = 0

    for i in range(num_points):
        for j in range(i + 1, num_points):
            val1 = land_mark_processed_1.get(f"point_{i}_to_point_{j}", 0)
            val2 = land_mark_processed_2.get(f"point_{i}_to_point_{j}", 0)
            if(normalized):
                score += min(1.0, abs(val1 - val2)) #0 means identical 1 means opposite
            else:
                score += abs(val1 - val2)

    score /= float(num_combinations) #normalizing the score, num combination is an INT, so I wanna make sure it doesnt try to cast it in an int and round it

    return score

def get_face_helper():

    
    device = comfy.model_management.get_torch_device()

    face_helper = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    device=device,
    )

    face_helper.face_parse = None
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

    return face_helper

def process_model_spreadsheet(bbox_detector_model_name, faces_model_spreadsheet, threshold, dilation, crop_factor, drop_size, labels, face_helper, normalize_score=False):

    from .image import image_list_to_batch, tensor_to_image, image_to_tensor

    # getting the landmarks for all the faces of the model spreadsheet
    
    faces = get_faces_list(bbox_detector_model_name, faces_model_spreadsheet, threshold, dilation, crop_factor, drop_size, labels)
    faces = image_list_to_batch(faces) #tensor shape is BHWC (Batch Height Width Channel)

    faces_batch_width, faces_batch_height = faces.shape[2], faces.shape[1]
    faces_images = tensor_to_image(faces)

    model_spreadsheets_landmarks = []

    for i in range(faces_images.shape[0]):
        landmarks = get_landmarks(faces_images[i], face_helper, normalized=normalize_score)
        model_spreadsheets_landmarks.append(landmarks)

    generated_image_ratio = [1.0, 1.0]

    if(faces_batch_width > faces_batch_height):
        generated_image_ratio = [float(faces_batch_width/faces_batch_height), 1.0]
    else :
        generated_image_ratio = [1.0, float(faces_batch_height/faces_batch_width)]

    return model_spreadsheets_landmarks, generated_image_ratio, faces_images

def process_generated_image(bbox_detector_model_name, generated_image, threshold, dilation, crop_factor, drop_size, labels, face_helper, generated_image_ratio, normalize_score=False):
    from .image import image_list_to_batch, tensor_to_image

    generated_image_face = get_faces_list(bbox_detector_model_name, generated_image, threshold, dilation, crop_factor, drop_size, labels)
    generated_image_face = image_list_to_batch(generated_image_face, target_ratio=generated_image_ratio)
    generated_image_face = tensor_to_image(generated_image_face)
    generated_image_face_landmarks = get_landmarks(generated_image_face[0], face_helper, normalized=normalize_score)

    return generated_image_face_landmarks


def get_best_face_match(generated_image_face_landmarks, model_spreadsheets_landmarks, faces_images, normalize_score=False):
    
    from .image import image_to_tensor

    generated_image_angles = get_landmarks_angles(generated_image_face_landmarks, normalize_score)
    
    scores = []
    for current_landmark in model_spreadsheets_landmarks:
        current_computed_angles = get_landmarks_angles(current_landmark, normalize_score)
        score = get_landmarks_difference_score(generated_image_angles, current_computed_angles, normalize_score)
        print(f"SCORE : {score}")
        scores.append(score)

    max_score = min(scores)
    max_score_index = scores.index(max_score)
    print(f"BEST SCORE INDEX : {max_score_index}")
    closest_face = image_to_tensor(faces_images)[max_score_index].unsqueeze(0)

    return closest_face

def draw_pointsOnImg(image, landmarks, color=(255, 0, 0), radius=3):
    image_cpy = image.copy()
    for n in range(landmarks.shape[0]):
        try:
            cv2.circle(image_cpy, (int(landmarks[n][0]), int(landmarks[n][1])), radius, color, -1)                        
        except:
                pass
    return image_cpy

def drawLineBetweenPoints(image, pointsA, pointsB, color=(255, 0, 0), thickness=1):
    image_cpy = image.copy()
    for n in range(pointsA.shape[0]):
        try:
            cv2.line(image_cpy, (int(pointsA[n][0]), int(pointsA[n][1])), (int(pointsB[n][0]), int(pointsB[n][1])), color, thickness)                        
        except:
            pass
    return image_cpy

def load_insightface_cropper():
    cropper_init_config = {
        'keep_model_loaded': True,
        'onnx_device': "CPU",
        'detection_threshold': 0.50
    }
    
    cropper = CropperInsightFace(**cropper_init_config)

    return cropper

def face_cropper(cropper, source_image, dsize,scale, vx_ratio, vy_ratio, face_index, face_index_order, rotate):
    source_image_np = (source_image.contiguous() * 255).byte().numpy()

    crop_info_list = []
    cropped_images_list = []
    for i in tqdm(range(len(source_image_np)), desc='Detecting, cropping, and processing..', total=len(source_image_np)):
        # Cropping operation
        crop_info, cropped_image = cropper.crop_single_image(source_image_np[i], dsize, scale, vy_ratio, vx_ratio, face_index, face_index_order, rotate)
        
        # Processing source images
        if crop_info:
            crop_info['dsize'] = dsize
            crop_info_list.append(crop_info)

            cropped_images_list.append(cropped_image)
            
        else:
            cropped_images_list.append(np.zeros((256, 256, 3), dtype=np.uint8))
            crop_info_list.append(None)

    cropped_tensors_out = (
        torch.stack([torch.from_numpy(np_array) for np_array in cropped_images_list])
        / 255
    )
    
    crop_info_dict = {
        'crop_info_list': crop_info_list,
    }
    
    return cropped_tensors_out, crop_info_dict

def LandMark203_to_68(source):
    #jawLine
    out = [source[108]]
    out.append( (source[108+2]*3 + source[108+3] )/4)
    out.append((source[108+4]+source[108+5])/2)
    out.append( (source[108+6] + source[108+7]*3 )/4)
    out.append(source[108+9])
    out.append( (source[108+11]*3 + source[108+12] )/4) 
    out.append((source[108+13]+source[108+14])/2)
    out.append( (source[108+15] + source[108+16]*3 )/4)

    #for i in range(0,7):
    #    out.append((source[110+i*2]+source[111+i*2])/2)  
    out.append(source[126])
    # for i in range(0,7):
    #     out.append((source[128+i*2]+source[129+i*2])/2)  
    out.append( (source[126+2]*3 + source[126+3] )/4)
    out.append((source[126+4]+source[126+5])/2)
    out.append( (source[126+6] + source[126+7]*3 )/4)
    out.append(source[126+9])
    out.append( (source[126+11]*3 + source[126+12] )/4) 
    out.append((source[126+13]+source[126+14])/2)
    out.append( (source[126+15] + source[126+16]*3 )/4)

    out.append(source[144])   
    #leftEyeBow
    out.append(source[145])
    out.append((source[148]+source[162])/2)           
    out.append((source[150]+source[160])/2)   
    out.append((source[152]+source[158])/2)   
    out.append(source[155])
    #rightEyeBow
    out.append(source[165])
    out.append((source[168]+source[182])/2)           
    out.append((source[170]+source[180])/2)   
    out.append((source[172]+source[177])/2)   
    out.append(source[175])

    #nose
    out.append(source[199])
    out.append((source[199]+source[200])/2)           
    out.append(source[200])
    out.append(source[201])
    out.append(source[189])
    out.append(source[190])
    out.append(source[202])
    out.append(source[191])
    out.append(source[192])
    
    #leftEye
    out.append(source[0])
    out.append(source[3])
    out.append(source[8])
    out.append(source[12])
    out.append(source[16])
    out.append(source[21])

    #rightEye
    out.append(source[24])
    out.append(source[28])
    out.append(source[33])
    out.append(source[36])
    out.append(source[39])
    out.append(source[45])

    #UpperLipUp
    out.append(source[48])
    out.append(source[51])
    out.append(source[54])
    out.append(source[57])
    out.append(source[60])
    out.append(source[63])
    out.append(source[66])

    #LowerLipDown
    out.append(source[69])
    out.append(source[72])
    out.append(source[75])
    out.append(source[78])
    out.append(source[81])


    out.append(source[84])
    out.append(source[87])
    out.append(source[90])
    out.append(source[93])
    out.append(source[96])

    out.append(source[99])
    out.append(source[102])
    out.append(source[105])

    return out

def face_shaper(source_image,source_crop_info, target_crop_info, landmarkType, AlignType):
    tensor1 = source_image*255
    tensor1 = np.array(tensor1, dtype=np.uint8)

    image1 = tensor1[0]

    height,width = image1.shape[:2]
    w=width
    h=height
    landmarks1 = source_crop_info["crop_info_list"][0]['lmk_crop']
    landmarks2 = target_crop_info["crop_info_list"][0]['lmk_crop']

    use_68_points=True
    if(use_68_points):
        landmarks1 = LandMark203_to_68(landmarks1)
        landmarks2 = LandMark203_to_68(landmarks2)
        landmarks1 = landmarks1[0:65]
        landmarks2 = landmarks2[0:65]


    if(use_68_points):
        leftEye1=np.mean( landmarks1[36:42],axis=0)
        rightEye1=np.mean( landmarks1[42:48],axis=0)
        leftEye2=np.mean( landmarks2[36:42],axis=0)
        rightEye2=np.mean( landmarks2[42:48],axis=0)
        jaw1=landmarks1[0:17]
        jaw2=landmarks2[0:17]
        centerOfJaw1=np.mean( jaw1,axis=0)
        centerOfJaw2=np.mean( jaw2,axis=0)   
    else:                            
        leftEye1=np.mean( landmarks1[0:24],axis=0)
        rightEye1=np.mean( landmarks1[24:48],axis=0)
        leftEye2=np.mean( landmarks2[0:24],axis=0)
        rightEye2=np.mean( landmarks2[24:48],axis=0)
        jaw1=landmarks1[108:145]
        jaw2=landmarks2[108:145]
        centerOfJaw1=np.mean( jaw1,axis=0)
        centerOfJaw2=np.mean( jaw2,axis=0)

    src_points = np.array([
        [x, y]
        for x in np.linspace(0, w, 16)
        for y in np.linspace(0, h, 16)
    ])
    
    src_points = src_points[(src_points[:, 0] <= w/8) | (src_points[:, 0] >= 7*w/8) |  (src_points[:, 1] >= 7*h/8)| (src_points[:, 1] <= h/8)]            
    dst_points = src_points.copy()


    landmarks2=np.array(landmarks2)
    min_x = np.min(landmarks2[:, 0])
    max_x = np.max(landmarks2[:, 0])
    min_y = np.min(landmarks2[:, 1])
    max_y = np.max(landmarks2[:, 1])
    
    ratio2 = (max_x - min_x) / (max_y - min_y)

    landmarks1=np.array(landmarks1)
    min_x = np.min(landmarks1[:, 0])
    max_x = np.max(landmarks1[:, 0])
    min_y = np.min(landmarks1[:, 1])
    max_y = np.max(landmarks1[:, 1])

    ratio1 = (max_x - min_x) / (max_y - min_y)
    middlePoint = [ (max_x + min_x) / 2, (max_y + min_y) / 2]

    #print("ratio1",ratio1)
    

    if AlignType=="Width":  
        if(landmarkType=="ALL"):  
            dst_points = np.append(dst_points,landmarks1,axis=0)                  
            target_points = landmarks1.copy()                                        
        else:
            dst_points = np.append(dst_points,jaw1,axis=0)
            jaw1=np.array(jaw1)
            target_points = jaw1.copy() 
        target_points[:, 1] = (target_points[:, 1] - middlePoint[1]) * ratio1 / ratio2 + middlePoint[1]
        src_points = np.append(src_points,target_points,axis=0)

    elif AlignType=="Height":
        if(landmarkType=="ALL"):  
            dst_points = np.append(dst_points,landmarks1,axis=0)             
            target_points = landmarks1.copy()                                        
        else:
            dst_points = np.append(dst_points,jaw1,axis=0)
            jaw1=np.array(jaw1)
            target_points = jaw1.copy() 
        target_points[:, 0] = (target_points[:, 0] - middlePoint[0]) * ratio2 / ratio1 + middlePoint[0]
        src_points = np.append(src_points,target_points,axis=0)

    elif AlignType=="Landmarks":
        if(landmarkType=="ALL"):
            MiddleOfEyes1 = (leftEye1+rightEye1)/2
            MiddleOfEyes2 = (leftEye2+rightEye2)/2
            distance1 =  ((leftEye1[0] - rightEye1[0]) ** 2 + (leftEye1[1] - rightEye1[1]) ** 2) ** 0.5
            distance2 =  ((leftEye2[0] - rightEye2[0]) ** 2 + (leftEye2[1] - rightEye2[1]) ** 2) ** 0.5
            factor = distance1 / distance2
            MiddleOfEyes2 = np.array(MiddleOfEyes2)
            target_points = (landmarks2 - MiddleOfEyes2) * factor + MiddleOfEyes1

            centerOfJaw2 = np.array(centerOfJaw2)
            jawLineTarget = (landmarks2[108:144] - centerOfJaw2) * factor + centerOfJaw1
            target_points[108:144] = jawLineTarget

            dst_points = np.append(dst_points,landmarks1,axis=0)
        else:
            dst_points = np.append(dst_points,jaw1,axis=0)
            target_points=(jaw2-centerOfJaw2)+centerOfJaw1
        src_points = np.append(src_points,target_points,axis=0)


    elif AlignType=="JawLine":
        lenOfJaw=len(jaw1)
        distance1=  ((jaw1[0][0] - jaw1[lenOfJaw-1][0]) ** 2 + (jaw1[0][1] - jaw1[lenOfJaw-1][1]) ** 2) ** 0.5
        distance2=  ((jaw2[0][0] - jaw2[lenOfJaw-1][0]) ** 2 + (jaw2[0][1] - jaw2[lenOfJaw-1][1]) ** 2) ** 0.5
        factor = distance1 / distance2
        if landmarkType == "ALL":
            dst_points = np.append(dst_points,landmarks1,axis=0)
            target_points=(landmarks2-jaw2[0])*factor+jaw1[0]
            src_points = np.append(src_points,target_points,axis=0)
        else:
            dst_points = np.append(dst_points,jaw1,axis=0)
            target_points=(jaw2-jaw2[0])*factor+jaw1[0]
            src_points = np.append(src_points,target_points,axis=0)
    
    mark_img = draw_pointsOnImg(image1, dst_points, color=(255, 255, 0),radius=4)
    mark_img = draw_pointsOnImg(mark_img, src_points, color=(255, 0, 0),radius=3)
    mark_img = drawLineBetweenPoints(mark_img, dst_points,src_points)
                
    src_points[:, [0, 1]] = src_points[:, [1, 0]]
    dst_points[:, [0, 1]] = dst_points[:, [1, 0]]

    rbfy = RBFInterpolator(src_points,dst_points[:,1],kernel="thin_plate_spline")
    rbfx = RBFInterpolator(src_points,dst_points[:,0],kernel="thin_plate_spline")

    img_grid = np.mgrid[0:height, 0:width]

    flatten=img_grid.reshape(2, -1).T

    map_y = rbfy(flatten).reshape(height,width).astype(np.float32)
    map_x = rbfx(flatten).reshape(height,width).astype(np.float32)
    warped_image = cv2.remap(image1, map_y, map_x, cv2.INTER_LINEAR)

    warped_image = torch.from_numpy(warped_image.astype(np.float32) / 255.0).unsqueeze(0)               
    mark_img = torch.from_numpy(mark_img.astype(np.float32) / 255.0).unsqueeze(0)  

    return warped_image, mark_img

def face_shaper_composite(source_image, cropped_image, crop_info, mask=None):
    comfy.model_management.soft_empty_cache()
    gc.collect()
    device = comfy.model_management.get_torch_device()
    if comfy.model_management.is_device_mps(device): 
        device = torch.device('cpu') #this function returns NaNs on MPS, defaulting to CPU

    B, H, W, C = source_image.shape
    source_image = source_image.permute(0, 3, 1, 2) # B,H,W,C -> B,C,H,W
    #cropped_image = cropped_image.permute(0, 3, 1, 2)

    if mask is not None:
        if len(mask.size())==2:
            crop_mask = mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)
        else:    
            crop_mask = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
    else:
        crop_mask = cv2.imread(os.path.join(script_directory, "LivePortraitCropper", "utils", "resources", "mask_template.png"), cv2.IMREAD_COLOR)
        crop_mask = torch.from_numpy(crop_mask)
        crop_mask = crop_mask.unsqueeze(0).float() / 255.0

    composited_image_list = []
    out_mask_list = []

    total_frames = len(crop_info["crop_info_list"])

    pbar = comfy.utils.ProgressBar(total_frames)
    for i in tqdm(range(total_frames), desc='Compositing..', total=total_frames):
        safe_index = min(i, len(crop_info["crop_info_list"]) - 1)

        source_frame = source_image[safe_index].unsqueeze(0).to(device)

        croppedImage = cropped_image[safe_index].unsqueeze(0).to(device)

        cropped_image_to_original = _transform_img_kornia(
            croppedImage,
            crop_info["crop_info_list"][safe_index]["M_c2o"],
            dsize=(W, H),
            device=device
            )
        
        mask_ori = _transform_img_kornia(
            crop_mask[0].unsqueeze(0),
            crop_info["crop_info_list"][safe_index]["M_c2o"],
            dsize=(W, H),
            device=device
            )
        
        cropped_image_to_original_blend = torch.clip(
                mask_ori * cropped_image_to_original + (1 - mask_ori) * source_frame, 0, 1
                )

        composited_image_list.append(cropped_image_to_original_blend.cpu())
        out_mask_list.append(mask_ori.cpu())

        pbar.update(1)

    full_tensors_out = torch.cat(composited_image_list, dim=0)
    full_tensors_out = full_tensors_out.permute(0, 2, 3, 1)

    mask_tensors_out = torch.cat(out_mask_list, dim=0)
    mask_tensors_out = mask_tensors_out[:, 0, :, :]
    
    final_image = full_tensors_out.float()
    final_mask = mask_tensors_out.float()
    return final_image, final_mask


def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold
):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt

def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)

def sam_segment(
    sam_model,
    image,
    boxes
):
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    sam_device = comfy.model_management.get_torch_device()
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    return create_tensor_output(image_np, masks, boxes)

def segment_anything(grounding_dino_model, sam_model, image, prompt, threshold, attempts=10):
    from .model_management import unload_models
    
    for i in range(attempts):
        try:
            res_images = []
            res_masks = []
            for item in image:
                item = Image.fromarray(
                    np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
                boxes = groundingdino_predict(
                    grounding_dino_model,
                    item,
                    prompt,
                    threshold
                )
                if boxes.shape[0] == 0:
                    break
                (images, masks) = sam_segment(
                    sam_model,
                    item,
                    boxes
                )
                res_images.extend(images)
                res_masks.extend(masks)
            if len(res_images) == 0:
                _, height, width, _ = image.size()
                empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
                return empty_mask, empty_mask

            final_image = torch.cat(res_images, dim=0)
            final_mask = torch.cat(res_masks, dim=0)
            break
        except torch.OutOfMemoryError:  # Catch only PyTorch OOM
            unload_models()
            if i == attempts - 1:
                raise Exception("All attempts to sample have failed, try increasing the value of 'sampling_attempt_number'.")
            
    return final_image, final_mask