import folder_paths
import os
import numpy as np
from .image import pil2tensor
import cv2
import torch
import requests
import tqdm
import comfy.utils
import yaml
import copy
from collections import OrderedDict
from ultralytics import YOLO

from .LivePortrait.live_portrait_wrapper import LivePortraitWrapper
from .LivePortrait.utils.camera import get_rotation_matrix
from .LivePortrait.config.inference_config import InferenceConfig

from .LivePortrait.modules.spade_generator import SPADEDecoder
from .LivePortrait.modules.warping_network import WarpingNetwork
from .LivePortrait.modules.motion_extractor import MotionExtractor
from .LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from .LivePortrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork

import random

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

cur_device = None

def get_device():
    global cur_device
    if cur_device == None:
        if torch.cuda.is_available():
            cur_device = torch.device('cuda')
            print("Uses CUDA device.")
        elif torch.backends.mps.is_available():
            cur_device = torch.device('mps')
            print("Uses MPS device.")
        else:
            cur_device = torch.device('cpu')
            print("Uses CPU device.")
    return cur_device


def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]

def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]

def create_transform_matrix(x, y, s_x, s_y):
    return np.float32([[s_x, 0, x], [0, s_y, y]])

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

def retargeting(delta_out, driving_exp, factor, idxes):
    for idx in idxes:
        #delta_out[0, idx] -= src_exp[0, idx] * factor
        delta_out[0, idx] += driving_exp[0, idx] * factor
        
class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori

class LP_Engine:
    pipeline = None
    detect_model = None
    mask_img = None
    temp_img_idx = 0

    def get_temp_img_name(self):
        self.temp_img_idx += 1
        return "expression_edit_preview" + str(self.temp_img_idx) + ".png"

    def download_model(_, file_path, model_url):
        print('AdvancedLivePortrait: Downloading model...')
        response = requests.get(model_url, stream=True)
        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                # tqdm will display a progress bar
                with open(file_path, 'wb') as file, tqdm(
                        desc='Downloading',
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)

        except requests.exceptions.RequestException as err:
            print('AdvancedLivePortrait: Model download failed: {err}')
            print(f'AdvancedLivePortrait: Download it manually from: {model_url}')
            print(f'AdvancedLivePortrait: And put it in {file_path}')
        except Exception as e:
            print(f'AdvancedLivePortrait: An unexpected error occurred: {e}')

    def remove_ddp_dumplicate_key(_, state_dict):
        state_dict_new = OrderedDict()
        for key in state_dict.keys():
            state_dict_new[key.replace('module.', '')] = state_dict[key]
        return state_dict_new

    def filter_for_model(_, checkpoint, prefix):
        filtered_checkpoint = {key.replace(prefix + "_module.", ""): value for key, value in checkpoint.items() if
                               key.startswith(prefix)}
        return filtered_checkpoint

    def load_model(self, model_config, model_type):

        device = get_device()

        if model_type == 'stitching_retargeting_module':
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "retargeting_models", model_type + ".pth")
        else:
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "base_models", model_type + ".pth")

        is_safetensors = None
        if os.path.isfile(ckpt_path) == False:
            is_safetensors = True
            ckpt_path = os.path.join(get_model_dir("liveportrait"), model_type + ".safetensors")
            if os.path.isfile(ckpt_path) == False:
                self.download_model(ckpt_path,
                "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/" + model_type + ".safetensors")
        model_params = model_config['model_params'][f'{model_type}_params']
        if model_type == 'appearance_feature_extractor':
            model = AppearanceFeatureExtractor(**model_params).to(device)
        elif model_type == 'motion_extractor':
            model = MotionExtractor(**model_params).to(device)
        elif model_type == 'warping_module':
            model = WarpingNetwork(**model_params).to(device)
        elif model_type == 'spade_generator':
            model = SPADEDecoder(**model_params).to(device)
        elif model_type == 'stitching_retargeting_module':
            # Special handling for stitching and retargeting module
            config = model_config['model_params']['stitching_retargeting_module_params']
            checkpoint = comfy.utils.load_torch_file(ckpt_path)

            stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
            if is_safetensors:
                stitcher.load_state_dict(self.filter_for_model(checkpoint, 'retarget_shoulder'))
            else:
                stitcher.load_state_dict(self.remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
            stitcher = stitcher.to(device)
            stitcher.eval()

            return {
                'stitching': stitcher,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


        model.load_state_dict(comfy.utils.load_torch_file(ckpt_path))
        model.eval()
        return model

    def load_models(self):
        model_path = get_model_dir("liveportrait")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_config_path = os.path.join(current_directory, 'LivePortrait', 'config', 'models.yaml')
        model_config = yaml.safe_load(open(model_config_path, 'r'))

        appearance_feature_extractor = self.load_model(model_config, 'appearance_feature_extractor')
        motion_extractor = self.load_model(model_config, 'motion_extractor')
        warping_module = self.load_model(model_config, 'warping_module')
        spade_generator = self.load_model(model_config, 'spade_generator')
        stitching_retargeting_module = self.load_model(model_config, 'stitching_retargeting_module')

        self.pipeline = LivePortraitWrapper(InferenceConfig(), appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module)

    def get_detect_model(self):
        if self.detect_model == None:
            model_dir = get_model_dir("ultralytics")
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            model_path = os.path.join(model_dir, "face_yolov8n.pt")
            if not os.path.exists(model_path):
                self.download_model(model_path, "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            self.detect_model = YOLO(model_path)

        return self.detect_model

    def get_face_bboxes(self, image_rgb):
        detect_model = self.get_detect_model()
        pred = detect_model(image_rgb, conf=0.7, device="")
        return pred[0].boxes.xyxy.cpu().numpy()

    def detect_face(self, image_rgb, crop_factor, sort = True):
        bboxes = self.get_face_bboxes(image_rgb)
        w, h = get_rgb_size(image_rgb)

        print(f"w, h:{w, h}")

        cx = w / 2
        min_diff = w
        best_box = None
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            if bbox_w < 30: continue
            diff = abs(cx - (x1 + bbox_w / 2))
            if diff < min_diff:
                best_box = [x1, y1, x2, y2]
                print(f"diff, min_diff, best_box:{diff, min_diff, best_box}")
                min_diff = diff

        if best_box == None:
            print("Failed to detect face!!")
            return [0, 0, w, h]

        x1, y1, x2, y2 = best_box

        #for x1, y1, x2, y2 in bboxes:
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        crop_w = bbox_w * crop_factor
        crop_h = bbox_h * crop_factor

        crop_w = max(crop_h, crop_w)
        crop_h = crop_w

        kernel_x = int(x1 + bbox_w / 2)
        kernel_y = int(y1 + bbox_h / 2)

        new_x1 = int(kernel_x - crop_w / 2)
        new_x2 = int(kernel_x + crop_w / 2)
        new_y1 = int(kernel_y - crop_h / 2)
        new_y2 = int(kernel_y + crop_h / 2)

        if not sort:
            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        elif w < new_x2:
            new_x1 -= (new_x2 - w)
            new_x2 = w
            if new_x1 < 0:
                new_x2 -= new_x1
                new_x1 = 0

        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        elif h < new_y2:
            new_y1 -= (new_y2 - h)
            new_y2 = h
            if new_y1 < 0:
                new_y2 -= new_y1
                new_y1 = 0

        if w < new_x2 and h < new_y2:
            over_x = new_x2 - w
            over_y = new_y2 - h
            over_min = min(over_x, over_y)
            new_x2 -= over_min
            new_y2 -= over_min

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


    def calc_face_region(self, square, dsize):
        region = copy.deepcopy(square)
        is_changed = False
        if dsize[0] < region[2]:
            region[2] = dsize[0]
            is_changed = True
        if dsize[1] < region[3]:
            region[3] = dsize[1]
            is_changed = True

        return region, is_changed

    def expand_img(self, rgb_img, square):
        #new_img = rgb_crop(rgb_img, face_region)
        crop_trans_m = create_transform_matrix(max(-square[0], 0), max(-square[1], 0), 1, 1)
        new_img = cv2.warpAffine(rgb_img, crop_trans_m, (square[2] - square[0], square[3] - square[1]),
                                        cv2.INTER_LINEAR)
        return new_img

    def get_pipeline(self):
        if self.pipeline == None:
            print("Load pipeline...")
            self.load_models()

        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256,256]
        if h != input_shape[0] or w != input_shape[1]:
            if 256 < h: interpolation = cv2.INTER_AREA
            else: interpolation = cv2.INTER_LINEAR
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation = interpolation)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(get_device())
        return x

    def GetMaskImg(self):
        if self.mask_img is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./LivePortrait/utils/resources/mask_template.png")
            self.mask_img = cv2.imread(path, cv2.IMREAD_COLOR)
        return self.mask_img

    def crop_face(self, img_rgb, crop_factor):
        crop_region = self.detect_face(img_rgb, crop_factor)
        face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))
        face_img = rgb_crop(img_rgb, face_region)
        if is_changed: face_img = self.expand_img(face_img, crop_region)
        return face_img

    def prepare_source(self, source_image, crop_factor, is_video = False, tracking = False):
        print("Prepare source...")
        engine = self.get_pipeline()
        source_image_np = (source_image * 255).byte().numpy()
        img_rgb = source_image_np[0]

        psi_list = []
        for img_rgb in source_image_np:
            if tracking or len(psi_list) == 0:
                crop_region = self.detect_face(img_rgb, crop_factor)
                face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))

                s_x = (face_region[2] - face_region[0]) / 512.
                s_y = (face_region[3] - face_region[1]) / 512.
                crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s_x, s_y)
                mask_ori = cv2.warpAffine(self.GetMaskImg(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
                mask_ori = mask_ori.astype(np.float32) / 255.

                if is_changed:
                    s = (crop_region[2] - crop_region[0]) / 512.
                    crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s, s)

            face_img = rgb_crop(img_rgb, face_region)
            if is_changed: face_img = self.expand_img(face_img, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = engine.get_kp_info(i_s)
            f_s_user = engine.extract_feature_3d(i_s)
            x_s_user = engine.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        pipeline = self.get_pipeline()
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = self.prepare_src_image(f_img)
            d_info = pipeline.get_kp_info(i_d)
            out_list.append(d_info)

        return out_list

    def calc_fe(_, x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile,
                rotate_pitch, rotate_yaw, rotate_roll):

        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003
        x_d_new[0, 1, 1] += eyes * -0.00025
        x_d_new[0, 2, 1] += eyes * 0.00025


        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003


        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])


g_engine = LP_Engine()

class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(get_device())
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

def randomize_expressions(random_expression_multiplier = 1.0):
    blink = random.uniform(-20.0, 5.0) * random_expression_multiplier
    eyebrow = random.uniform(-10.0, 15.0) * random_expression_multiplier
    wink = random.uniform(0.0, 25.0) * random_expression_multiplier
    pupil_x = random.uniform(-15.0, 15.0) * random_expression_multiplier
    pupil_y = random.uniform(-15.0, 15.0) * random_expression_multiplier
    aaa = random.uniform(-30.0, 120.0) * random_expression_multiplier
    eee = random.uniform(-20.0, 15.0) * random_expression_multiplier
    woo = random.uniform(-20.0, 15.0) * random_expression_multiplier
    smile = random.uniform(-0.30, 1.30) * random_expression_multiplier

    return blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile

def create_expression(src_image, sample_image=None, random_expression=False, random_expression_multiplier = 1.0, rotate_pitch = 0.0, rotate_yaw = 0.0, rotate_roll = 0.0, blink = 0.0, eyebrow = 0.0, wink = 0.0, pupil_x = 0.0, pupil_y = 0.0, aaa = 0.0, eee = 0.0, woo = 0.0, smile = 0.0,
        src_ratio=1.0, sample_ratio=1.0, sample_parts="OnlyExpression", crop_factor=1.7):
    
    if(random_expression):
        blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile = randomize_expressions(random_expression_multiplier)
    
    rotate_yaw = -rotate_yaw

    crop_factor = crop_factor
    psi = g_engine.prepare_source(src_image, crop_factor)

    pipeline = g_engine.get_pipeline()
    s_info = psi.x_s_info
    #delta_new = copy.deepcopy()
    s_exp = s_info['exp'] * src_ratio
    s_exp[0, 5] = s_info['exp'][0, 5]
    s_exp += s_info['kp']

    es = ExpressionSet()

    if sample_image != None:
        d_image_np = (sample_image * 255).byte().numpy()
        d_face = g_engine.crop_face(d_image_np[0], 1.7)
        i_d = g_engine.prepare_src_image(d_face)
        d_info = pipeline.get_kp_info(i_d)
        d_info['exp'][0, 5, 0] = 0
        d_info['exp'][0, 5, 1] = 0

        # "OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"
        if sample_parts == "OnlyExpression" or sample_parts == "All":
            es.e += d_info['exp'] * sample_ratio
        if sample_parts == "OnlyRotation" or sample_parts == "All":
            rotate_pitch += d_info['pitch'] * sample_ratio
            rotate_yaw += d_info['yaw'] * sample_ratio
            rotate_roll += d_info['roll'] * sample_ratio
        elif sample_parts == "OnlyMouth":
            retargeting(es.e, d_info['exp'], sample_ratio, (14, 17, 19, 20))
        elif sample_parts == "OnlyEyes":
            retargeting(es.e, d_info['exp'], sample_ratio, (1, 2, 11, 13, 15, 16))

    es.r = g_engine.calc_fe(es.e, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
                                rotate_pitch, rotate_yaw, rotate_roll)

    new_rotate = get_rotation_matrix(s_info['pitch'] + es.r[0], s_info['yaw'] + es.r[1],
                                        s_info['roll'] + es.r[2])
    x_d_new = (s_info['scale'] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info['t']

    x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)

    crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
    crop_out = pipeline.parse_output(crop_out['out'])[0]

    crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
    out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

    out_img = pil2tensor(out)

    return out_img