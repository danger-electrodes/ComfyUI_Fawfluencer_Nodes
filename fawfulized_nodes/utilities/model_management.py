import os
import re
import copy
import torch
from huggingface_hub import snapshot_download, hf_hub_download
import comfy.model_management
import comfy.controlnet
from comfy.clip_vision import clip_preprocess
from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
from insightface.utils.download import download_file
from insightface.utils.storage import BASE_REPO_URL
from insightface.app import FaceAnalysis
from .eva_clip.factory import create_model_and_transforms
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .pulid_flux import load_pulid_flux_model_patcher, apply_pulid_flux
from .pulid import load_pulid_model_patcher, apply_pulid
from .ic_light import ICLight
from .ic_light_convert_unet import convert_iclight_unet
from .ic_light_patches import calculate_weight_adjust_channel
from comfy.utils import load_torch_file
import zipfile
import folder_paths
from .detectors import load_yolo, UltraBBoxDetector
from .grounding_dino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from comfy import lora
import types
import glob
from .grounding_dino.util.utils import clean_state_dict
from .grounding_dino.models import build_model
from .sams.build_sam_hq import sam_model_registry
from .flux_clip import FluxClipViT
from comfy.clip_vision import load as load_clip_vision
from safetensors import safe_open
from .flux_layers import (DoubleStreamBlockLoraProcessor,
                     DoubleStreamBlockProcessor,
                     DoubleStreamBlockLorasMixerProcessor,
                     DoubleStreamMixerProcessor,
                     IPProcessor,
                     ImageProjModel)
from .flux_utils import (FirstHalfStrengthModel, FluxUpdateModules, LinearStrengthModel, 
                SecondHalfStrengthModel, SigmoidStrengthModel, attn_processors, 
                set_attn_processor,
                is_model_patched, merge_loras, LATENT_PROCESSOR_COMFY,
                ControlNetContainer,
                comfy_to_xlabs_lora, check_is_comfy_lora)
from comfy.sd import load_lora_for_models
from .CrossAttentionPatch import Attn2Replace, ipadapter_attention

def extract_version(file_name):
    version_pattern = re.compile(r'(\d+)\.(\d+)\.(\d+)')
    match = version_pattern.search(file_name)
    if match:
        return tuple(map(int, match.groups()))  # Convert version to a tuple (X, Y, Z)
    return (0, 0, 0)  # Default version if no match

def get_folder(current_dir, folder_rel_path):
    parent_dir = os.path.dirname(current_dir)
    folder_path = os.path.join(parent_dir, folder_rel_path)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_model_folder(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = get_folder(current_dir, f'models\{model_name}')
    return model_folder

def load_model(model_name, repo_id, filename=None, revision=None):
    model_folder = get_model_folder(model_name)
    model_path = os.path.join(model_folder, repo_id)

    if(filename is not None):
        model_path = os.path.join(model_path, filename, filename)

    if not os.path.exists(model_path):
        if(filename is None):
            snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False, revision=revision)
        else:
            hf_hub_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False, filename=filename, revision=revision)

    return model_path

def download_insightface_model(root, sub_dir, name, force=False):
    _root = root
    dir_path = os.path.join(_root, sub_dir, name)
    if os.path.exists(dir_path) and not force:
        return dir_path
    print('download_path:', dir_path)
    zip_file_path = os.path.join(_root, sub_dir, name + '.zip')
    model_url = "%s/%s.zip"%(BASE_REPO_URL, name)
    download_file(model_url,
             path=zip_file_path,
             overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # zip file has contains ${name}
    real_dir_path = os.path.join(_root, sub_dir)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(real_dir_path)
    #os.remove(zip_file_path)
    return dir_path

def load_insight_face_model(provider, width, height, insight_face_type = "antelopev2"):
    insight_face_model_path =  get_model_folder("insightface")
    download_insightface_model(insight_face_model_path, "models", insight_face_type)
    exec_provider = provider + 'ExecutionProvider'
    insight_face_model = FaceAnalysis(name=insight_face_type, root=insight_face_model_path, providers=[exec_provider]) # alternative to buffalo_l
    insight_face_model.prepare(ctx_id=0, det_size=(width, height))
    return insight_face_model

def load_eva_clip_model():
    clip_file_path = folder_paths.get_full_path("text_encoders", 'EVA02_CLIP_L_336_psz14_s6B.pt')
    if clip_file_path is None:
        clip_dir = os.path.join(folder_paths.models_dir, "clip")
    else:
        clip_dir = os.path.dirname(clip_file_path)
    model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, local_dir=clip_dir)

    model = model.visual

    eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        model["image_mean"] = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        model["image_std"] = (eva_transform_std,) * 3
    return model

def get_latest_flux_pulid_model_file():
    model_path = load_model("pulid", "guozinan/PuLID")
    model_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") and "flux" in f]
    latest = max(model_files, key=extract_version)
    file_path = os.path.join(model_path, latest)
    return file_path

def get_latest_sdxl_pulid_model_file():
    model_path = load_model("pulid", "huchenlei/ipadapter_pulid")
    model_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") and "sdxl" in f]
    latest = max(model_files, key=extract_version)
    file_path = os.path.join(model_path, latest)
    return file_path

def get_latest_pulid_model_path(type):
    match type:
        case "flux":
            file_path = get_latest_flux_pulid_model_file()
        case "sdxl":
            file_path = get_latest_sdxl_pulid_model_file()
        case _:
            file_path = get_latest_flux_pulid_model_file()
    return file_path

def load_pulid_model(face_image, pulid_weight, model, provider, insightface_input_width, insightface_input_height, type):

    file_path = get_latest_pulid_model_path(type)
    insight_face_model = load_insight_face_model(provider, insightface_input_width, insightface_input_height)
    eva_clip_model = load_eva_clip_model()
    
    match type:
        case "flux":
            model_patcher = load_pulid_flux_model_patcher(file_path)
            pulid_model = apply_pulid_flux(model, model_patcher, eva_clip_model, insight_face_model, face_image, pulid_weight, 0.0, 1.0)
        case "sdxl":
            model_patcher = load_pulid_model_patcher(file_path)
            pulid_model = apply_pulid(model, model_patcher, eva_clip_model, insight_face_model, face_image, pulid_weight, 0.0, 1.0, "style")
        case _:
            model_patcher = load_pulid_flux_model_patcher(file_path)
            pulid_model = apply_pulid_flux(model, model_patcher, eva_clip_model, insight_face_model, face_image, pulid_weight, 0.0, 1.0)

    del eva_clip_model
    del insight_face_model
    del model_patcher
    
    return pulid_model

def load_union_control_net(type):
    match type:
        case "flux":
            control_net_path = os.path.join(get_model_folder("control_nets"), "flux_union_control_net.safetensors")
        case "sdxl":
            control_net_path = os.path.join(get_model_folder("control_nets"), "control_net_union_sdxl_1_0.safetensors")
        case _:
            control_net_path = os.path.join(get_model_folder("control_nets"), "flux_union_control_net.safetensors")
    
    controlnet = comfy.controlnet.load_controlnet(control_net_path)
    return controlnet

def set_union_control_net_type(control_net, type):
    control_net = control_net.copy()
    type_number = UNION_CONTROLNET_TYPES.get(type, -1)
    print(f"CONTROL NET TYPE : {type_number}")
    if type_number >= 0:
        control_net.set_extra_arg("control_type", [type_number])
    else:
        control_net.set_extra_arg("control_type", [])

    return control_net

def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent, vae=None, extra_concat=[]):
    if strength == 0:
        return (positive, negative)

    control_hint = image.movedim(-1,1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return out[0], out[1]

def apply_flux_conditions(positive, negative, image, strength, start_percent, end_percent, vae, type, base_model_type):
    controlnet = load_union_control_net(base_model_type)
    typed_controlnet = set_union_control_net_type(controlnet, type)
    new_pos, new_neg = apply_controlnet(positive, negative, typed_controlnet, image, strength, start_percent, end_percent, vae)
    return new_pos, new_neg

def apply_sdxl_conditions(positive, negative, image, strength, start_percent, end_percent, vae, base_model_type):
    controlnet = load_union_control_net(base_model_type)
    new_pos, new_neg = apply_controlnet(positive, negative, controlnet, image, strength, start_percent, end_percent, vae)
    return new_pos, new_neg

def apply_controlnet_conditions(positive, negative, image, strength, start_percent, end_percent, vae, type, base_model_type):
    match base_model_type:
        case "flux":
            new_pos, new_neg = apply_flux_conditions(positive, negative, image, strength, start_percent, end_percent, vae, type, base_model_type)
        case "sdxl":
            new_pos, new_neg = apply_sdxl_conditions(positive, negative, image, strength, start_percent, end_percent, vae, base_model_type)
        case _:
           new_pos, new_neg = apply_flux_conditions(positive, negative, image, strength, start_percent, end_percent, vae, type, base_model_type)

    return new_pos, new_neg

def get_bbox_detector(model_name):
    model_dir = load_model("ultralytics", "bbox")
    model_path = os.path.join(model_dir, model_name)
    model = load_yolo(model_path)

    return UltraBBoxDetector(model)

def load_ic_light_model(model, model_name="iclight_sd15_fc.safetensors"):
    model_path = load_model("ic_light", "lllyasviel/ic-light")

    type_str = str(type(model.model.model_config).__name__)
    if "SD15" not in type_str:
        raise Exception(f"Attempted to load {type_str} model, IC-Light is only compatible with SD 1.5 models.")

    print("LoadAndApplyICLightUnet: Checking IC-Light Unet path")
    model_full_path = os.path.join(model_path, model_name)
    if not os.path.exists(model_full_path):
        raise Exception("Invalid model path")
    else:
        print("LoadAndApplyICLightUnet: Loading IC-Light Unet weights")
        model_clone = model.clone()

        iclight_state_dict = load_torch_file(model_full_path)
        
        print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
        try:          
            if 'conv_in.weight' in iclight_state_dict:
                iclight_state_dict = convert_iclight_unet(iclight_state_dict)
                in_channels = iclight_state_dict["diffusion_model.input_blocks.0.0.weight"].shape[1]
                for key in iclight_state_dict:
                    model_clone.add_patches({key: (iclight_state_dict[key],)}, 1.0, 1.0)
            else:
                for key in iclight_state_dict:
                    model_clone.add_patches({"diffusion_model." + key: (iclight_state_dict[key],)}, 1.0, 1.0)

                in_channels = iclight_state_dict["input_blocks.0.0.weight"].shape[1]

        except:
            raise Exception("Could not patch model")
        print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")

        #Patch ComfyUI's LoRA weight application to accept multi-channel inputs. Thanks @huchenlei
        try:
            if hasattr(lora, 'calculate_weight'):
                lora.calculate_weight = calculate_weight_adjust_channel(lora.calculate_weight)
            else:
                raise Exception("IC-Light: The 'calculate_weight' function does not exist in 'lora'")
        except Exception as e:
            raise Exception(f"IC-Light: Could not patch calculate_weight - {str(e)}")
        
        # Mimic the existing IP2P class to enable extra_conds
        def bound_extra_conds(self, **kwargs):
                return ICLight.extra_conds(self, **kwargs)
        new_extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
        model_clone.add_object_patch("extra_conds", new_extra_conds)
        

        model_clone.model.model_config.unet_config["in_channels"] = in_channels        

        return model_clone
    

def apply_ic_light_model(positive, negative, vae, foreground, multiplier, opt_background=None):
    samples_1 = foreground["samples"]

    if opt_background is not None:
        samples_2 = opt_background["samples"]

        repeats_1 = samples_2.size(0) // samples_1.size(0)
        repeats_2 = samples_1.size(0) // samples_2.size(0)
        if samples_1.shape[1:] != samples_2.shape[1:]:
            samples_2 = comfy.utils.common_upscale(samples_2, samples_1.shape[-1], samples_1.shape[-2], "bilinear", "disabled")

        # Repeat the tensors to match the larger batch size
        if repeats_1 > 1:
            samples_1 = samples_1.repeat(repeats_1, 1, 1, 1)
        if repeats_2 > 1:
            samples_2 = samples_2.repeat(repeats_2, 1, 1, 1)

        concat_latent = torch.cat((samples_1, samples_2), dim=1)
    else:
        concat_latent = samples_1

    out_latent = torch.zeros_like(samples_1)

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()
            d["concat_latent_image"] = concat_latent * multiplier
            n = [t[0], d]
            c.append(n)
        out.append(c)

    ic_light_positive = out[0]
    ic_light_negative = out[1]
    ic_light_empty_latent = {"samples": out_latent}
    return ic_light_positive, ic_light_negative, ic_light_empty_latent

def load_sam_model(model_name="sam_vit_h_4b8939.pth", attempts=10):
    for i in range(attempts):
        try:
            sams_model_path = load_model("sams", "scenario-labs/sam_vit")
            sam_model_full_path = os.path.join(sams_model_path, model_name)
            model_file_name = os.path.basename(sam_model_full_path)
            model_type = model_file_name.split('.')[0]
            if 'hq' not in model_type and 'mobile' not in model_type:
                model_type = '_'.join(model_type.split('_')[:-1])
            sam = sam_model_registry[model_type](checkpoint=sam_model_full_path)
            sam_device = comfy.model_management.get_torch_device()
            sam.to(device=sam_device)
            sam.eval()
            sam.model_name = model_file_name
            break
        except torch.OutOfMemoryError:  # Catch only PyTorch OOM
            unload_models()
            if i == attempts - 1:
                raise Exception("All attempts to sample have failed, try increasing the value of 'sampling_attempt_number'.")
    return sam

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def load_grounding_dino_model(model_name="groundingdino_swint_ogc.pth", model_config="GroundingDINO_SwinT_OGC.cfg.py", attempts=10):
    for i in range(attempts):
        try:
            grounding_dino_model_path = load_model("grounding_dino", "ShilongLiu/GroundingDINO")
            grounding_dino_model_full_path = os.path.join(grounding_dino_model_path, model_name)
            grounding_dino_config_full_path = os.path.join(grounding_dino_model_path, model_config)

            dino_model_args = local_groundingdino_SLConfig.fromfile(grounding_dino_config_full_path)

            if dino_model_args.text_encoder_type == 'bert-base-uncased':
                dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
            
            dino = build_model(dino_model_args)
            checkpoint = torch.load(grounding_dino_model_full_path)
            dino.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

            device = comfy.model_management.get_torch_device()
            dino.to(device=device)
            dino.eval()
            break
        except torch.OutOfMemoryError:  # Catch only PyTorch OOM
            unload_models()
            if i == attempts - 1:
                raise Exception("All attempts to sample have failed, try increasing the value of 'sampling_attempt_number'.")

    return dino

def get_sdxl_clip_vision_file():
    clip_vit_bigG_14_laion2B_39B_b160k = load_model("clip_vit_b", "axssel/IPAdapter_ClipVision_models", "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors")
    return clip_vit_bigG_14_laion2B_39B_b160k

def get_flux_clip_vision_file():
    clip_vision_l_model = load_model("clip_vision_l", "XLabs-AI/flux-ip-adapter", "clip_vision_l.safetensors", "d3cb0c5bb46ff37bf3deb241f02987dfcf9a7963")
    return clip_vision_l_model

def get_clipvision_file(model_type):
    match model_type:
        case "flux":
            clipvision_file = get_flux_clip_vision_file()
        case "sdxl":
            clipvision_file = get_sdxl_clip_vision_file()
        case _:
            clipvision_file = get_flux_clip_vision_file()

    return clipvision_file

def get_sdxl_ipadapter_file():
    ipadapter_sdxl = load_model("ipadapter_sdxl", "axssel/IPAdapter_ClipVision_models", "ip-adapter_sdxl.safetensors")
    return ipadapter_sdxl

def get_flux_ipadapter_file():
    ipadapter_flux = load_model("ipadapter_flux", "XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors")
    return ipadapter_flux

def get_ipadapter_file(model_type):
    match model_type:
        case "flux":
            ipadapter_file = get_flux_ipadapter_file()
        case "sdxl":
            ipadapter_file = get_sdxl_ipadapter_file()
        case _:
            ipadapter_file = get_flux_ipadapter_file()

    return ipadapter_file

def ipadapter_model_loader_sdxl(file, clip_file):
    model = comfy.utils.load_torch_file(file, safe_load=True)
    clip_vision_model = load_clip_vision(clip_file)
    if file.lower().endswith(".safetensors"):
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in model.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            elif key.startswith("adapter_modules."):
                st_model["ip_adapter"][key.replace("adapter_modules.", "")] = model[key]
        model = st_model
        del st_model
    elif "adapter_modules" in model.keys():
        model["ip_adapter"] = model.pop("adapter_modules")

    if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
        raise Exception("invalid IPAdapter model {}".format(file))

    if 'plusv2' in file.lower():
        model["faceidplusv2"] = True
    
    if 'unnorm' in file.lower():
        model["portraitunnorm"] = True

    return model, clip_vision_model, None

def load_flux_ipadapter_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def ipadapter_model_loader_flux(file, clip_file):
    ret_ipa = {}
    ckpt = load_flux_ipadapter_safetensors(file)

    try: 
        clip = FluxClipViT(clip_file)
    except:
        clip = load_clip_vision(clip_file).model
    
    ret_ipa["clip_vision"] = clip
    prefix = "double_blocks."
    blocks = {}
    proj = {}
    for key, value in ckpt.items():
        if key.startswith(prefix):
            blocks[key[len(prefix):].replace('.processor.', '.')] = value
        if key.startswith("ip_adapter_proj_model"):
            proj[key[len("ip_adapter_proj_model."):]] = value
    img_vec_in_dim=768
    context_in_dim=4096
    num_ip_tokens=16        
    if ckpt['ip_adapter_proj_model.proj.weight'].shape[0]//4096==4:
        num_ip_tokens=4
    else:
        num_ip_tokens=16
    improj = ImageProjModel(context_in_dim, img_vec_in_dim, num_ip_tokens)
    improj.load_state_dict(proj)
    ret_ipa["ip_adapter_proj_model"] = improj

    ret_ipa["double_blocks"] = torch.nn.ModuleList([IPProcessor(4096, 3072) for i in range(19)])
    ret_ipa["double_blocks"].load_state_dict(blocks)
    return ret_ipa, clip, None

def ipadapter_model_loader(file, clip_file, model_type):
    match model_type:
        case "flux":
            ipadapter_model, clip_model, model_with_lora = ipadapter_model_loader_flux(file, clip_file)
        case "sdxl":
            ipadapter_model, clip_model, model_with_lora = ipadapter_model_loader_sdxl(file, clip_file)
        case _:
            ipadapter_model, clip_model, model_with_lora = ipadapter_model_loader_flux(file, clip_file)

    return ipadapter_model, clip_model, model_with_lora


def load_ip_adapter(model, lora_strength=0.0,  model_type="flux"):
    from .model_management import get_clipvision_file, get_ipadapter_file, ipadapter_model_loader

    # 1. Load the files
    clipvision_file = get_clipvision_file(model_type)
    ipadapter_file = get_ipadapter_file(model_type)

    # 2. Load the models
    ip_adapter_model, clip_vision_model, lora_file = ipadapter_model_loader(ipadapter_file, clipvision_file, model_type)

    # 3. Load the lora model if needed
    if lora_file is not None:

        lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)

        if lora_strength > 0:
            model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

    return model, ip_adapter_model, clip_vision_model


def apply_ip_adapter_flux(model, ip_adapter_flux, image, ip_strength):

    device=comfy.model_management.get_torch_device()
    bi = model.clone()
    tyanochky = bi.model

    clip = ip_adapter_flux['clip_vision']
    
    if isinstance(clip, FluxClipViT):
        clip_device = next(clip.model.parameters()).device
        image = torch.clip(image*255, 0.0, 255)
        out = clip(image).to(dtype=torch.bfloat16)
        neg_out = clip(torch.zeros_like(image)).to(dtype=torch.bfloat16)
    else:
        print("Using old vit clip")
        clip_device = next(clip.parameters()).device
        pixel_values = clip_preprocess(image.to(clip_device)).float()
        out = clip(pixel_values=pixel_values)
        neg_out = clip(pixel_values=torch.zeros_like(pixel_values))    
        neg_out = neg_out[2].to(dtype=torch.bfloat16)
        out = out[2].to(dtype=torch.bfloat16)

    #TYANOCHKYBY=16
    ip_projes_dev = next(ip_adapter_flux['ip_adapter_proj_model'].parameters()).device
    ip_adapter_flux['ip_adapter_proj_model'].to(dtype=torch.bfloat16)
    ip_projes = ip_adapter_flux['ip_adapter_proj_model'](out.to(ip_projes_dev, dtype=torch.bfloat16)).to(device, dtype=torch.bfloat16)
    ip_neg_pr = ip_adapter_flux['ip_adapter_proj_model'](neg_out.to(ip_projes_dev, dtype=torch.bfloat16)).to(device, dtype=torch.bfloat16)


    ipad_blocks = []
    for block in ip_adapter_flux['double_blocks']:
        ipad = IPProcessor(block.context_dim, block.hidden_dim, ip_projes, ip_strength)
        ipad.load_state_dict(block.state_dict())
        ipad.in_hidden_states_neg = ip_neg_pr
        ipad.in_hidden_states_pos = ip_projes
        ipad.to(dtype=torch.bfloat16)
        npp = DoubleStreamMixerProcessor()
        npp.add_ipadapter(ipad)
        ipad_blocks.append(npp)

    i=0
    for name, _ in attn_processors(tyanochky.diffusion_model).items():
        attribute = f"diffusion_model.{name}"
        #old = copy.copy(get_attr(bi.model, attribute))
        if attribute in model.object_patches.keys():
            old = copy.copy((model.object_patches[attribute]))
        else:
            old = None
        processor = merge_loras(old, ipad_blocks[i])
        processor.to(device, dtype=torch.bfloat16)
        bi.add_object_patch(attribute, processor)
        i+=1

    return bi

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(ipadapter_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(ipadapter_attention, **patch_kwargs)

def ipadapter_execute(model,
                      ipadapter,
                      clipvision,
                      insightface=None,
                      image=None,
                      image_composition=None,
                      image_negative=None,
                      weight=1.0,
                      weight_composition=1.0,
                      weight_faceidv2=None,
                      weight_kolors=1.0,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      attn_mask=None,
                      pos_embed=None,
                      neg_embed=None,
                      unfold_batch=False,
                      embeds_scaling='V only',
                      layer_weights=None,
                      encode_batch_size=0,
                      style_boost=None,
                      composition_boost=None,
                      enhance_tiles=1,
                      enhance_ratio=1.0,):
    
    from insightface.utils import face_align
    from .image import tensor_to_image, tensor_to_size
    from .sdxl_ipadapter_utils import encode_image_masked, IPAdapter
        
    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

    is_full = "proj.3.weight" in ipadapter["image_proj"]
    is_portrait_unnorm = "portraitunnorm" in ipadapter
    is_plus = (is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]) and not is_portrait_unnorm
    output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048
    is_kwai_kolors_faceid = "perceiver_resampler.layers.0.0.to_out.weight" in ipadapter["image_proj"] and ipadapter["image_proj"]["perceiver_resampler.layers.0.0.to_out.weight"].shape[0] == 4096
    is_faceidv2 = "faceidplusv2" in ipadapter or is_kwai_kolors_faceid
    is_kwai_kolors = (is_sdxl and "layers.0.0.to_out.weight" in ipadapter["image_proj"] and ipadapter["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048) or is_kwai_kolors_faceid
    is_portrait = "proj.2.weight" in ipadapter["image_proj"] and not "proj.3.weight" in ipadapter["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] and not is_kwai_kolors_faceid
    is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] or is_portrait_unnorm or is_kwai_kolors_faceid

    if is_faceid and not insightface:
        raise Exception("insightface model is required for FaceID models")

    if is_faceidv2:
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

    if is_kwai_kolors_faceid:
        cross_attention_dim = 4096
    elif is_kwai_kolors:
        cross_attention_dim = 2048
    elif (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm:
        cross_attention_dim = 1280
    else:
        cross_attention_dim = output_cross_attention_dim
    
    if is_kwai_kolors_faceid:
        clip_extra_context_tokens = 6
    elif (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm:
        clip_extra_context_tokens = 16
    else:
        clip_extra_context_tokens = 4

    if image is not None and image.shape[1] != image.shape[2]:
        print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

    if isinstance(weight, list):
        weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1).to(device, dtype=dtype) if unfold_batch else weight[0]

    if style_boost is not None:
        weight_type = "style transfer precise"
    elif composition_boost is not None:
        weight_type = "composition precise"

    # special weight types
    if layer_weights is not None and layer_weights != '':
        weight = { int(k): float(v)*weight for k, v in [x.split(":") for x in layer_weights.split(",")] }
        weight_type = weight_type if weight_type == "style transfer precise" or weight_type == "composition precise" else "linear"
    elif weight_type == "style transfer":
        weight = { 6:weight } if is_sdxl else { 0:weight, 1:weight, 2:weight, 3:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition":
        weight = { 3:weight } if is_sdxl else { 4:weight*0.25, 5:weight }
    elif weight_type == "strong style transfer":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style and composition":
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "strong style and composition":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight_composition, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition, 5:weight_composition, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style transfer precise":
        weight_composition = style_boost if style_boost is not None else weight
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition precise":
        weight_composition = weight
        weight = composition_boost if composition_boost is not None else weight
        if is_sdxl:
            weight = { 0:weight*.1, 1:weight*.1, 2:weight*.1, 3:weight_composition, 4:weight*.1, 5:weight*.1, 6:weight, 7:weight*.1, 8:weight*.1, 9:weight*.1, 10:weight*.1 }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 6:weight*.1, 7:weight*.1, 8:weight*.1, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }

    clipvision_size = 224 if not is_kwai_kolors else 336

    img_comp_cond_embeds = None
    face_cond_embeds = None
    if is_faceid:
        if insightface is None:
            raise Exception("Insightface model is required for FaceID models")

        insightface.det_model.input_size = (640,640) # reset the detection size
        image_iface = tensor_to_image(image)
        face_cond_embeds = []
        image = []

        for i in range(image_iface.shape[0]):
            for size in [(size, size) for size in range(640, 256, -64)]:
                insightface.det_model.input_size = size # TODO: hacky but seems to be working
                face = insightface.get(image_iface[i])
                if face:
                    if not is_portrait_unnorm:
                        face_cond_embeds.append(torch.from_numpy(face[0].normed_embedding).unsqueeze(0))
                    else:
                        face_cond_embeds.append(torch.from_numpy(face[0].embedding).unsqueeze(0))
                    image.append(image_to_tensor(face_align.norm_crop(image_iface[i], landmark=face[0].kps, image_size=336 if is_kwai_kolors_faceid else 256 if is_sdxl else 224)))

                    if 640 not in size:
                        print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                    break
            else:
                raise Exception('InsightFace: No face detected.')
        face_cond_embeds = torch.stack(face_cond_embeds).to(device, dtype=dtype)
        image = torch.stack(image)
        del image_iface, face

    if image is not None:
        img_cond_embeds = encode_image_masked(clipvision, image, batch_size=encode_batch_size, tiles=enhance_tiles, ratio=enhance_ratio, clipvision_size=clipvision_size)
        if image_composition is not None:
            img_comp_cond_embeds = encode_image_masked(clipvision, image_composition, batch_size=encode_batch_size, tiles=enhance_tiles, ratio=enhance_ratio, clipvision_size=clipvision_size)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            image_negative = image_negative if image_negative is not None else torch.zeros([1, clipvision_size, clipvision_size, 3])
            img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size, clipvision_size=clipvision_size).penultimate_hidden_states
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds if not is_faceid else face_cond_embeds
            if image_negative is not None and not is_faceid:
                img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size, clipvision_size=clipvision_size).image_embeds
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.image_embeds
        del image_negative, image_composition

        image = None if not is_faceid else image # if it's face_id we need the cropped face for later
    elif pos_embed is not None:
        img_cond_embeds = pos_embed

        if neg_embed is not None:
            img_uncond_embeds = neg_embed
        else:
            if is_plus:
                img_uncond_embeds = encode_image_masked(clipvision, torch.zeros([1, clipvision_size, clipvision_size, 3]), clipvision_size=clipvision_size).penultimate_hidden_states
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        del pos_embed, neg_embed
    else:
        raise Exception("Images or Embeds are required")

    # ensure that cond and uncond have the same batch size
    img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

    img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
    img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
    if img_comp_cond_embeds is not None:
        img_comp_cond_embeds = img_comp_cond_embeds.to(device, dtype=dtype)

    # combine the embeddings if needed
    if combine_embeds != "concat" and img_cond_embeds.shape[0] > 1 and not unfold_batch:
        if combine_embeds == "add":
            img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.sum(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.sum(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "subtract":
            img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
            img_cond_embeds = img_cond_embeds.unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = face_cond_embeds[0] - torch.mean(face_cond_embeds[1:], dim=0)
                face_cond_embeds = face_cond_embeds.unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = img_comp_cond_embeds[0] - torch.mean(img_comp_cond_embeds[1:], dim=0)
                img_comp_cond_embeds = img_comp_cond_embeds.unsqueeze(0)
        elif combine_embeds == "average":
            img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "norm average":
            img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds / torch.norm(face_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds / torch.norm(img_comp_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

    if attn_mask is not None:
        attn_mask = attn_mask.to(device, dtype=dtype)

    encoder_hid_proj = None

    if is_kwai_kolors_faceid and hasattr(model.model, "diffusion_model") and hasattr(model.model.diffusion_model, "encoder_hid_proj"):
        encoder_hid_proj = model.model.diffusion_model.encoder_hid_proj.state_dict()

    ipa = IPAdapter(
        ipadapter,
        cross_attention_dim=cross_attention_dim,
        output_cross_attention_dim=output_cross_attention_dim,
        clip_embeddings_dim=img_cond_embeds.shape[-1],
        clip_extra_context_tokens=clip_extra_context_tokens,
        is_sdxl=is_sdxl,
        is_plus=is_plus,
        is_full=is_full,
        is_faceid=is_faceid,
        is_portrait_unnorm=is_portrait_unnorm,
        is_kwai_kolors=is_kwai_kolors,
        encoder_hid_proj=encoder_hid_proj,
        weight_kolors=weight_kolors
    ).to(device, dtype=dtype)

    if is_faceid and is_plus:
        cond = ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
        # TODO: check if noise helps with the uncond face embeds
        uncond = ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
    else:
        cond, uncond = ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds, encode_batch_size)
        if img_comp_cond_embeds is not None:
            cond_comp = ipa.get_image_embeds(img_comp_cond_embeds, img_uncond_embeds, encode_batch_size)[0]

    cond = cond.to(device, dtype=dtype)
    uncond = uncond.to(device, dtype=dtype)

    cond_alt = None
    if img_comp_cond_embeds is not None:
        cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

    del img_cond_embeds, img_uncond_embeds, img_comp_cond_embeds, face_cond_embeds

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": ipa,
        "weight": weight,
        "cond": cond,
        "cond_alt": cond_alt,
        "uncond": uncond,
        "weight_type": weight_type,
        "mask": attn_mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "unfold_batch": unfold_batch,
        "embeds_scaling": embeds_scaling,
    }

    number = 0
    if not is_sdxl:
        for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            number += 1
        for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            number += 1
        patch_kwargs["module_key"] = str(number*2+1)
        set_model_patch_replace(model, patch_kwargs, ("middle", 1))
    else:
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 1, index))
            number += 1

    return model

def apply_ip_adapter_sdxl(model, ipadapter_model, clip_vision_model, image, weight, start_at, end_at, weight_type, attn_mask=None):
        if weight_type.startswith("style"):
            weight_type = "style transfer"
        elif weight_type == "prompt is more important":
            weight_type = "ease out"
        else:
            weight_type = "linear"

        ipa_args = {
            "image": image,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "weight_type": weight_type,
            "insightface": None,
        }

        applied_model = ipadapter_execute(model.clone(), ipadapter_model, clip_vision_model, **ipa_args)
        return applied_model

def apply_ip_adapter(model, ipadapter_model, clip_vision_model, image, weight, start_at, end_at, weight_type, attn_mask=None, model_type="flux"):
    match model_type:
        case "flux":
            model = apply_ip_adapter_flux(model, ipadapter_model, image, weight)
        case "sdxl":
            model = apply_ip_adapter_sdxl(model, ipadapter_model, clip_vision_model, image, weight, start_at, end_at, weight_type, attn_mask)
        case _:
            model = apply_ip_adapter_flux(model, ipadapter_model, image, weight)

    return model

def unload_models():
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(True)
    torch.cuda.empty_cache()