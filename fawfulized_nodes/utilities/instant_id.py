import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
import cv2
import PIL.Image
import numpy as np
from .resampler import Resampler

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T


class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

    
class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        #image_prompt_embeds = clip_embed.clone().detach()
        image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds

def _set_model_patch_replace(model, patch_kwargs, key):
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
        to["patches_replace"]["attn2"][key] = Attn2Replace(instantid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(instantid_attention, **patch_kwargs)


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def extractFeatures(insightface, image, extract_kps=False):
    from .image import tensor_to_image
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

                if extract_kps:
                    out.append(draw_kps(face_img[i], face['kps']))
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out

class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]
    
    def add(self, callback, **kwargs):          
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])
        
        return out.to(dtype=dtype)

def instantid_attention(out, q, k, v, extra_options, module_key='', ipadapter=None, weight=1.0, cond=None, cond_alt=None, uncond=None, weight_type="linear", mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False, embeds_scaling='V only', **kwargs):
    from .image import tensor_to_size
    dtype = q.dtype
    cond_or_uncond = extra_options["cond_or_uncond"]
    block_type = extra_options["block"][0]
    #block_id = extra_options["block"][1]
    t_idx = extra_options["transformer_index"]
    layers = 11 if '101_to_k_ip' in ipadapter.ip_layers.to_kvs else 16
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    # extra options for AnimateDiff
    ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

    b = q.shape[0]
    seq_len = q.shape[1]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    if weight_type == 'ease in':
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == 'ease out':
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == 'ease in-out':
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'reverse in-out':
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'weak input' and block_type == 'input':
        weight = weight * 0.2
    elif weight_type == 'weak middle' and block_type == 'middle':
        weight = weight * 0.2
    elif weight_type == 'weak output' and block_type == 'output':
        weight = weight * 0.2
    elif weight_type == 'strong middle' and (block_type == 'input' or block_type == 'output'):
        weight = weight * 0.2
    elif isinstance(weight, dict):
        if t_idx not in weight:
            return 0

        weight = weight[t_idx]

        if cond_alt is not None and t_idx in cond_alt:
            cond = cond_alt[t_idx]
            del cond_alt

    if unfold_batch:
        # Check AnimateDiff context window
        if ad_params is not None and ad_params["sub_idxs"] is not None:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, ad_params["full_length"])
                weight = torch.Tensor(weight[ad_params["sub_idxs"]])
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            # if image length matches or exceeds full_length get sub_idx images
            if cond.shape[0] >= ad_params["full_length"]:
                cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
            # otherwise get sub_idxs images
            else:
                cond = tensor_to_size(cond, ad_params["full_length"])
                uncond = tensor_to_size(uncond, ad_params["full_length"])
                cond = cond[ad_params["sub_idxs"]]
                uncond = uncond[ad_params["sub_idxs"]]
        else:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, batch_prompt)
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            cond = tensor_to_size(cond, batch_prompt)
            uncond = tensor_to_size(uncond, batch_prompt)

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond)
    else:
        # TODO: should we always convert the weights to a tensor?
        if isinstance(weight, torch.Tensor):
            weight = tensor_to_size(weight, batch_prompt)
            if torch.all(weight == 0):
                return 0
            weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
        elif weight == 0:
            return 0

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)

    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)
    
    if embeds_scaling == 'K+mean(V) w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
        ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        del ip_v_mean
    elif embeds_scaling == 'K+V w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    elif embeds_scaling == 'K+V':
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    else:
        #ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        out_ip = out_ip * weight # I'm doing this to get the same results as before

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        # check if using AnimateDiff and sliding context window
        if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
            # if mask length matches or exceeds full_length, get sub_idx masks
            if mask.shape[0] >= ad_params["full_length"]:
                mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            else:
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                mask = tensor_to_size(mask, ad_params["full_length"])
                mask = mask[ad_params["sub_idxs"]]
        else:
            mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    #out = out + out_ip

    return out_ip.to(dtype=dtype)

def apply_instantid(instantid, insightface, control_net, image, model, positive, negative, start_at, end_at, weight=.8, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
    dtype = comfy.model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
    
    device = comfy.model_management.get_torch_device()

    ip_weight = weight if ip_weight is None else ip_weight
    cn_strength = weight if cn_strength is None else cn_strength

    face_embed = extractFeatures(insightface, image)
    if face_embed is None:
        raise Exception('Reference Image: No face detected.')

    # if no keypoints image is provided, use the image itself (only the first one in the batch)
    face_kps = extractFeatures(insightface, image_kps if image_kps is not None else image[0].unsqueeze(0), extract_kps=True)

    if face_kps is None:
        face_kps = torch.zeros_like(image) if image_kps is None else image_kps
        print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

    clip_embed = face_embed
    # InstantID works better with averaged embeds (TODO: needs testing)
    if clip_embed.shape[0] > 1:
        if combine_embeds == 'average':
            clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
        elif combine_embeds == 'norm average':
            clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

    if noise > 0:
        seed = int(torch.sum(clip_embed).item()) % 1000000007
        torch.manual_seed(seed)
        clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        #clip_embed_zeroed = add_noise(clip_embed, noise)
    else:
        clip_embed_zeroed = torch.zeros_like(clip_embed)

    # 1: patch the attention
    instantid.to(device, dtype=dtype)

    image_prompt_embeds, uncond_image_prompt_embeds = instantid.get_image_embeds(clip_embed.to(device, dtype=dtype), clip_embed_zeroed.to(device, dtype=dtype))

    image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)

    work_model = model.clone()

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    if mask is not None:
        mask = mask.to(device)

    patch_kwargs = {
        "ipadapter": instantid,
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "mask": mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }

    number = 0
    for id in [4,5,7,8]: # id of input_blocks that have cross attention
        block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
            number += 1
    for id in range(6): # id of output_blocks that have cross attention
        block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
            number += 1
    for index in range(10):
        patch_kwargs["module_key"] = str(number*2+1)
        _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
        number += 1

    # 2: do the ControlNet
    if mask is not None and len(mask.shape) < 3:
        mask = mask.unsqueeze(0)

    cnets = {}
    cond_uncond = []

    is_cond = True
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

            if mask is not None and is_cond:
                d['mask'] = mask
                d['set_area_to_bounds'] = False

            n = [t[0], d]
            c.append(n)
        cond_uncond.append(c)
        is_cond = False

    return work_model, cond_uncond[0], cond_uncond[1]

def get_instantid_conditionning(insightface, image, control_net, cn_strength, start_at, end_at, positive, negative, image_prompt_embeds, uncond_image_prompt_embeds, mask=None, image_kps=None):
    
    # if no keypoints image is provided, use the image itself (only the first one in the batch)
    face_kps = extractFeatures(insightface, image_kps if image_kps is not None else image[0].unsqueeze(0), extract_kps=True)

    if face_kps is None:
        face_kps = torch.zeros_like(image) if image_kps is None else image_kps
        print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

    # 2: do the ControlNet
    if mask is not None and len(mask.shape) < 3:
        mask = mask.unsqueeze(0)

    cnets = {}
    cond_uncond = []

    is_cond = True
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

            if mask is not None and is_cond:
                d['mask'] = mask
                d['set_area_to_bounds'] = False

            n = [t[0], d]
            c.append(n)
        cond_uncond.append(c)
        is_cond = False
    
    return cond_uncond[0], cond_uncond[1]


def apply_instantid_combine(instantid, insightface, image, model, start_at, end_at, weight=.8, ip_weight=None, noise=0.35, mask=None, combine_embeds='average'):
    dtype = comfy.model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
    
    device = comfy.model_management.get_torch_device()

    ip_weight = weight if ip_weight is None else ip_weight

    face_embed = extractFeatures(insightface, image)
    if face_embed is None:
        raise Exception('Reference Image: No face detected.')

    clip_embed = face_embed
    # InstantID works better with averaged embeds (TODO: needs testing)
    if clip_embed.shape[0] > 1:
        if combine_embeds == 'average':
            clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
        elif combine_embeds == 'norm average':
            clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

    if noise > 0:
        seed = int(torch.sum(clip_embed).item()) % 1000000007
        torch.manual_seed(seed)
        clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        #clip_embed_zeroed = add_noise(clip_embed, noise)
    else:
        clip_embed_zeroed = torch.zeros_like(clip_embed)

    # 1: patch the attention
    instantid.to(device, dtype=dtype)

    image_prompt_embeds, uncond_image_prompt_embeds = instantid.get_image_embeds(clip_embed.to(device, dtype=dtype), clip_embed_zeroed.to(device, dtype=dtype))

    image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)

    work_model = model.clone()

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    if mask is not None:
        mask = mask.to(device)

    patch_kwargs = {
        "ipadapter": instantid,
        "weight": ip_weight,
        "cond": image_prompt_embeds,
        "uncond": uncond_image_prompt_embeds,
        "mask": mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }

    number = 0
    for id in [4,5,7,8]: # id of input_blocks that have cross attention
        block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
            number += 1
    for id in range(6): # id of output_blocks that have cross attention
        block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
        for index in block_indices:
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
            number += 1
    for index in range(10):
        patch_kwargs["module_key"] = str(number*2+1)
        _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
        number += 1

    return work_model, image_prompt_embeds, uncond_image_prompt_embeds