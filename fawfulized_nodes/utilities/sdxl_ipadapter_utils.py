import re
import torch
import os
import folder_paths
from comfy.clip_vision import clip_preprocess, Output
import comfy.model_management as model_management
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T
import copy
import torch.nn as nn
from .image_proj_models import MLPProjModel, MLPProjModelFaceId, ProjModelFaceIdPlus, Resampler, ImageProjModel

def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    
        out.append(x_split)
    
    x_split = torch.stack(out, dim=0)
    
    return x_split

def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])
        
        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)
    
    out = torch.cat(out, dim=0)

    return out

def merge_embeddings(x, tiles): # TODO: this needs so much testing that I don't even
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        grid_size = int(num_tiles ** 0.5)
        tile_size = int(embeds.shape[1] ** 0.5)
        reshaped = embeds.reshape(grid_size, grid_size, tile_size, tile_size)
        
        # Merge the tiles
        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged, (tile_size, tile_size))  # pool to [1, tile_size, tile_size]
        pooled = pooled.flatten(1)  # flatten to [1, tile_size^2]
        out.append(pooled)
    out = torch.cat(out, dim=0)
    
    return out

def encode_image_masked(clip_vision, image, mask=None, batch_size=0, tiles=1, ratio=1.0, clipvision_size=224):
    # full image embeds
    embeds = encode_image_masked_(clip_vision, image, mask, batch_size, clipvision_size=clipvision_size)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = Output()
        for i in image_split:
            encoded = encode_image_masked_(clip_vision, i, mask, batch_size, clipvision_size=clipvision_size)
            if not hasattr(embeds_split, "image_embeds"):
                #embeds_split["last_hidden_state"] = encoded["last_hidden_state"]
                embeds_split["image_embeds"] = encoded["image_embeds"]
                embeds_split["penultimate_hidden_states"] = encoded["penultimate_hidden_states"]
            else:
                #embeds_split["last_hidden_state"] = torch.cat((embeds_split["last_hidden_state"], encoded["last_hidden_state"]), dim=0)
                embeds_split["image_embeds"] = torch.cat((embeds_split["image_embeds"], encoded["image_embeds"]), dim=0)
                embeds_split["penultimate_hidden_states"] = torch.cat((embeds_split["penultimate_hidden_states"], encoded["penultimate_hidden_states"]), dim=0)

        #embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'])
        embeds_split["image_embeds"] = merge_embeddings(embeds_split["image_embeds"], tiles)
        embeds_split["penultimate_hidden_states"] = merge_hiddenstates(embeds_split["penultimate_hidden_states"], tiles)

        #embeds['last_hidden_state'] = torch.cat([embeds_split['last_hidden_state'], embeds['last_hidden_state']])
        if embeds['image_embeds'].shape[0] > 1: # if we have more than one image we need to average the embeddings for consistency
            embeds['image_embeds'] = embeds['image_embeds']*ratio + embeds_split['image_embeds']*(1-ratio)
            embeds['penultimate_hidden_states'] = embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']*(1-ratio)
            #embeds['image_embeds'] = (embeds['image_embeds']*ratio + embeds_split['image_embeds']) / 2
            #embeds['penultimate_hidden_states'] = (embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']) / 2
        else: # otherwise we can concatenate them, they can be averaged later
            embeds['image_embeds'] = torch.cat([embeds['image_embeds']*ratio, embeds_split['image_embeds']])
            embeds['penultimate_hidden_states'] = torch.cat([embeds['penultimate_hidden_states']*ratio, embeds_split['penultimate_hidden_states']])

    #del embeds_split

    return embeds

def encode_image_masked_(clip_vision, image, mask=None, batch_size=0, clipvision_size=224):
    model_management.load_model_gpu(clip_vision.patcher)
    outputs = Output()

    if batch_size == 0:
        batch_size = image.shape[0]
    elif batch_size > image.shape[0]:
        batch_size = image.shape[0]

    image_batch = torch.split(image, batch_size, dim=0)

    for img in image_batch:
        img = img.to(clip_vision.load_device)
        pixel_values = clip_preprocess(img, size=clipvision_size).float()

        # TODO: support for multiple masks
        if mask is not None:
            pixel_values = pixel_values * mask.to(clip_vision.load_device)

        out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

        if not hasattr(outputs, "last_hidden_state"):
            outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
            outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
        else:
            outputs["last_hidden_state"] = torch.cat((outputs["last_hidden_state"], out[0].to(model_management.intermediate_device())), dim=0)
            outputs["image_embeds"] = torch.cat((outputs["image_embeds"], out[2].to(model_management.intermediate_device())), dim=0)
            outputs["penultimate_hidden_states"] = torch.cat((outputs["penultimate_hidden_states"], out[1].to(model_management.intermediate_device())), dim=0)

    del img, pixel_values, out
    torch.cuda.empty_cache()

    return outputs

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

# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
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


class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False, is_plus=False, is_full=False, is_faceid=False, is_portrait_unnorm=False, is_kwai_kolors=False, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus
        self.is_portrait_unnorm = is_portrait_unnorm
        self.is_kwai_kolors = is_kwai_kolors

        if is_faceid and not is_portrait_unnorm:
            self.image_proj_model = self.init_proj_faceid()
        elif is_full:
            self.image_proj_model = self.init_proj_full()
        elif is_plus or is_portrait_unnorm:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(ipadapter_model["ip_adapter"], encoder_hid_proj=encoder_hid_proj, weight_kolors=weight_kolors)

        self.multigpu_clones = {}

    def create_multigpu_clone(self, device):
        if device not in self.multigpu_clones:
            orig_multigpu_clones = self.multigpu_clones
            try:
                self.multigpu_clones = {}
                new_clone = copy.deepcopy(self)
                new_clone = new_clone.to(device)
                orig_multigpu_clones[device] = new_clone
            finally:
                self.multigpu_clones = orig_multigpu_clones

    def get_multigpu_clone(self, device):
        return self.multigpu_clones.get(device, self)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20 if self.is_sdxl and not self.is_kwai_kolors else 12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    def init_proj_full(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim
        )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=self.clip_extra_context_tokens,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed, batch_size):
        torch_device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        clip_embed = torch.split(clip_embed, batch_size, dim=0)
        clip_embed_zeroed = torch.split(clip_embed_zeroed, batch_size, dim=0)
        
        image_prompt_embeds = []
        uncond_image_prompt_embeds = []

        for ce, cez in zip(clip_embed, clip_embed_zeroed):
            image_prompt_embeds.append(self.image_proj_model(ce.to(torch_device)).to(intermediate_device))
            uncond_image_prompt_embeds.append(self.image_proj_model(cez.to(torch_device)).to(intermediate_device))

        del clip_embed, clip_embed_zeroed

        image_prompt_embeds = torch.cat(image_prompt_embeds, dim=0)
        uncond_image_prompt_embeds = torch.cat(uncond_image_prompt_embeds, dim=0)

        torch.cuda.empty_cache()

        #image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(self, face_embed, clip_embed, s_scale, shortcut, batch_size):
        torch_device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        face_embed_batch = torch.split(face_embed, batch_size, dim=0)
        clip_embed_batch = torch.split(clip_embed, batch_size, dim=0)

        embeds = []
        for face_embed, clip_embed in zip(face_embed_batch, clip_embed_batch):
            embeds.append(self.image_proj_model(face_embed.to(torch_device), clip_embed.to(torch_device), scale=s_scale, shortcut=shortcut).to(intermediate_device))

        embeds = torch.cat(embeds, dim=0)
        del face_embed_batch, clip_embed_batch
        torch.cuda.empty_cache()
        #embeds = self.image_proj_model(face_embed, clip_embed, scale=s_scale, shortcut=shortcut)
        return embeds

class To_KV(nn.Module):
    def __init__(self, state_dict, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        if encoder_hid_proj is not None:
            hid_proj = nn.Linear(encoder_hid_proj["weight"].shape[1], encoder_hid_proj["weight"].shape[0], bias=True)
            hid_proj.weight.data = encoder_hid_proj["weight"] * weight_kolors
            hid_proj.bias.data = encoder_hid_proj["bias"] * weight_kolors

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            if encoder_hid_proj is not None:
                linear_proj = nn.Linear(value.shape[1], value.shape[0], bias=False)
                linear_proj.weight.data = value
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Sequential(hid_proj, linear_proj)
            else:
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
                self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value