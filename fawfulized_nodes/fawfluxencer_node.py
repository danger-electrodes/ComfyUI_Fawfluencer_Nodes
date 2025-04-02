import folder_paths
import os
import torch
import comfy.samplers
from server import PromptServer
import comfy.model_management
import gc
import time
#from memory_profiler import profile
import subprocess
import sys

# UTILITIES IMPORTS
from .utilities import sampling as sampling_utilities
from .utilities import image as image_utilities
from .utilities import model_management as model_management_utilities
from .utilities.cache import remove_cache
from .utilities.detectors import get_faces_list, segment_anything, process_generated_image, process_model_spreadsheet, get_best_face_match, get_face_helper, load_insightface_cropper, face_cropper, face_shaper, face_shaper_composite
# CONTROL NETS IMPORTS
from .control_nets.control_nets_preprocessors import try_process_image as process_image

# CONSTANTS IMPORTS
from .constants import ASPECT_RATIOS, CONTROL_NET_CUSTOM_DATAS, INFLUENCER_FACE_CUSTOM_DATAS, INFLUENCER_BODY_CUSTOM_DATAS, BACKGROUND_CUSTOM_DATAS, PROMPT_CUSTOM_DATAS, BBOX_MODELS

#we need to upgrade protobuf otherwise insightface take its holy time to load
def upgrade_protobuf():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf"])

upgrade_protobuf()

def get_dimensions(resolution):
    dimensions = resolution.split(' ')[0]
    width, height = map(int, dimensions.split('x'))
    return width, height

def get_all_value(kwargs):
    id = kwargs['unique_id']
    prompt = kwargs['prompt']
    int_id = int(id)
    values = prompt[str(int_id)]
    return values["inputs"]

def get_widget_value(kwargs, widget_name):
    return get_all_value(kwargs)[widget_name]

def set_widget_value(kwargs, widget_name, type, value):
    id = kwargs['unique_id']
    PromptServer.instance.send_sync("fawfulized-feedback", {"node_id": id, "widget_name": widget_name, "type": type, "value": value})

class FawfluxencerNode:

    def __init__(self):
        self.cache_pulid_model = None
        self.cache_sam_model = None
        self.cache_grounding_dino_model = None
        '''self.cache_spreadsheet_landmarks = None
        self.cache_faces_images = None
        self.cache_generated_image_ratio = None
        self.cache_face_helper = None
        self.cache_insightface_cropper = None'''
        self.cache_face_image_file = None
        
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        

        # Define aspect ratios with human-readable format

        return {"required":
                    {
                        "type": (["flux", "sdxl"], {"default":"flux"}),
                        "number_of_pictures":("INT", {"default": 1, "min": 1, "max": 100}),
                        "sampling_attempt_number":("INT", {"default": 10, "min": 1, "max": 100}),
                        "noise": ("NOISE", ),
                        "flux_base_model": ("MODEL",),
                        "flux_and_loras_model": ("MODEL",),
                        "flux_details_model": ("MODEL",),
                        "dual_clip":("CLIP",),
                        "flux_vae":("VAE",),
                        "resolution": (ASPECT_RATIOS, {"default":"896x1152 (7:9)"}),

                        "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default":"gradient_estimation"}),
                        "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default":"ddim_uniform"}),
                        "steps": ("INT", {"default": 20,"min": 5, "max": 100, "step": 1}),
                        "cfg": ("FLOAT", {"default": 6, "precision": 2}),

                        "details_sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default":"deis"}),
                        "details_scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default":"ddim_uniform"}),
                        "details_steps": ("INT", {"default": 15,"min": 5, "max": 40, "step": 1}),
                        "details_cfg": ("FLOAT", {"default": 6, "precision": 2}),

                        "details_denoise": ("FLOAT", {"default": 0.10,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                        "details_amounts": ("FLOAT", {"default": 0.10,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                        "upscale_image_by": ("FLOAT", {"default": 0.10,"min": 1.0, "max": 3.0, "step": 0.01, "precision": 2}),

                        "control_net": (sorted(files), {"custom_datas": CONTROL_NET_CUSTOM_DATAS}),
                        "influencer_face": (sorted(files), {"custom_datas": INFLUENCER_FACE_CUSTOM_DATAS}),
                        "influencer_body": (sorted(files), {"custom_datas": INFLUENCER_BODY_CUSTOM_DATAS}),
                        "background": (sorted(files), {"custom_datas": BACKGROUND_CUSTOM_DATAS}),
                        "image_description": (["text area", "llm generated text"], {"custom_datas": PROMPT_CUSTOM_DATAS}), #only "text area" and "llm generated text" supported yet
                    },
                    "hidden": {
                        "prompt": "PROMPT",
                        "unique_id": "UNIQUE_ID",
                    },
                }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_NAMES = ("images", "masks", "closest_face")
    FUNCTION = "generate"
    CATEGORY = "Fawfulized/Influencer"

    def get_latent(self, kwargs, width, height):
        use_background = get_widget_value(kwargs, "enable_background")
        if(use_background):
            bg_file = kwargs['background']
            bg_img, bg_mask = image_utilities.load_image(bg_file)
            resize_bg_img = image_utilities.resize(bg_img, width, height, "nearest-exact")
            latent = self.encode(kwargs, resize_bg_img)
        else:
            latent = image_utilities.get_empty_latent(width, height, 1)
        
        return latent, use_background


    def get_pulid_model(self, face_img, kwargs):
        type = kwargs['type']
        model = kwargs['flux_base_model']

        bbox_model_name = next((s for s in BBOX_MODELS if "face" in s), None)
        faces = get_faces_list(bbox_model_name, face_img, 0.50, 10, 1.0, 10, "face")
        face_batch = image_utilities.image_list_to_batch(faces)

        pulid_weight = get_widget_value(kwargs, "pulid_strength")
        insight_face_provider = get_widget_value(kwargs, "insight_face_provider")
        insight_face_input_width = int(get_widget_value(kwargs, "insight_face_input_width"))
        insight_face_input_height = int(get_widget_value(kwargs, "insight_face_input_height"))

        return model_management_utilities.load_pulid_model(face_batch, pulid_weight, model, insight_face_provider, insight_face_input_width, insight_face_input_height, type)

    def get_cfg_guider(self, kwargs, positive, negative):
        model = kwargs['flux_and_loras_model']
        cfg = kwargs['cfg']
        guider = sampling_utilities.get_cfg_guider(model, positive, negative, cfg)
        return guider
    
    def get_basic_face_guider(self, kwargs, conditionning, cache_pulid_model):

        guider = sampling_utilities.get_basic_guider(cache_pulid_model, conditionning)
        return guider
    
    def get_details_cfg_guider(self, positive, negative, kwargs):
        model = kwargs['flux_details_model']
        cfg = kwargs['details_cfg']
        guider = sampling_utilities.get_cfg_guider(model, positive, negative, cfg)
        return guider

    def get_sigmas(self, kwargs):
        use_background = get_widget_value(kwargs, "enable_background")
        denoise = get_widget_value(kwargs, "background_denoise") if use_background else 1.0

        model = kwargs['flux_and_loras_model']
        steps = kwargs['steps']
        scheduler = kwargs['scheduler']
        split_at_steps = int(get_widget_value(kwargs, "face_swap_step_start"))
        generation_sigmas = sampling_utilities.get_sigmas(model, scheduler, steps, denoise)
        high, low = sampling_utilities.split_sigmas(generation_sigmas, split_at_steps)
        return high, low
    
    def get_details_sigmas(self, kwargs):
        model = kwargs['flux_details_model']
        details_denoise = kwargs['details_denoise']
        details_steps = kwargs['details_steps']
        scheduler = kwargs['details_scheduler']
        details_sigmas = sampling_utilities.get_sigmas(model, scheduler, details_steps, details_denoise)

        details_amount = kwargs['details_amounts']
        factor = 1.0 - details_amount
        details_sigmas_multiplied = sampling_utilities.multiply_sigmas(details_sigmas, factor, 0.0, 1.0)
        return details_sigmas_multiplied
    
    def get_sampler(self, kwargs):
        sampler_name = kwargs['sampler_name']
        sampler = sampling_utilities.get_sampler(sampler_name)
        return sampler
    
    def get_details_sampler(self, kwargs):
        sampler_name = kwargs['details_sampler_name']
        sampler = sampling_utilities.get_sampler(sampler_name)
        details_amount = kwargs['details_amounts']
        factor = -details_amount
        lying_sampler = sampling_utilities.get_lying_sampler(sampler, factor, 0.1, 0.9)
        return lying_sampler


    def encode_text(self, kwargs, flux_guidance = None):
        clip = kwargs['dual_clip']
        text = get_widget_value(kwargs, "image_description_conditionning")
        neg_text = get_widget_value(kwargs, "negative_image_description_conditionning")
        formatted_text = text # Here in the middle we want to do the necessary to format the text correctly, then set the widget value to the new formatted text and pass the formatted text into encode_text
        set_widget_value(kwargs, "image_description_preview", "TEXT", formatted_text)
        return sampling_utilities.encode_text(clip, formatted_text, neg_text, flux_guidance)
    
    def process_controlnet_image(self, kwargs):
        preprocessor = get_widget_value(kwargs, "control_net_type")
        face_image_file = kwargs['control_net']
        face_img, face_mask = image_utilities.load_image(face_image_file)
        out = process_image(preprocessor, face_img)
        return out
    
    def get_controlnet_conditions(self, positive, negative, kwargs):
        processed_image = self.process_controlnet_image(kwargs)
        strength = get_widget_value(kwargs, "control_net_strength")
        start = get_widget_value(kwargs, "control_net_start_at")
        end = get_widget_value(kwargs, "control_net_end_at")
        vae = kwargs['flux_vae']
        type = get_widget_value(kwargs, "control_net_type")
        base_model_type = kwargs['type']
        new_pos, new_neg = model_management_utilities.apply_controlnet_conditions(positive, negative, processed_image, strength, start, end, vae, type, base_model_type)
        return new_pos, new_neg
     
    def decode(self, kwargs, samples):
        vae = kwargs['flux_vae']
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images
    
    def encode(self, kwargs, pixels):
        vae = kwargs['flux_vae']
        t = vae.encode(pixels[:,:,:,:3])
        return {"samples":t}

    def initialized(self, face_image_file):
        is_initialized = face_image_file == self.cache_face_image_file
        return is_initialized

    def initialize_model_and_landmarks(self, face_img, bbox_model_name, face_image_file, kwargs):
        self.cache_pulid_model = self.get_pulid_model(face_img, kwargs)
        self.cache_sam_model = model_management_utilities.load_sam_model()
        self.cache_grounding_dino_model = model_management_utilities.load_grounding_dino_model()
        '''self.cache_face_helper = get_face_helper()
        model_spreadsheets_landmarks, generated_image_ratio, faces_images = process_model_spreadsheet(bbox_model_name, face_img, 0.50, 10, 1.2, 10, "face", self.cache_face_helper, False)
        self.cache_spreadsheet_landmarks = model_spreadsheets_landmarks
        self.cache_generated_image_ratio = generated_image_ratio
        self.cache_faces_images = faces_images'''
        self.cache_face_image_file = face_image_file

    def load_ipadapter(self, kwargs):
        model = kwargs['flux_details_model']
        base_model_type = kwargs['type']
        model, ip_adapter_model, clip_vision_model = model_management_utilities.load_ip_adapter(model, 0, base_model_type)
        return model, ip_adapter_model, clip_vision_model
    
    def apply_ipadapter(self, details_model, details_ip_adapter_model, details_clip_vision_model, image, strength, start_at, end_at, kwargs):
        base_model_type = kwargs['type']
        applied_ipadapter_model = model_management_utilities.apply_ip_adapter(details_model, details_ip_adapter_model, details_clip_vision_model, image, strength, start_at, end_at, "standard", None, base_model_type)
        return applied_ipadapter_model
    
    #@profile
    def generate(self, **kwargs):

        images = []
        masks = []

        #details_model, details_ip_adapter_model, details_clip_vision_model = self.load_ipadapter(kwargs)
        num_pictures = kwargs['number_of_pictures']
        attempts = kwargs['sampling_attempt_number']
        start_noise = kwargs['noise']
        
        width, height = get_dimensions(kwargs['resolution'])

        face_image_file = kwargs['influencer_face']
        face_img, face_mask = image_utilities.load_image(face_image_file)

        is_initialized = self.initialized(face_image_file)

        bbox_model_name = next((s for s in BBOX_MODELS if "face" in s), None)

        '''if self.cache_insightface_cropper is None:
            self.cache_insightface_cropper = load_insightface_cropper()'''

        if not is_initialized:
            self.initialize_model_and_landmarks(face_img, bbox_model_name, face_image_file, kwargs)

        closest_face = None

        vae = kwargs['flux_vae']

        for i in range(num_pictures):
            
            latent, use_background = self.get_latent(kwargs, width, height)

            positive, negative = self.encode_text(kwargs, 3.5)

            use_control_net = get_widget_value(kwargs, "enable_control_net")

            cnet_positive = positive
            cnet_negative = negative

            if(use_control_net):
                cnet_positive, cnet_negative = self.get_controlnet_conditions(cnet_positive, cnet_negative, kwargs)

            high, low = self.get_sigmas(kwargs)
            sampler = self.get_sampler(kwargs)
            
            seed = start_noise.seed + i
            noise = sampling_utilities.get_noise(seed)

            guider = self.get_cfg_guider(kwargs, cnet_positive, cnet_negative)

            mask = None
            
            if(use_background):
                '''vae = kwargs['flux_vae']
                mask_type = get_widget_value(kwargs, 'mask_type')
                mask = image_utilities.create_random_mask(noise, guider, sampler, high, low, latent, vae, mask_grow_by, mask_blur_size, mask_type)'''
                mask_grow_by = int(get_widget_value(kwargs, 'mask_grow_by'))
                mask_blur_size = get_widget_value(kwargs, 'mask_blur_size')
                bg_file = kwargs['background']
                bg_img, bg_mask = image_utilities.load_image(bg_file)
                resize_bg_img = image_utilities.resize(bg_img, width, height, "nearest-exact")
                segment_anything_img, segment_anything_mask = segment_anything(self.cache_grounding_dino_model, self.cache_sam_model, resize_bg_img, "person", 0.7)
                segment_anything_mask = image_utilities.expand_mask(segment_anything_mask, mask_grow_by, False, False, mask_blur_size, 0.0, 1.00, 1.00)
                latent["noise_mask"] = segment_anything_mask.reshape((-1, 1, segment_anything_mask.shape[-2], segment_anything_mask.shape[-1]))

            out, out_denoised = sampling_utilities.try_sample(noise, guider, sampler, high, latent, attempts)

            noise_face = sampling_utilities.get_noise(noise.seed + 1)

            guider_face = self.get_basic_face_guider(kwargs, positive, self.cache_pulid_model)

            out_face, out_denoised_face = sampling_utilities.try_sample(noise_face, guider_face, sampler, low, out_denoised, attempts)

            upscale_size = kwargs['upscale_image_by']
            unscaled_image = self.decode(kwargs, out_denoised_face)
            scaled_image = image_utilities.upscale_image_by(unscaled_image, "lanczos", upscale_size)

            '''#WE WILL MATCH THE FACE SHAPE HERE --BEGIN

            generated_image_face_landmarks = process_generated_image(bbox_model_name, scaled_image, 0.50, 10, 1.2, 10, "face", self.cache_face_helper, self.cache_generated_image_ratio, False)
            closest_face = get_best_face_match(generated_image_face_landmarks, self.cache_spreadsheet_landmarks, self.cache_faces_images, False)

            generated_face_cropped, generated_face_crop_infos = face_cropper(self.cache_insightface_cropper, scaled_image, 512, 1.00, 0.00, 0.00, 0, "large-small", True)
            closest_face_cropped, closest_face_crop_infos = face_cropper(self.cache_insightface_cropper, closest_face, 512, 1.00, 0.00, 0.00, 0, "large-small", True)

            face_match, face_match_landmark = face_shaper(generated_face_cropped, generated_face_crop_infos, closest_face_crop_infos, "ALL", "Landmarks")

            face_matched_combined, face_matched_combined_mask = face_shaper_composite(scaled_image, face_match, generated_face_crop_infos)

            #WE WILL MATCH THE FACE SHAPE HERE --END'''

            details_denoise = kwargs['details_denoise']

            if(details_denoise > 0):
                #max_dim = max(scaled_image.shape[1], scaled_image.shape[2])
                #ip_adapter_img = image_utilities.resize_image_any(scaled_image, max_dim, max_dim)

                #conditionned_ipadapter_model = self.apply_ipadapter(details_model, details_ip_adapter_model, details_clip_vision_model, ip_adapter_img, 5, 0.0, 1.0, kwargs)
                #cnet_image = process_image("tile", scaled_image)
                cond_zero = sampling_utilities.zero_out(positive)
                details_cnet_pos, details_cnet_neg = model_management_utilities.apply_controlnet_conditions(cond_zero, cond_zero, scaled_image, 1.0, 0.0, 1.0, vae, "tile", "sdxl")
                #upscaled_latent = self.encode(kwargs, face_matched_combined)
                upscaled_latent = self.encode(kwargs, scaled_image)

                noise_details = sampling_utilities.get_noise(noise_face.seed + 1)
                details_guider = self.get_details_cfg_guider(details_cnet_pos, details_cnet_neg, kwargs)
                details_sigmas = self.get_details_sigmas(kwargs)
                details_sampler = self.get_details_sampler(kwargs)
                
                out_details, out_denoised_details = sampling_utilities.try_sample(noise_details, details_guider, details_sampler, details_sigmas, upscaled_latent, attempts)

                final = self.decode(kwargs, out_denoised_details)
            else:
                #final = face_matched_combined
                final = scaled_image
                
            images.append(final)

            if(mask is not None):
                masks.append(mask)
        
        model_management_utilities.unload_models()
        return (images, masks, closest_face)
