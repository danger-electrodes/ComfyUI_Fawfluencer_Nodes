import folder_paths
import os
from .constants import ASPECT_RATIOS, IMG2IMG_FOLDER_CUSTOM_DATAS, CONTROL_NETS, IMG2IMG_INFLUENCER_FACE_CUSTOM_DATAS, BBOX_MODELS, IP_ADAPTER_WEIGHT_TYPES
from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
from .utilities import sampling, image, model_management
from server import PromptServer
from .utilities.detectors import get_faces_list, segment_anything
from .control_nets.control_nets_preprocessors import try_process_image as process_image
from ..custom_routes import refresh_folder
from .utilities.instant_id import apply_instantid, apply_instantid_combine, get_instantid_conditionning
from .utilities.head_orientation import get_closest_face_by_orientation

def get_all_value(kwargs):
    id = kwargs['unique_id']
    prompt = kwargs['prompt']
    int_id = int(id)
    values = prompt[str(int_id)]
    return values["inputs"]

def get_widget_value(kwargs, widget_name):
    return get_all_value(kwargs)[widget_name]


def get_dimensions(resolution):
    dimensions = resolution.split(' ')[0]
    width, height = map(int, dimensions.split('x'))
    return width, height

def create_image_folder(folder_name):
    path_to_image_folder = os.path.abspath(__file__)
    path_to_image_folder = os.path.dirname(os.path.dirname(path_to_image_folder))
    target_folder = os.path.join(path_to_image_folder, folder_name)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    return target_folder

input_folder = "img_to_img_folder"
path_to_image_folder = create_image_folder(input_folder)


def get_images_paths(path_to_image_folder, image_folder):
    input_folder = "img_to_img_folder"  # Assuming you defined this elsewhere
    sub_relative = os.path.relpath(image_folder, start=input_folder)
    full_path = os.path.join(path_to_image_folder, sub_relative)

    # Supported image extensions
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = []

    for root, dirs, files in os.walk(full_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

class Img2ImgFawfluencerNodeSDXL:

    def __init__(self):
        self.cache_sam_model = None
        self.cache_grounding_dino_model = None
        self.cache_model_ip_adapter = None
        self.cache_ip_adapter_model = None
        self.cache_clip_vision_model = None
        self.instant_id_model = None
        self.instant_id_work_model = None
        self.instant_id_image_prompt_embeds = None
        self.instant_id_uncond_image_prompt_embeds = None
        self.insightface_model = None
        self.instant_id_controlnet_model = None
        self.cache_face_image_file = None
        self.faces = None
        self.faces_batch = None

    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        folders = refresh_folder(input_folder)    

        # Define aspect ratios with human-readable format

        return {
                "required":
                {
                    "noise": ("NOISE", ),
                    "model": ("MODEL",),
                    "clip":("CLIP",),
                    "vae":("VAE",),
                    "resolution": (ASPECT_RATIOS, {"default":"896x1152 (7:9)"}),
                    "sampling_attempt_number":("INT", {"default": 10, "min": 1, "max": 100}),
                    "sampler_name": (SAMPLER_NAMES,),
                    "scheduler": (SCHEDULER_NAMES,),
                    "steps": ("INT", {"default": 20,"min": 5, "max": 100, "step": 1}),
                    "cfg": ("FLOAT", {"default": 6, "precision": 2}),
                    "denoise": ("FLOAT", {"default": 0.90,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "character_mask_strength": ("FLOAT", {"default": 0.10,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "background_mask_strength": ("FLOAT", {"default": 0.10,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "control_net_type" : (CONTROL_NETS,),
                    "control_net_strength": ("FLOAT", {"default": 0.50,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "control_net_start_at": ("FLOAT", {"default": 0.00,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "control_net_end_at": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "influencer_face": (sorted(files), {"custom_datas": IMG2IMG_INFLUENCER_FACE_CUSTOM_DATAS}),
                    "image_folder": (folders, {"custom_datas": IMG2IMG_FOLDER_CUSTOM_DATAS}),
                    "positive": ("STRING", {"multiline": True, "default": "", "placeholder": "positive prompt"}),
                    "negative": ("STRING", {"multiline": True, "default": "", "placeholder": "negative prompt"}),
                    "upscale_image_by": ("FLOAT", {"default": 1.50,"min": 1.0, "max": 3.0, "step": 0.01, "precision": 2}),
                    "details_control_net_type" : (CONTROL_NETS,),
                    "details_control_net_strength": ("FLOAT", {"default": 0.20,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "details_control_net_start_at": ("FLOAT", {"default": 0.00,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "details_control_net_end_at": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "details_sampler_name": (SAMPLER_NAMES,),
                    "details_scheduler": (SCHEDULER_NAMES,),
                    "details_steps": ("INT", {"default": 20,"min": 5, "max": 100, "step": 1}),
                    "details_cfg": ("FLOAT", {"default": 6, "precision": 2}),
                    "details_denoise": ("FLOAT", {"default": 0.30,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "details_noise_scale": ("FLOAT", {"default": 0.90,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "ip_adapter_strength": ("FLOAT", {"default": 0.8,"min": 0.0, "max": 1.0, "step": 0.01, "precision": 2}),
                    "ip_adapter_weight_type": (IP_ADAPTER_WEIGHT_TYPES,),
                },
                "hidden":
                {
                    "prompt": "PROMPT",
                    "unique_id": "UNIQUE_ID",
                }
            }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    OUTPUT_IS_LIST = (True,True,)
    RETURN_NAMES = ("images", "closest faces orientation match",)
    FUNCTION = "generate"
    CATEGORY = "Fawfulized/Influencer"

    def initialized(self, face_image_file):
        is_initialized = face_image_file == self.cache_face_image_file
        return is_initialized
    
    def initialize_faces(self, face_img, face_image_file, instantid_face_selection_type, model,instantid_strength, ip_strength, instantid_combine_embeddings):
        self.cache_face_image_file = face_image_file
        bbox_model_name = next((s for s in BBOX_MODELS if "face" in s), None)
        self.faces = get_faces_list(bbox_model_name, face_img, 0.50, 10, 1.5, 10, "face")
        self.faces_batch = image.image_list_to_batch(self.faces)

        if(instantid_face_selection_type == 'combine faces'):
            self.instant_id_work_model, self.instant_id_image_prompt_embeds, self.instant_id_uncond_image_prompt_embeds = apply_instantid_combine(self.instant_id_model, self.insightface_model, self.faces_batch, model, 0.0, 1.0, instantid_strength, ip_strength, 0.35, None, instantid_combine_embeddings)

    def generate(self, **kwargs):

        image_folder = kwargs['image_folder']
        
        images_paths = get_images_paths(path_to_image_folder, image_folder)

        imgs, _ = image.load_image_from_path_multiple(images_paths)

        attempts = kwargs['sampling_attempt_number']

        start_noise = kwargs['noise']
        model = kwargs['model']
        clip = kwargs['clip']
        vae = kwargs['vae']

        sampler_name = kwargs['sampler_name']
        scheduler = kwargs['scheduler']
        steps = int(kwargs['steps'])
        denoise = kwargs['denoise']
        character_mask_strength = kwargs['character_mask_strength']
        background_mask_strength = kwargs['background_mask_strength']
        cfg = kwargs['cfg']

        control_net_type = kwargs['control_net_type']
        control_net_strength = kwargs['control_net_strength']
        control_net_start_at = kwargs['control_net_start_at']
        control_net_end_at = kwargs['control_net_end_at']

        positive = kwargs['positive']
        negative = kwargs['negative']

        mask_grow_by = int(get_widget_value(kwargs, 'mask_grow_by'))
        mask_blur_size = get_widget_value(kwargs, 'mask_blur_size')

        width, height = get_dimensions(kwargs['resolution'])

        insight_face_provider = get_widget_value(kwargs, "insight_face_provider")
        instantid_strength = get_widget_value(kwargs, "instantid_strength")
        instantid_cnet_strength = get_widget_value(kwargs, "instantid_cnet_strength")
        ip_strength = get_widget_value(kwargs, "ip_strength")
        instantid_face_selection_type = get_widget_value(kwargs, "instantid_face_selection_type")
        instantid_combine_embeddings = get_widget_value(kwargs, "instantid_combine_embeddings")
        insight_face_input_width = int(get_widget_value(kwargs, "insight_face_input_width"))
        insight_face_input_height = int(get_widget_value(kwargs, "insight_face_input_height"))

        upscale_image_by = kwargs['upscale_image_by']
        details_control_net_type = kwargs['details_control_net_type']
        details_control_net_strength = kwargs['details_control_net_strength']
        details_control_net_start_at = kwargs['details_control_net_start_at']
        details_control_net_end_at = kwargs['details_control_net_end_at']
        details_sampler_name = kwargs['details_sampler_name']
        details_scheduler = kwargs['details_scheduler']
        details_steps = int(kwargs['details_steps'])
        details_cfg = kwargs['details_cfg']
        details_denoise = kwargs['details_denoise']
        details_noise_scale = kwargs['details_noise_scale']
        face_image_file = kwargs['influencer_face']

        ip_adapter_strength = kwargs['ip_adapter_strength']
        ip_adapter_weight_type = kwargs['ip_adapter_weight_type']

        if(self.cache_sam_model is None):
            self.cache_sam_model = model_management.load_sam_model()

        if(self.cache_grounding_dino_model is None):
            self.cache_grounding_dino_model = model_management.load_grounding_dino_model()

        if(self.cache_ip_adapter_model is None or self.cache_clip_vision_model is None or self.cache_model_ip_adapter is None):
            self.cache_model_ip_adapter, self.cache_ip_adapter_model, self.cache_clip_vision_model = model_management.load_ip_adapter(model, 0, "sdxl")

        if(self.instant_id_model is None):
            self.instant_id_model = model_management.load_instant_id_model()

        if(self.insightface_model is None):
            self.insightface_model = model_management.load_insight_face_model(insight_face_provider, insight_face_input_width, insight_face_input_height)

        if(self.instant_id_controlnet_model is None):
            self.instant_id_controlnet_model = model_management.load_instant_id_controlnet_model()

        face_img, face_mask = image.load_image(face_image_file)

        is_initialized = self.initialized(face_image_file)

        if(not is_initialized):
            self.initialize_faces(face_img, face_image_file, instantid_face_selection_type, model, instantid_strength, ip_strength, instantid_combine_embeddings)

        details_positive = "seamless blend with surroundings, consistent lighting and shadows, harmonized color palette, matched texture and detail, integrated into background, natural depth and perspective, slight imperfections, casual framing, soft focus areas, natural ambient light, same rendering style as background, unnoticeable edit, realistic environmental reflection"
        details_negative = "mismatched lighting, harsh color contrast, plastic look, overly sharp edges, too perfect, cinematic style, floating object, artificial shadows, hyperreal detail, style mismatch, unrealistic reflections, glossy texture, low detail compared to background"
        details_pos_cond, details_neg_cond = sampling.encode_text(clip, details_positive, details_negative)
        images = []
        for i in range(len(imgs)):

            #prepping the image
            img = imgs[i]
            img_resized = image.resize(img, width, height, "nearest-exact")
            latent = sampling.encode(vae, img_resized)

            #creating mask dynamically through segment anything
            _, character_mask = segment_anything(self.cache_grounding_dino_model, self.cache_sam_model, img_resized, "person", 0.7)
            character_mask = image.expand_mask(character_mask, mask_grow_by, False, False, mask_blur_size, 0.0, 1.00, 1.00)
            background_mask = image.get_inverse_mask(character_mask)
            character_mask = image.set_mask_strength(character_mask, character_mask_strength)
            background_mask = image.set_mask_strength(background_mask, background_mask_strength)
            combined_mask = image.combine_mask(character_mask, background_mask)
            latent = image.set_latent_noise_mask(latent, combined_mask)

            #processing image and conditionning for controlnet
            cnet_image = process_image(control_net_type, img_resized, attempts)
            pos_cond, neg_cond = sampling.encode_text(clip, positive, negative)
            cnet_pos, cnet_neg = model_management.apply_controlnet_conditions(pos_cond, neg_cond, cnet_image, control_net_strength, control_net_start_at, control_net_end_at, vae, control_net_type, "sdxl")

            #instant id
            if(instantid_face_selection_type == 'combine faces'):
                instant_id_pos, instant_id_neg = get_instantid_conditionning(self.insightface_model, self.faces_batch, self.instant_id_controlnet_model, instantid_cnet_strength, 0.0, 1.0, cnet_pos, cnet_neg, self.instant_id_image_prompt_embeds, self.instant_id_uncond_image_prompt_embeds, None, img_resized)
            else:
                #getting fest face orientation
                best_face = get_closest_face_by_orientation(img_resized, self.faces_batch)
                self.instant_id_work_model, instant_id_pos, instant_id_neg = apply_instantid(self.instant_id_model, self.insightface_model, self.instant_id_controlnet_model, best_face, model, cnet_pos, cnet_neg, 0.0, 1.0, instantid_strength, ip_strength, instantid_cnet_strength, 0.35, img_resized, None, instantid_combine_embeddings)
            
            #processing the image
            noise = sampling.get_noise(start_noise.seed + i)
            cfg_guider = sampling.get_cfg_guider(self.instant_id_work_model, instant_id_pos, instant_id_neg, cfg)
            sampler = sampling.get_sampler(sampler_name)
            sigmas = sampling.get_sigmas(self.instant_id_work_model, scheduler, steps, denoise)
            
            _, out_denoised = sampling.try_sample(noise, cfg_guider, sampler, sigmas, latent, attempts)

            #upscaling the image in pixel space then putting it in latent space
            base_image = sampling.decode(vae, out_denoised)
            upscaled_base_image = image.upscale_image_by(base_image, "lanczos", upscale_image_by)
            upscaled_latent = sampling.encode(vae, upscaled_base_image)


            #apply the ip adapter model
            #max_dim = max(upscaled_base_image.shape[1], upscaled_base_image.shape[2])
            ip_adapter_img = image.prepare_image_for_clipvision(upscaled_base_image)#image.resize_image_any(upscaled_base_image, max_dim, max_dim)
            applied_model = model_management.apply_ip_adapter_sdxl(self.cache_model_ip_adapter, self.cache_ip_adapter_model, self.cache_clip_vision_model, ip_adapter_img, ip_adapter_strength, 0.0, 1.0, ip_adapter_weight_type)

            #setting up the parameters for the details sampling
            details_noise = sampling.get_noise_scaled(noise.seed + 1, details_noise_scale)
            details_cnet_image = process_image(details_control_net_type, upscaled_base_image)
            details_cnet_cond_pos, details_cnet_neg_cond = model_management.apply_controlnet_conditions(details_pos_cond, details_neg_cond, details_cnet_image, details_control_net_strength, details_control_net_start_at, details_control_net_end_at, vae, details_control_net_type, "sdxl")
            details_cfg_guider = sampling.get_cfg_guider(applied_model, details_cnet_cond_pos, details_cnet_neg_cond, details_cfg)
            details_sampler = sampling.get_sampler(details_sampler_name)
            details_sigmas = sampling.get_sigmas(applied_model, details_scheduler, details_steps, details_denoise)

            #sampling and decoding
            _, details_out_denoised = sampling.try_sample(details_noise, details_cfg_guider, details_sampler, details_sigmas, upscaled_latent, attempts)

            final = sampling.decode(vae, details_out_denoised)

            images.append(final)

        return (images,)

