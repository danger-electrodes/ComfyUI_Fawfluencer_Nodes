import random
from .utilities import model_management as model_management_utilities
from .utilities import image as image_utilities
from .utilities.detectors import MAX_RESOLUTION, get_faces_list
from .utilities import sampling as sampling_utilities
import comfy.samplers
from .constants import CONTROL_NETS, LIGHTS, LIGHTS_INTENSITIES, BBOX_MODELS
from .utilities.ic_light import get_light_source, create_lightmap
from.utilities.expressions import create_expression
from .control_nets.control_nets_preprocessors import try_process_image as process_image


class FawfaceModelSpreadsheetRealismNode:
    @classmethod
    def INPUT_TYPES(s):  
        return {"required": {
                        "noise" : ("NOISE",),
                        "model" : ("MODEL",),
                        "clip": ("CLIP",),
                        "vae" : ("VAE",),
                        "model_sd15_for_ic_light" : ("MODEL",),
                        "clip_sd15_for_ic_light": ("CLIP",),
                        "vae_sd15_for_ic_light" : ("VAE",),
                        "faces_model_spreadsheets": ("IMAGE", ),
                        "lights_count": ("INT", {"default": 1,"min": 0, "max": 10, "step": 1}),
                        "ic_light_multiplier":("FLOAT", {"default": 0.182, "min": 0.000, "max": 1.000, "step": 0.001, "precision" : 3}),
                        "ic_light_shadow_expand_size":("INT", {"default": 5, "min": 0, "max": 100}),
                        "ic_light_shadow_blur":("FLOAT", {"default": 10, "min": 0.00, "max": 50.00, "step": 0.01, "precision" : 2}),
                        "ic_light_source_mulitplier": ("FLOAT", {"default": 1.000, "precision": 2, "min": 0.000, "max": 1.000, "step": 0.001}),
                        "ic_light_sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default":"dpmpp_2m"}),
                        "ic_light_scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default":"karras"}),
                        "ic_light_steps": ("INT", {"default": 20,"min": 5, "max": 100, "step": 1}),
                        "ic_light_cfg": ("FLOAT", {"default": 10, "precision": 2}),
                        "ic_light_denoise": ("FLOAT", {"default": 1.00, "precision": 2, "min": 0.00, "max": 1.00, "step": 0.01}),
                        "random_expression_multiplier" : ("FLOAT", {"default": 1.00, "precision": 2, "min": 0.00, "max": 1.00, "step": 0.01}),
                        "type": (["flux", "sdxl"], {"default":"sdxl"}),
                        "denoise_passes": ("INT", {"default": 3,"min": 1, "max": 10, "step": 1}),
                        "sampling_attempts": ("INT", {"default": 5,"min": 5, "max": 20, "step": 1}),
                        "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default":"dpmpp_2m_sde"}),
                        "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default":"karras"}),
                        "steps": ("INT", {"default": 26,"min": 5, "max": 100, "step": 1}),
                        "cfg": ("FLOAT", {"default": 6, "precision": 2}),
                        "denoise": ("FLOAT", {"default": 0.41, "precision": 2, "min": 0.00, "max": 1.00, "step": 0.01}),
                        "final_step_sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default":"dpmpp_2m_sde"}),
                        "final_step_scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default":"karras"}),
                        "final_step_steps": ("INT", {"default": 15,"min": 5, "max": 100, "step": 1}),
                        "final_step_cfg": ("FLOAT", {"default": 6, "precision": 2}),
                        "final_step_denoise": ("FLOAT", {"default": 0.20, "precision": 2, "min": 0.00, "max": 1.00, "step": 0.01}),
                        "positive": ("STRING", {"multiline": True, "default": "caucasian young female face, detailed skin, perfect skin, smooth skin, photorealistic, real photo, shadows on face, casted shadows, light,", "placeholder": "positive prompt"}),
                        "negative": ("STRING", {"multiline": True, "default": "black female face, asian female face, text, bad quality, noise, grain, skin imperfections, deformed, 3d, 2d, painting, drawing, sketch, no shadows, flat,", "placeholder": "negative prompt"}),
                        "control_net":(CONTROL_NETS, {"default":"anyline"}),
                        "control_net_start_at":("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "control_net_end_at":("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "control_net_strength":("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_detector_model_name": (BBOX_MODELS, ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                        "labels": ("STRING", {"multiline": True, "default": "face", "placeholder": "List the types of segments to be allowed, separated by commas"}),
                      },
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("realistic_faces_combined", "realistic_faces", "light_sources")
    OUTPUT_IS_LIST = (False, True, True)
    FUNCTION = "generate"

    CATEGORY = "ImpactPack/Detector"
    
    def get_cfg_guider(self, model, cfg, positive, negative):
        guider = sampling_utilities.get_cfg_guider(model, positive, negative, cfg)
        return guider
    
    def get_sampler(self, sampler_name):
        sampler = sampling_utilities.get_sampler(sampler_name)
        return sampler
    
    def get_sigmas(self, model, steps, denoise, scheduler):
        sigmas = sampling_utilities.get_sigmas(model, scheduler, steps, denoise)
        return sigmas
    
    def process_controlnet_image(self, preprocessor, face_img):
        out = process_image(preprocessor, face_img)
        return out
    
    def get_controlnet_conditions(self, positive, negative, strength, start, end, vae, preprocessor_type, base_model_type, face_img):
        processed_image = self.process_controlnet_image(preprocessor_type, face_img)
        new_pos, new_neg = model_management_utilities.apply_controlnet_conditions(positive, negative, processed_image, strength, start, end, vae, preprocessor_type, base_model_type)
        return new_pos, new_neg
    
    def encode_text(self, clip, positive, negative, flux_guidance = None):
        return sampling_utilities.encode_text(clip, positive, negative, flux_guidance)
    
    def decode(self, vae, samples):
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images
    
    def encode(self, vae, pixels):
        t = vae.encode(pixels[:,:,:,:3])
        return {"samples":t}
    
    def get_random_item(self, items):
        return random.choice(items)
    
    def make_realistic(self, vae, orig_cropped_image, denoise_passes, noise, positive_cond, negative_cond, control_net_strength, control_net_start, control_net_end, control_net, type, model, cfg, steps, denoise, scheduler, sampler, sampling_attempts):

        latent = self.encode(vae, orig_cropped_image)

        for pass_index in range(denoise_passes):
            noise_pass = sampling_utilities.get_noise(noise.seed + pass_index)
            factor = 1.0/(pass_index+1)
            cnet_positive_cond, cnet_negative_cond = self.get_controlnet_conditions(positive_cond, negative_cond, control_net_strength * factor, control_net_start, control_net_end, vae, control_net, type, orig_cropped_image)
            guider = self.get_cfg_guider(model, cfg, cnet_positive_cond, cnet_negative_cond)
            sigmas = self.get_sigmas(model, steps, denoise * factor, scheduler)
            out, out_denoised = sampling_utilities.try_sample(noise_pass, guider, sampler, sigmas, latent, sampling_attempts)
            latent = out_denoised

        detailed_image = self.decode(vae, latent)

        return detailed_image
    
    def interpolate_color(self, value, min_hex, max_hex):
        min_rgb = tuple(int(min_hex[i:i+2], 16) for i in (1, 3, 5))
        max_rgb = tuple(int(max_hex[i:i+2], 16) for i in (1, 3, 5))
        interp_rgb = tuple(int(min_c + (max_c - min_c) * value) for min_c, max_c in zip(min_rgb, max_rgb))
        return f'#{interp_rgb[0]:02X}{interp_rgb[1]:02X}{interp_rgb[2]:02X}'
    
    def relight(self, vae_sd15_for_ic_light, detailed_image_for_lighting, clip_sd15_for_ic_light, ic_light_multiplier, ic_light_source_mulitplier, ic_light_model, ic_light_cfg, noise, light_index, ic_light_sampler, ic_light_sigmas, ic_light_shadow_expand_size, ic_light_shadow_blur, sampling_attempts):
        
        normal_map = process_image("metric3Dnormalmap", detailed_image_for_lighting)

        _, H, W, _ = detailed_image_for_lighting.shape
        normal_map = image_utilities.resize(normal_map, W, H)

        ic_light_latent = self.encode(vae_sd15_for_ic_light, detailed_image_for_lighting)
        random_light = self.get_random_item(LIGHTS)
        random_intensity = self.get_random_item(LIGHTS_INTENSITIES)
        random_light_angle = random.uniform(0.0, 360.0)

        start_color = self.interpolate_color(random.uniform(0.0, 1.0), random_light['start_color_min'], random_light['start_color_max'])
        end_color = self.interpolate_color(random.uniform(0.0, 1.0), random_light['end_color_min'], random_light['end_color_max'])

        light_source = create_lightmap(normal_map, random_light_angle, ic_light_shadow_expand_size, ic_light_shadow_blur, start_color, end_color) # get_light_source(random_light_position, ic_light_source_mulitplier, start_color, end_color, W, H)
        light_source_latent = self.encode(vae_sd15_for_ic_light, light_source)

        ic_light_positive_text = f"{random_intensity}, {random_light['name']}" # for example, [faint] [sunlight]
        ic_light_negative_text = "bad quality, overexposed, overly bright, glowing excessively, washed out, too intense, extreme contrast, unnatural highlights, blinding light, oversaturated, no detail in highlights, harsh glare, pure white spots, excessive bloom effect, unrealistic radiance, lens flare overload, loss of shadow detail, lack of depth, artificial brightness, blown-out whites, light bleeding too much"

        ic_light_pos_cond, ic_light_neg_cond = self.encode_text(clip_sd15_for_ic_light, ic_light_positive_text, ic_light_negative_text)
        lighting_positive_cond, lighting_negative_cond, _ = model_management_utilities.apply_ic_light_model(ic_light_pos_cond, ic_light_neg_cond, vae_sd15_for_ic_light, ic_light_latent, ic_light_multiplier)
        ic_light_guider = self.get_cfg_guider(ic_light_model, ic_light_cfg, lighting_positive_cond, lighting_negative_cond)

        ic_light_noise = sampling_utilities.get_noise(noise.seed + light_index)
        relit_latent, denoised_relit_latent = sampling_utilities.try_sample(ic_light_noise, ic_light_guider, ic_light_sampler, ic_light_sigmas, light_source_latent, sampling_attempts)
        relit_image = self.decode(vae_sd15_for_ic_light, denoised_relit_latent)

        return relit_image, ic_light_positive_text, ic_light_negative_text, light_source
    
    def refine(self, vae, image, noise, denoise_passes, final_guider, final_sampler, final_sigmas, sampling_attempts):
        final_latent = self.encode(vae, image)
        final_noise_pass = sampling_utilities.get_noise(noise.seed + denoise_passes)
        out_final, out_denoised_final = sampling_utilities.try_sample(final_noise_pass, final_guider, final_sampler, final_sigmas, final_latent, sampling_attempts)

        detailed_face = self.decode(vae, out_denoised_final)

        return detailed_face

    def generate(self, **kwargs):

        noise = kwargs['noise']
        denoise_passes = int(kwargs['denoise_passes'])

        model = kwargs['model']
        clip = kwargs['clip']
        vae = kwargs['vae']

        model_sd15_for_ic_light = kwargs['model_sd15_for_ic_light']
        clip_sd15_for_ic_light = kwargs['clip_sd15_for_ic_light']
        vae_sd15_for_ic_light = kwargs['vae_sd15_for_ic_light']
        ic_light_multiplier = kwargs['ic_light_multiplier']
        ic_light_source_mulitplier = kwargs['ic_light_source_mulitplier']
        
        random_expression_multiplier = kwargs['random_expression_multiplier']
        ic_light_sampler_name = kwargs['ic_light_sampler_name']
        ic_light_scheduler = kwargs['ic_light_scheduler']

        ic_light_steps = kwargs['ic_light_steps']
        ic_light_cfg = kwargs['ic_light_cfg']
        ic_light_denoise = kwargs['ic_light_denoise']
        ic_light_shadow_expand_size = int(kwargs['ic_light_shadow_expand_size'])
        ic_light_shadow_blur = kwargs['ic_light_shadow_blur']

        ic_light_model = model_management_utilities.load_ic_light_model(model_sd15_for_ic_light)
    
        lights_count = int(kwargs['lights_count'])

        type = kwargs['type']
        sampling_attempts = int(kwargs['sampling_attempts'])

        sampler_name = kwargs['sampler_name']
        scheduler = kwargs['scheduler']
        steps = int(kwargs['steps'])
        cfg = kwargs['cfg']
        denoise = kwargs['denoise']

        final_step_sampler_name = kwargs['final_step_sampler_name']
        final_step_scheduler = kwargs['final_step_scheduler']
        final_step_steps = int(kwargs['final_step_steps'])
        final_step_cfg = kwargs['final_step_cfg']
        final_step_denoise = kwargs['final_step_denoise']

        pos_text = kwargs['positive']
        neg_text = kwargs['negative']
        control_net = kwargs['control_net']
        control_net_start = kwargs['control_net_start_at']
        control_net_end = kwargs['control_net_end_at']
        control_net_strength = kwargs['control_net_strength']

        bbox_detector_model_name = kwargs['bbox_detector_model_name']
        faces_model_spreadsheets = kwargs['faces_model_spreadsheets']
        threshold = kwargs['threshold']
        dilation = kwargs['dilation']
        crop_factor = kwargs['crop_factor']
        drop_size = kwargs['drop_size']
        labels = kwargs['labels']

        faces_list = get_faces_list(bbox_detector_model_name, faces_model_spreadsheets, threshold, dilation, crop_factor, drop_size, labels)
        cropped_list = []
        lights_list = []

        positive_cond, negative_cond = self.encode_text(clip, pos_text, neg_text)

        ic_light_sampler = self.get_sampler(ic_light_sampler_name)
        sampler = self.get_sampler(sampler_name)
        ic_light_sigmas = self.get_sigmas(ic_light_model, ic_light_steps, ic_light_denoise, ic_light_scheduler)
        final_sampler = self.get_sampler(final_step_sampler_name)
        #final_guider = self.get_cfg_guider(model, final_step_cfg, positive_cond, negative_cond)
        final_sigmas = self.get_sigmas(model, final_step_steps, final_step_denoise, final_step_scheduler)

        for face_index in range(len(faces_list)):    
            orig_cropped_image = faces_list[face_index]

            # MAKING MODEL REALISTIC THROUGH STEP ITERATIONS
            detailed_image = self.make_realistic(vae, orig_cropped_image, denoise_passes, noise, positive_cond, negative_cond, control_net_strength, control_net_start, control_net_end, control_net, type, model, cfg, steps, denoise, scheduler, sampler, sampling_attempts)
            cropped_list.append(detailed_image)

            for light_index in range(lights_count):

                detailed_image_for_lighting = detailed_image.clone()

                # EXPRESSIONS
                detailed_image_for_lighting = create_expression(detailed_image_for_lighting, random_expression=True, random_expression_multiplier=random_expression_multiplier)

                # RELIGHTING
                relit_image, ic_light_positive, ic_light_negative, light_source = self.relight(vae_sd15_for_ic_light, detailed_image_for_lighting, clip_sd15_for_ic_light, ic_light_multiplier, ic_light_source_mulitplier, ic_light_model, ic_light_cfg, noise, light_index, ic_light_sampler, ic_light_sigmas, ic_light_shadow_expand_size, ic_light_shadow_blur, sampling_attempts)

                # REFINING
                refiner_positive = f"{pos_text}, {ic_light_positive}"
                refiner_negative = f"{neg_text}, {ic_light_negative}"
                refiner_positive_cond, refiner_negative_cond = self.encode_text(clip, refiner_positive, refiner_negative)
                refiner_final_guider = self.get_cfg_guider(model, final_step_cfg, refiner_positive_cond, refiner_negative_cond)
                detailed_face = self.refine(vae, relit_image, noise, denoise_passes, refiner_final_guider, final_sampler, final_sigmas, sampling_attempts)
                cropped_list.append(detailed_face)
                lights_list.append(light_source)
   
        cropped_batch = image_utilities.image_list_to_batch(cropped_list)

        inv_cropped_batch = cropped_batch.clone()
        inv_cropped_batch = image_utilities.flip_batch(inv_cropped_batch, flip_x=True)

        batch_combined = image_utilities.merge_batch(cropped_batch, inv_cropped_batch)

        combined_list = image_utilities.image_batch_to_list(batch_combined)
        combined = image_utilities.combine_images_dynamic(combined_list, (lights_count + 1) * 2) # since we are also adding the inverted images, we double the length of the row
        return (combined, combined_list, lights_list)
