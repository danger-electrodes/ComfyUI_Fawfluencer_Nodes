import node_helpers
import torch
import comfy.utils
import latent_preview
import comfy.samplers
import comfy.sample
from comfy.samplers import KSAMPLER
from .model_management import unload_models
import numpy as np
import gc 
import copy

def prepare_noise_scaled(latent_image, seed, scale, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        return noise * scale

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises * scale

class Noise_RandomNoise_Scaled:
    def __init__(self, seed, scale):
        self.seed = seed
        self.scale = scale

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return prepare_noise_scaled(latent_image, self.seed, self.scale, batch_inds)
    
class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
    
def zero_out(conditionning):
    negative = []
    for t in conditionning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        n = [torch.zeros_like(t[0]), d]
        negative.append(n)
    return negative

def encode_text(clip, text, neg_text, flux_guidance = None):
    tokens = clip.tokenize(text)
    positive  = clip.encode_from_tokens_scheduled(tokens)

    neg_tokens = clip.tokenize(neg_text)
    negative  = clip.encode_from_tokens_scheduled(neg_tokens)

    if(flux_guidance is not None):
        positive = node_helpers.conditioning_set_values(positive, {"guidance": flux_guidance})

    return positive, negative


def sample(noise, guider, sampler, sigmas, latent_image):
    # Ensure latent is a fresh copy
    latent = copy.deepcopy(latent_image)  # Deep copy to prevent reference leaks
    latent_image = latent["samples"]
    
    # Fix channels
    latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
    latent["samples"] = latent_image

    noise_mask = latent.get("noise_mask", None)

    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # Generate noise and sample
    generated_noise = noise.generate_noise(latent)  # Ensure noise is new each iteration
    samples = guider.sample(generated_noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
    
    # Move samples to CPU before reassigning
    samples_cpu = samples.cpu()
    del samples
    torch.cuda.empty_cache()

    out = copy.deepcopy(latent)  # Deep copy to prevent reference leaks
    out["samples"] = samples_cpu  # Use CPU tensor

    if "x0" in x0_output:
        out_denoised = copy.deepcopy(latent)  # Another deep copy
        x0_cpu = x0_output["x0"].cpu()  # Move to CPU before assignment
        del x0_output["x0"]  # Remove old GPU tensor
        out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_cpu)
    else:
        out_denoised = out

    # Final cleanup
    del latent, latent_image, noise_mask, generated_noise
    torch.cuda.empty_cache()
    gc.collect()

    return out, out_denoised

def try_sample(noise, guider, sampler, sigmas, latent, attempts=1):
    for i in range(attempts):
        try:
            out, out_denoised = sample(noise, guider, sampler, sigmas, latent)
            break
        except torch.OutOfMemoryError:  # Catch only PyTorch OOM
            unload_models()
            if i == attempts - 1:
                raise Exception("All attempts to sample have failed, try increasing the value of 'sampling_attempt_number'.")
    return out, out_denoised

def get_sigmas(model, scheduler, steps, denoise):
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = int(steps/denoise)

    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    sigmas = sigmas[-(steps + 1):]
    return sigmas

def split_sigmas(sigmas, step):
    sigmas1 = sigmas[:step + 1]
    sigmas2 = sigmas[step:]
    return sigmas1, sigmas2

def unify_sigmas(sigmas1, sigmas2):
    if len(sigmas1) == 0:
        return torch.tensor(sigmas2, dtype=torch.float32)
    if len(sigmas2) == 0:
        return torch.tensor(sigmas1, dtype=torch.float32)
    
    if sigmas1[-1] == sigmas2[0]:
        unified = np.concatenate((sigmas1, sigmas2[1:]))
    else:
        unified = np.concatenate((sigmas1, sigmas2))

    return torch.tensor(unified, dtype=torch.float32)

def multiply_sigmas(sigmas, factor, start, end):
    # Clone the sigmas to ensure the input is not modified (stateless)
    sigmas = sigmas.clone()
    
    total_sigmas = len(sigmas)
    start_idx = int(start * total_sigmas)
    end_idx = int(end * total_sigmas)

    for i in range(start_idx, end_idx):
        sigmas[i] *= factor

    return sigmas


class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

def get_basic_guider(model, conditioning):
    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    return guider

def get_cfg_guider(model, positive, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider

def get_sampler(sampler_name):
    sampler = comfy.samplers.sampler_object(sampler_name)
    return sampler

def lying_sigma_sampler(
    model,
    x,
    sigmas,
    *,
    lss_wrapped_sampler,
    lss_dishonesty_factor,
    lss_startend_percent,
    **kwargs,
):
    start_percent, end_percent = lss_startend_percent
    ms = model.inner_model.inner_model.model_sampling
    start_sigma, end_sigma = (
        round(ms.percent_to_sigma(start_percent), 4),
        round(ms.percent_to_sigma(end_percent), 4),
    )
    del ms

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if end_sigma <= sigma_float <= start_sigma:
            sigma = sigma * (1.0 + lss_dishonesty_factor)
        return model(x, sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return lss_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **lss_wrapped_sampler.extra_options,
    )

def get_lying_sampler(sampler, dishonesty_factor, start, end):
    extra_options = extra_options={"lss_wrapped_sampler": sampler, "lss_dishonesty_factor": dishonesty_factor, "lss_startend_percent": (start, end),}
    lying_sampler = KSAMPLER(lying_sigma_sampler, extra_options=extra_options,)
    return lying_sampler

    
def get_noise(noise_seed):
    noise = Noise_RandomNoise(noise_seed)
    return noise

def get_noise_scaled(noise_seed, scale):
    noise = Noise_RandomNoise_Scaled(noise_seed, scale)
    return noise

def disable_noise():
    return Noise_EmptyNoise()

def rescale_model_cfg(model, multiplier):
    def rescale_cfg(args):
        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        sigma = args["sigma"]
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        x_orig = args["input"]

        #rescale cfg has to be done on v-pred model output
        x = x_orig / (sigma * sigma + 1.0)
        cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
        uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

        #rescalecfg
        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1,2,3), keepdim=True)

        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

        return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

    m = model.clone()
    m.set_model_sampler_cfg_function(rescale_cfg)
    return m
    
def encode(vae, image):
    t = vae.encode(image[:,:,:,:3])
    return {"samples":t}

def decode(vae, samples):
    images = vae.decode(samples["samples"])
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images