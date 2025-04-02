from .fawfulized_nodes.fawfunyuan_custom_nodes import (
    FawfulizedHunyuanSamplerCustom,
    FawfulizedHunyuanBasicScheduler,
    FawfulizedHunyuanKarrasScheduler,
    FawfulizedHunyuanExponentialScheduler,
    FawfulizedHunyuanPolyexponentialScheduler,
    FawfulizedHunyuanLaplaceScheduler,
    FawfulizedHunyuanVPScheduler,
    FawfulizedHunyuanBetaSamplingScheduler,
    FawfulizedHunyuanSDTurboScheduler,
    FawfulizedHunyuanKSamplerSelect,
    FawfulizedHunyuanSamplerEulerAncestral,
    FawfulizedHunyuanSamplerEulerAncestralCFGPP,
    FawfulizedHunyuanSamplerLMS,
    FawfulizedHunyuanSamplerDPMPP_3M_SDE,
    FawfulizedHunyuanSamplerDPMPP_2M_SDE,
    FawfulizedHunyuanSamplerDPMPP_SDE,
    FawfulizedHunyuanSamplerDPMPP_2S_Ancestral,
    FawfulizedHunyuanSamplerDPMAdaptative,
    FawfulizedHunyuanSplitSigmas,
    FawfulizedHunyuanSplitSigmasDenoise,
    FawfulizedHunyuanFlipSigmas,
    FawfulizedHunyuanSetFirstSigma,
    FawfulizedHunyuanCFGGuider,
    FawfulizedHunyuanDualCFGGuider,
    FawfulizedHunyuanBasicGuider,
    FawfulizedHunyuanRandomNoise,
    FawfulizedHunyuanDisableNoise,
    FawfulizedHunyuanAddNoise,
    FawfulizedHunyuanSamplerCustomAdvanced,
)

from .fawfulized_nodes.fawfluxencer_node import FawfluxencerNode
from .fawfulized_nodes.fawface_model_spreadsheet_realism_node import FawfaceModelSpreadsheetRealismNode

from .fawfulized_nodes.fawfake_authentic_image_save_node import FawfakeAuthenticImageSaveNode
from .fawfulized_nodes.empty_image_list import FawfulizedEmptyImageList
from .fawfulized_nodes.add_images_to_list import FawfulizedAddImagesToImageList

from .fawfulized_nodes.fawfunyuan_controlnet_nodes import (
    FawfulizedHunyuanControlNetLoader,
    FawfulizedHunyuanDiffControlNetLoader,
    FawfulizedHunyuanControlNetApply,
    FawfulizedHunyuanControlNetApplyAdvanced,
)

from .fawfulized_nodes.fawfunyuan_latent_video import FawfulizedHunyuanLatentVideo
from .fawfulized_nodes.fawfunyuan_set_latent_noise_mask import FawfulizedHunyuanSetLatentNoiseMask
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FawfakeAuthenticImageSaveNode": FawfakeAuthenticImageSaveNode,
    "FawfluxencerNode": FawfluxencerNode,
    "FawfaceModelSpreadsheetRealismNode": FawfaceModelSpreadsheetRealismNode,
    "FawfulizedEmptyImageList": FawfulizedEmptyImageList,
    "FawfulizedAddImagesToImageList": FawfulizedAddImagesToImageList,
    "FawfulizedHunyuanLatentVideo": FawfulizedHunyuanLatentVideo,
    "FawfulizedHunyuanSetLatentNoiseMask":FawfulizedHunyuanSetLatentNoiseMask,

    "FawfulizedHunyuanSamplerCustom": FawfulizedHunyuanSamplerCustom,
    "FawfulizedHunyuanBasicScheduler": FawfulizedHunyuanBasicScheduler,
    "FawfulizedHunyuanKarrasScheduler": FawfulizedHunyuanKarrasScheduler,
    "FawfulizedHunyuanExponentialScheduler": FawfulizedHunyuanExponentialScheduler,
    "FawfulizedHunyuanPolyexponentialScheduler": FawfulizedHunyuanPolyexponentialScheduler,
    "FawfulizedHunyuanLaplaceScheduler": FawfulizedHunyuanLaplaceScheduler,
    "FawfulizedHunyuanVPScheduler": FawfulizedHunyuanVPScheduler,
    "FawfulizedHunyuanBetaSamplingScheduler": FawfulizedHunyuanBetaSamplingScheduler,
    "FawfulizedHunyuanSDTurboScheduler": FawfulizedHunyuanSDTurboScheduler,
    "FawfulizedHunyuanKSamplerSelect": FawfulizedHunyuanKSamplerSelect,
    "FawfulizedHunyuanSamplerEulerAncestral": FawfulizedHunyuanSamplerEulerAncestral,
    "FawfulizedHunyuanSamplerEulerAncestralCFGPP": FawfulizedHunyuanSamplerEulerAncestralCFGPP,
    "FawfulizedHunyuanSamplerLMS": FawfulizedHunyuanSamplerLMS,
    "FawfulizedHunyuanSamplerDPMPP_3M_SDE": FawfulizedHunyuanSamplerDPMPP_3M_SDE,
    "FawfulizedHunyuanSamplerDPMPP_2M_SDE": FawfulizedHunyuanSamplerDPMPP_2M_SDE,
    "FawfulizedHunyuanSamplerDPMPP_SDE": FawfulizedHunyuanSamplerDPMPP_SDE,
    "FawfulizedHunyuanSamplerDPMPP_2S_Ancestral": FawfulizedHunyuanSamplerDPMPP_2S_Ancestral,
    "FawfulizedHunyuanSamplerDPMAdaptative": FawfulizedHunyuanSamplerDPMAdaptative,
    "FawfulizedHunyuanSplitSigmas": FawfulizedHunyuanSplitSigmas,
    "FawfulizedHunyuanSplitSigmasDenoise": FawfulizedHunyuanSplitSigmasDenoise,
    "FawfulizedHunyuanFlipSigmas": FawfulizedHunyuanFlipSigmas,
    "FawfulizedHunyuanSetFirstSigma": FawfulizedHunyuanSetFirstSigma,

    "FawfulizedHunyuanCFGGuider": FawfulizedHunyuanCFGGuider,
    "FawfulizedHunyuanDualCFGGuider": FawfulizedHunyuanDualCFGGuider,
    "FawfulizedHunyuanBasicGuider": FawfulizedHunyuanBasicGuider,
    "FawfulizedHunyuanRandomNoise": FawfulizedHunyuanRandomNoise,
    "FawfulizedHunyuanDisableNoise": FawfulizedHunyuanDisableNoise,
    "FawfulizedHunyuanAddNoise": FawfulizedHunyuanAddNoise,
    "FawfulizedHunyuanSamplerCustomAdvanced": FawfulizedHunyuanSamplerCustomAdvanced,

    "FawfulizedHunyuanControlNetLoader" :FawfulizedHunyuanControlNetLoader,
    "FawfulizedHunyuanDiffControlNetLoader" :FawfulizedHunyuanDiffControlNetLoader,
    "FawfulizedHunyuanControlNetApply" :FawfulizedHunyuanControlNetApply,
    "FawfulizedHunyuanControlNetApplyAdvanced" :FawfulizedHunyuanControlNetApplyAdvanced,
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FawfakeAuthenticImageSaveNode": "Fawfake Authentic Image Save Node",
    "FawfluxencerNode": "Fawfluxencer Node",
    "FawfaceModelSpreadsheetRealismNode" : "Fawface Model Spreadsheet Realism Node",
    "FawfulizedEmptyImageList" : "Empty Image List",
    "FawfulizedAddImagesToImageList": "Fawfulized Add Images To Image List",
    "FawfulizedHunyuanLatentVideo": "Fawfulized Hunyuan Latent Video",
    "FawfulizedHunyuanSetLatentNoiseMask": "Fawfulized Hunyuan Set Latent Noise Mask",

    "FawfulizedHunyuanSamplerCustom": "Fawfulized Hunyuan Sampler Custom",
    "FawfulizedHunyuanBasicScheduler": "Fawfulized Hunyuan Basic Scheduler",
    "FawfulizedHunyuanKarrasScheduler": "Fawfulized Hunyuan Karras Scheduler",
    "FawfulizedHunyuanExponentialScheduler": "Fawfulized Hunyuan Exponential Scheduler",
    "FawfulizedHunyuanPolyexponentialScheduler": "Fawfulized Hunyuan Polyexponential Scheduler",
    "FawfulizedHunyuanLaplaceScheduler": "Fawfulized Hunyuan Laplace Scheduler",
    "FawfulizedHunyuanVPScheduler": "Fawfulized Hunyuan VP Scheduler",
    "FawfulizedHunyuanBetaSamplingScheduler": "Fawfulized Hunyuan Beta Sampling Scheduler",
    "FawfulizedHunyuanSDTurboScheduler": "Fawfulized Hunyuan SD Turbo Scheduler",
    "FawfulizedHunyuanKSamplerSelect": "Fawfulized Hunyuan KSampler Select",
    "FawfulizedHunyuanSamplerEulerAncestral": "Fawfulized Hunyuan Sampler Euler Ancestral",
    "FawfulizedHunyuanSamplerEulerAncestralCFGPP": "Fawfulized Hunyuan Sampler Euler Ancestral CFGPP",
    "FawfulizedHunyuanSamplerLMS": "Fawfulized Hunyuan Sampler LMS",
    "FawfulizedHunyuanSamplerDPMPP_3M_SDE": "Fawfulized Hunyuan Sampler DPMPP_3M_SDE",
    "FawfulizedHunyuanSamplerDPMPP_2M_SDE": "Fawfulized Hunyuan Sampler DPMPP_2M_SDE",
    "FawfulizedHunyuanSamplerDPMPP_SDE": "Fawfulized Hunyuan Sampler DPMPP_SDE",
    "FawfulizedHunyuanSamplerDPMPP_2S_Ancestral": "Fawfulized Hunyuan Sampler DPMPP_2S_Ancestral",
    "FawfulizedHunyuanSamplerDPMAdaptative": "Fawfulized Hunyuan Sampler DPM Adaptative",
    "FawfulizedHunyuanSplitSigmas": "Fawfulized Hunyuan Split Sigmas",
    "FawfulizedHunyuanSplitSigmasDenoise": "FawfulizedHunyuan Split Sigmas Denoise",
    "FawfulizedHunyuanFlipSigmas": "Fawfulized Hunyuan Flip Sigmas",
    "FawfulizedHunyuanSetFirstSigma": "FawfulizedHunyuan Set First Sigma",

    "FawfulizedHunyuanCFGGuider": "Fawfulized Hunyuan CFG Guider",
    "FawfulizedHunyuanDualCFGGuider": "Fawfulized Hunyuan Dual CFG Guider",
    "FawfulizedHunyuanBasicGuider": "Fawfulized Hunyuan Basic Guider",
    "FawfulizedHunyuanRandomNoise": "Fawfulized Hunyuan Random Noise",
    "FawfulizedHunyuanDisableNoise": "Fawfulized Hunyuan Disable Noise",
    "FawfulizedHunyuanAddNoise": "Fawfulized Hunyuan Add Noise",
    "FawfulizedHunyuanSamplerCustomAdvanced": "Fawfulized Hunyuan Sampler Custom Advanced",

    "FawfulizedHunyuanControlNetLoader": "Fawfulized Hunyuan ControlNet Loader",
    "FawfulizedHunyuanDiffControlNetLoader": "Fawfulized Hunyuan Diff ControlNet Loader",
    "FawfulizedHunyuanControlNetApply": "Fawfulized Hunyuan ControlNet Apply",
    "FawfulizedHunyuanControlNetApplyAdvanced": "Fawfulized Hunyuan ControlNet Apply Advanced",
}

WEB_DIRECTORY = "./web"