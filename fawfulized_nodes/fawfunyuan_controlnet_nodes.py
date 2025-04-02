import folder_paths
from .samplawful_utilities import fawfrol_net

class FawfulizedHunyuanControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = fawfrol_net.load_controlnet(controlnet_path)
        return (controlnet,)

class FawfulizedHunyuanDiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, model, control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = fawfrol_net.load_controlnet(controlnet_path, model)
        return (controlnet,)


class FawfulizedHunyuanControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    DEPRECATED = True
    CATEGORY = "conditioning/controlnet"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        if strength == 0:
            return (conditioning, )

        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )


class FawfulizedHunyuanControlNetApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             },
                "optional": {"vae": ("VAE", ),
                             }
    }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "conditioning/controlnet"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None, extra_concat=[]):
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
        return (out[0], out[1])