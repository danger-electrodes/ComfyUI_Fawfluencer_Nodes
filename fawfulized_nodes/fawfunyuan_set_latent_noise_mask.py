import torch
import comfy.model_management

class FawfulizedHunyuanSetLatentNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "frame_width": ("INT",),
                              "frame_height": ("INT",),
                              "frame_count": ("INT",),
                              "batch_size": ("INT",),
                              "start_denoise": ("FLOAT",),
                              "end_denoise": ("FLOAT",),
                              "gradual": ("BOOLEAN", {"default": False})
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"

    CATEGORY = "latent/inpaint"

    def set_mask(self, samples, frame_width, frame_height, frame_count, batch_size, start_denoise, end_denoise, gradual=False):

        print(start_denoise)
        print(end_denoise)
        s = samples.copy()

        # Create the noise mask tensor
        mask = torch.zeros(
            [batch_size, 16, ((frame_count - 1) // 4) + 1, frame_height // 8, frame_width // 8], 
            device=comfy.model_management.intermediate_device()
        )

        if gradual:
            # Create a smooth transition (gradual increase) or constant value mask
            num_latent_frames = mask.shape[2]  # Get the number of latent frames
            transition_values = torch.linspace(start_denoise, end_denoise, steps=num_latent_frames, device=mask.device)  # Gradual values from start_denoise to end_denoise

            # Apply the transition to each latent frame
            for i in range(0, num_latent_frames):
                mask[:, :, i, :, :] = torch.clamp(transition_values[i], min=0, max=1)
        else:
            # If gradual is False, use start_denoise for the first frame and end_denoise for the others
            mask[:, :, 0, :, :] = start_denoise  # First frame gets the start_denoise value
            mask[:, :, 1:, :, :] = end_denoise  # All subsequent frames get the end_denoise value

        # Assign the new mask
        s["noise_mask"] = mask

        print(s["noise_mask"].shape)
        return (s,)