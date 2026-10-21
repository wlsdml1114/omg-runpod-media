"""Save the model's predicted clean latent (x0) at every sampling step.

SamplerCustomAdvancedX0Steps mirrors SamplerCustomAdvanced at the pinned ComfyUI
commit, but keeps every step's x0 instead of only the last one, applies the same
unpack/process_latent_out post-processing, and torch.saves each to save_dir.
LoadX0Step loads one of those files back as a LATENT for VAE decoding.
"""
import os
import torch
import comfy.sample
import comfy.utils
import comfy.nested_tensor
import comfy.model_management
import latent_preview


def _post(x0, samples, model):
    if samples.is_nested and not x0.is_nested:
        shapes = [x.shape for x in samples.unbind()]
        x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, shapes))
    return model.process_latent_out(x0.cpu())


def _to_plain(t):
    if getattr(t, "is_nested", False):
        return {"nested": [x.contiguous() for x in t.unbind()]}
    return {"tensor": t.contiguous()}


class SamplerCustomAdvancedX0Steps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",),
                             "sigmas": ("SIGMAS",), "latent_image": ("LATENT",),
                             "save_dir": ("STRING", {"default": "/workspace/x0"})}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image, save_dir):
        latent = latent_image.copy()
        li = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"],
                                                    latent.get("downscale_ratio_spacial", None),
                                                    latent.get("downscale_ratio_temporal", None))
        latent["samples"] = li
        noise_mask = latent.get("noise_mask")
        steps = sigmas.shape[-1] - 1
        x0_output = {}
        base_cb = latent_preview.prepare_callback(guider.model_patcher, steps, x0_output)
        raw = []

        def cb(step, x0, x, total_steps):
            if getattr(x0, "is_nested", False):
                snap = comfy.nested_tensor.NestedTensor(tuple(t.detach().to("cpu", copy=True) for t in x0.unbind()))
            else:
                snap = x0.detach().to("cpu", copy=True)
            raw.append((step, snap, float(sigmas[step])))
            base_cb(step, x0, x, total_steps)

        samples = guider.sample(noise.generate_noise(latent), li, sampler, sigmas, denoise_mask=noise_mask,
                                callback=cb, disable_pbar=True, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())
        model = guider.model_patcher.model
        os.makedirs(save_dir, exist_ok=True)
        for step, x0, sigma in raw:
            torch.save({"step": step, "total_steps": steps, "sigma": sigma, "x0": _to_plain(_post(x0, samples, model))},
                       os.path.join(save_dir, f"x0_step{step + 1:02d}_of{steps:02d}.pt"))
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        den = latent.copy()
        den.pop("downscale_ratio_spacial", None)
        den.pop("downscale_ratio_temporal", None)
        den["samples"] = _post(x0_output["x0"], samples, model) if "x0" in x0_output else samples
        return (out, den)


class LoadX0Step:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent"

    @classmethod
    def IS_CHANGED(cls, path):
        return path

    def load(self, path):
        d = torch.load(path, map_location="cpu")["x0"]
        if "nested" in d:
            return ({"samples": comfy.nested_tensor.NestedTensor(tuple(d["nested"]))},)
        return ({"samples": d["tensor"]},)


NODE_CLASS_MAPPINGS = {"SamplerCustomAdvancedX0Steps": SamplerCustomAdvancedX0Steps, "LoadX0Step": LoadX0Step}
