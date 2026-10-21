"""Build ComfyUI API graphs for FLUX 3 Image Klein (distilled) text-to-image and reference editing.

The editing graph matches the official `flux3_klein_*_mi2i` template: ReferenceMultiLatent (resize 1 MP),
ModelSamplingFlux with shift log(16), euler/simple, CFG 1. The optional prompt upsampler uses the same
edit system prompt and generation settings as the template (configs/upsampler-edit-system-prompt.txt).
"""
import math
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
UNET = {
    'bf16': 'flux3-image-klein-bf16.safetensors',
    'int8': 'flux3-image-klein-int8-convrot.safetensors',
}
TEXT_ENCODER = 'qwen3vl_4b_bf16.safetensors'
UPSAMPLER = 'qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors'
VAE = 'flux2-vae.safetensors'
SHIFT = math.log(16.0)


def build(prompt, refs=(), width=None, height=None, megapixels=4.0, model='bf16', steps=16,
          seed=42, upsample=False, size_ref=None, prefix='klein'):
    """refs: input image file names (already in ComfyUI/input). size_ref: index of the ref that sets the
    output size (default: last ref, i.e. the background). Without refs, width/height are required."""
    refs = list(refs)
    g = {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': UNET[model], 'weight_dtype': 'default'}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': TEXT_ENCODER, 'type': 'flux3', 'device': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': VAE}},
        'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['clip', 0], 'text': prompt}},
        'neg': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['clip', 0], 'text': ''}},
        'save': {'class_type': 'SaveImage', 'inputs': {'images': ['dec', 0], 'filename_prefix': prefix}},
    }
    if refs:
        si = len(refs) - 1 if size_ref is None else size_ref
        for i, name in enumerate(refs):
            g[f'img{i}'] = {'class_type': 'LoadImage', 'inputs': {'image': name}}
        g['scale'] = {'class_type': 'ImageScaleToTotalPixels', 'inputs': {
            'image': [f'img{si}', 0], 'upscale_method': 'lanczos', 'megapixels': megapixels, 'resolution_steps': 16}}
        g['size'] = {'class_type': 'GetImageSize', 'inputs': {'image': ['scale', 0]}}
        w, h = ['size', 0], ['size', 1]
        ref = {'positive': ['pos', 0], 'negative': ['neg', 0], 'vae': ['vae', 0],
               'resolution': 'resize', 'resolution.megapixels': 1.0}
        for i in range(len(refs)):
            ref[f'images.image{i}'] = ['scale', 0] if i == si else [f'img{i}', 0]
        g['ref'] = {'class_type': 'ReferenceMultiLatent', 'inputs': ref}
        positive, negative = ['ref', 0], ['ref', 1]
        if upsample:
            _add_upsampler(g, prompt, len(refs), seed)
    else:
        if not (width and height):
            raise ValueError('width and height are required without reference images')
        if upsample:
            raise ValueError('this recipe only uses the upsampler for editing')
        w, h, positive, negative = width, height, ['pos', 0], ['neg', 0]
    g['lat'] = {'class_type': 'EmptyFlux2LatentImage', 'inputs': {'width': w, 'height': h, 'batch_size': 1}}
    g['ms'] = {'class_type': 'ModelSamplingFlux', 'inputs': {
        'model': ['unet', 0], 'max_shift': SHIFT, 'base_shift': SHIFT, 'width': w, 'height': h}}
    g['ks'] = {'class_type': 'KSampler', 'inputs': {
        'model': ['ms', 0], 'positive': positive, 'negative': negative, 'latent_image': ['lat', 0],
        'seed': seed, 'steps': steps, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}}
    g['dec'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['ks', 0], 'vae': ['vae', 0]}}
    return g


def _add_upsampler(g, prompt, n_refs, seed):
    system = (CONFIGS / 'upsampler-edit-system-prompt.txt').read_text().rstrip('\n')
    g['upclip'] = {'class_type': 'CLIPLoader', 'inputs': {'clip_name': UPSAMPLER, 'type': 'flux3', 'device': 'default'}}
    g['upimgs'] = {'class_type': 'BatchImagesNode', 'inputs': {f'images.image{i}': [f'img{i}', 0] for i in range(n_refs)}}
    g['upgen'] = {'class_type': 'TextGenerate', 'inputs': {
        'clip': ['upclip', 0], 'image': ['upimgs', 0], 'system_prompt': system,
        'prompt': ('Reference images follow in order (the first image, the second image, ...).\n'
                   f'User request: {prompt.strip()}\nOutput the caption.'),
        'max_length': 3584, 'sampling_mode': 'on', 'sampling_mode.temperature': 1.0, 'sampling_mode.top_k': 1,
        'sampling_mode.top_p': 0.95, 'sampling_mode.min_p': 0.0, 'sampling_mode.repetition_penalty': 1.0,
        'sampling_mode.seed': seed, 'sampling_mode.presence_penalty': 0.0,
        'thinking': False, 'use_default_template': True, 'mtp': 'off'}}
    g['upshow'] = {'class_type': 'PreviewAny', 'inputs': {'source': ['upgen', 0]}}
    g['pos']['inputs']['text'] = ['upgen', 0]
