import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/VisualStylePrompting_Controlnet-hf')
os.chdir('/content/VisualStylePrompting_Controlnet-hf')

import torch
from pipelines.inverted_ve_pipeline import STYLE_DESCRIPTION_DICT, create_image_grid
import os, json, cv2
import numpy as np
from PIL import Image

from pipelines.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from diffusers import ControlNetModel, AutoencoderKL
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from random import randint
from utils import init_latent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

def memory_efficient(model):
    try:
        model.to(device)
    except Exception as e:
        print("Error moving model to device:", e)

    try:
        model.enable_model_cpu_offload()
    except AttributeError:
        print("enable_model_cpu_offload is not supported.")
    try:
        model.enable_vae_slicing()
    except AttributeError:
        print("enable_vae_slicing is not supported.")

    if device == 'cuda':
        try:
            model.enable_xformers_memory_efficient_attention()
        except AttributeError:
            print("enable_xformers_memory_efficient_attention is not supported.")

# controlnet_scale, canny thres 1, 2 (2 > 1, 2:1, 3:1)

def parse_config(config):
    with open(config, 'r') as f:
        config = json.load(f)
    return config

def get_depth_map(image, depth_estimator, feature_extractor):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def get_depth_edge_array(depth_img_path, depth_estimator, feature_extractor):
    depth_image_tmp = Image.open(depth_img_path)
    # get depth map
    depth_map = get_depth_map(depth_image_tmp, depth_estimator, feature_extractor)
    return depth_map

def controlnet_fn(image_path, depth_image_path, style_name, content_text, output_number, controlnet_scale=0.5, diffusion_step=50, model_controlnet=None, depth_estimator=None, feature_extractor=None):
    """

    :param style_name: 어떤 json 파일 부를거냐 ?
    :param content_text: 어떤 콘텐츠로 변화를 원하니 ?
    :param output_number: 몇개 생성할거니 ?
    :return:
    """
    config_path = './config/{}.json'.format(style_name)
    config = parse_config(config_path)

    inf_object = content_text
    inf_seeds = [randint(0, 10**10) for _ in range(int(output_number))]
    # inf_seeds = [i for i in range(int(output_number))]

    activate_layer_indices_list = config['inference_info']['activate_layer_indices_list']
    activate_step_indices_list = config['inference_info']['activate_step_indices_list']
    ref_seed = config['reference_info']['ref_seeds'][0]

    attn_map_save_steps = config['inference_info']['attn_map_save_steps']
    guidance_scale = config['guidance_scale']
    use_inf_negative_prompt = config['inference_info']['use_negative_prompt']

    style_name = config["style_name_list"][0]

    ref_object = config["reference_info"]["ref_object_list"][0]
    ref_with_style_description = config['reference_info']['with_style_description']
    inf_with_style_description = config['inference_info']['with_style_description']

    use_shared_attention = config['inference_info']['use_shared_attention']
    adain_queries = config['inference_info']['adain_queries']
    adain_keys = config['inference_info']['adain_keys']
    adain_values = config['inference_info']['adain_values']

    use_advanced_sampling = config['inference_info']['use_advanced_sampling']

    #get canny edge array
    depth_image = get_depth_edge_array(depth_image_path, depth_estimator, feature_extractor)

    style_description_pos, style_description_neg = STYLE_DESCRIPTION_DICT[style_name][0], \
                                                   STYLE_DESCRIPTION_DICT[style_name][1]

    # Inference
    with torch.inference_mode():
        grid = None
        if ref_with_style_description:
            ref_prompt = style_description_pos.replace("{object}", ref_object)
        else:
            ref_prompt = ref_object

        if inf_with_style_description:
            inf_prompt = style_description_pos.replace("{object}", inf_object)
        else:
            inf_prompt = inf_object

        for activate_layer_indices in activate_layer_indices_list:

            for activate_step_indices in activate_step_indices_list:

                str_activate_layer, str_activate_step = model_controlnet.activate_layer(
                    activate_layer_indices=activate_layer_indices,
                    attn_map_save_steps=attn_map_save_steps,
                    activate_step_indices=activate_step_indices,
                    use_shared_attention=use_shared_attention,
                    adain_queries=adain_queries,
                    adain_keys=adain_keys,
                    adain_values=adain_values,
                )

                # ref_latent = model_controlnet.get_init_latent(ref_seed, precomputed_path=None)
                ref_latent = init_latent(model_controlnet, device_name=device, dtype=torch_dtype, seed=ref_seed)
                latents = [ref_latent]

                for inf_seed in inf_seeds:
                    # latents.append(model_controlnet.get_init_latent(inf_seed, precomputed_path=None))
                    inf_latent = init_latent(model_controlnet, device_name=device, dtype=torch_dtype, seed=inf_seed)
                    latents.append(inf_latent)


                latents = torch.cat(latents, dim=0)
                latents.to(device)

                images = model_controlnet.generated_ve_inference(
                    prompt=ref_prompt,
                    negative_prompt=style_description_neg,
                    guidance_scale=guidance_scale,
                    num_inference_steps=diffusion_step,
                    controlnet_conditioning_scale=controlnet_scale,
                    latents=latents,
                    num_images_per_prompt=len(inf_seeds) + 1,
                    target_prompt=inf_prompt,
                    image=depth_image,
                    use_inf_negative_prompt=use_inf_negative_prompt,
                    use_advanced_sampling=use_advanced_sampling
                )[0][1:]

                n_row = 1
                n_col = len(inf_seeds)  # 원본추가하려면 + 1

                # make grid
                grid = create_image_grid(images, n_row, n_col)

        torch.cuda.empty_cache()
        return grid

class Predictor(BasePredictor):
    def setup(self) -> None:
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
        self.model_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch_dtype)
        print("vae")
        memory_efficient(vae)
        print("control")
        memory_efficient(controlnet)
        print("ControlNet-SDXL")
        memory_efficient(self.model_controlnet)
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    def predict(
        self,
        style_image: Path = Input(description="Style Image"),
        depth_image: Path = Input(description="Depth Image"),
        style_name: str = Input(default='fire'),
        prompt: str = Input(default=''),
        controlnet_scale: float = Input(default=0.5),
        diffusion_steps: int = Input(default=50),
    ) -> Path:
        output_image = controlnet_fn(style_image, depth_image, style_name, prompt, 1, controlnet_scale, diffusion_steps, model_controlnet=self.model_controlnet, depth_estimator=self.depth_estimator, feature_extractor=self.feature_extractor)
        output_image.save('/content/output_image.png')
        return Path('/content/output_image.png')