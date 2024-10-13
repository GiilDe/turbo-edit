import os
from diffusers import AutoPipelineForImage2Image
from diffusers import DDPMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    retrieve_timesteps,
    retrieve_latents,
)
import torch
from PIL import Image
import jsonc as json
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionXLImg2ImgPipeline,
)
from functools import partial
from utils import SAMPLING_DEVICE, get_ddpm_inversion_scheduler, create_xts, device
from config import get_config
import argparse


VAE_SAMPLE = "argmax"  # "argmax" or "sample"
RESIZE_TYPE = None  # Image.LANCZOS


def encode_image(image, pipe, generator):
    pipe_dtype = pipe.dtype
    image = pipe.image_processor.preprocess(image)
    image = image.to(device=device, dtype=pipe.dtype)

    if pipe.vae.config.force_upcast:
        image = image.float()
        pipe.vae.to(dtype=torch.float32)

    init_latents = retrieve_latents(
        pipe.vae.encode(image), generator=generator, sample_mode=VAE_SAMPLE
    )

    if pipe.vae.config.force_upcast:
        pipe.vae.to(pipe_dtype)

    init_latents = init_latents.to(pipe_dtype)
    init_latents = pipe.vae.config.scaling_factor * init_latents

    return init_latents


def set_pipeline(pipeline: StableDiffusionXLImg2ImgPipeline, num_timesteps, generator):
    if num_timesteps == 3:
        config_from_file = "run_configs/noise_shift_steps_3.yaml"
    elif num_timesteps == 4:
        config_from_file = "run_configs/noise_shift_steps_4.yaml"
    else:
        raise ValueError("num_timesteps must be 3 or 4")

    config = get_config(config_from_file)
    if config.timesteps is None:
        denoising_start = config.step_start / config.num_steps_inversion
        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler, config.num_steps_inversion, device, None
        )
        timesteps, num_inference_steps = pipeline.get_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            denoising_start=denoising_start,
            strength=0,
        )
        timesteps = timesteps.type(torch.int64)
        pipeline.__call__ = partial(
            pipeline.__call__,
            num_inference_steps=config.num_steps_inversion,
            guidance_scale=0,
            generator=generator,
            denoising_start=denoising_start,
            strength=0,
        )
        pipeline.scheduler.set_timesteps(
            timesteps=timesteps.cpu(),
        )
    else:
        timesteps = torch.tensor(config.timesteps, dtype=torch.int64)
        pipeline.__call__ = partial(
            pipeline.__call__,
            timesteps=timesteps,
            guidance_scale=0,
            denoising_start=0,
            strength=1,
        )
        pipeline.scheduler.set_timesteps(
            timesteps=config.timesteps,  # device=pipeline.device
        )
    timesteps = [torch.tensor(t) for t in timesteps.tolist()]
    return timesteps, config


def run(
    image_path,
    src_prompt,
    tgt_prompt,
    seed,
    w1,
    num_timesteps,
    pipeline: StableDiffusionXLImg2ImgPipeline,
):

    generator = torch.Generator(device=SAMPLING_DEVICE).manual_seed(seed)

    timesteps, config = set_pipeline(pipeline, num_timesteps, generator)

    x_0_image = Image.open(image_path).convert("RGB").resize((512, 512), RESIZE_TYPE)
    x_0 = encode_image(x_0_image, pipeline, generator)
    x_ts = create_xts(
        config.noise_shift_delta,
        config.noise_timesteps,
        generator,
        pipeline.scheduler,
        timesteps,
        x_0,
    )
    x_ts = [xt.to(dtype=x_0.dtype) for xt in x_ts]
    latents = [x_ts[0]]
    pipeline.scheduler = get_ddpm_inversion_scheduler(
        pipeline.scheduler,
        config,
        timesteps,
        latents,
        x_ts,
    )
    pipeline.scheduler.w1 = w1

    latent = latents[0].expand(3, -1, -1, -1)
    prompt = [src_prompt, src_prompt, tgt_prompt]
    image = pipeline.__call__(image=latent, prompt=prompt).images
    return image[2]


def load_pipe(fp16, cache_dir):
    kwargs = (
        {
            "torch_dtype": torch.float16,
            "variant": "fp16",
        }
        if fp16
        else {}
    )
    pipeline: StableDiffusionXLImg2ImgPipeline = (
        AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            safety_checker=None,
            cache_dir=cache_dir,
            **kwargs,
        )
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDPMScheduler.from_pretrained(  # type: ignore
        "stabilityai/sdxl-turbo",
        subfolder="scheduler",
    )

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default="")
    parser.set_defaults(fp16=False)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--w", type=float, default=1.5)
    parser.add_argument("--timesteps", type=int, default=4)  # 3 or 4
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    img_paths_to_prompts = json.load(open(args.prompts_file, "r"))
    eval_dataset_folder = "/".join(args.prompts_file.split("/")[:-1]) + "/"
    img_paths = [
        f"{eval_dataset_folder}/{img_name}" for img_name in img_paths_to_prompts.keys()
    ]

    pipeline = load_pipe(args.fp16, args.cache_dir)

    for i, img_path in enumerate(img_paths):
        img_name = img_path.split("/")[-1]
        prompt = img_paths_to_prompts[img_name]["src_prompt"]
        edit_prompts = img_paths_to_prompts[img_name]["tgt_prompt"]

        res = run(
            img_path,
            prompt,
            edit_prompts[0],
            args.seed,
            args.w,
            args.timesteps,
            pipeline=pipeline,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        res.save(f"{args.output_dir}/output_{i}.png")
