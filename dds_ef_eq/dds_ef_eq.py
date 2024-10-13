from diffusers.schedulers import (
    DDPMScheduler,
)

from typing import Tuple, Union, Optional, List
import argparse
import jsonc as json

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
import os

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

device = torch.device("cuda:0")

torch.set_grad_enabled(False)


def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer(
        [text],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()


@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode(latent: T, pipe: StableDiffusionPipeline, im_cat: TN = None):
    image = pipeline.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)


def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:

    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class DDSLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int] = None, generator=None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device,
                dtype=torch.long,
            )
        if eps is None:
            eps = torch.randn(z.size(), generator=generator, device="cpu").to(z.device)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(
        self,
        z_t: T,
        timestep: T,
        text_embeddings: T,
        alpha_t: T,
        sigma_t: T,
        get_raw=False,
        guidance_scale_source=7.5,
        guidance_scale_target=7.5,
    ):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(
            -1, *text_embeddings.shape[2:]
        )
        e_t = self.unet(latent_input, timestep, embedd).sample
        if self.prediction_type == "v_prediction":
            e_t = (
                torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            )
        e_t_uncond, e_t = e_t.chunk(2)
        if get_raw:
            return e_t_uncond, e_t

        guidance_scale = torch.tensor(
            [guidance_scale_source, guidance_scale_target]
        ).to(e_t.device)

        e_t_guidance = (
            e_t_uncond + torch.mul(guidance_scale, (e_t - e_t_uncond).T).T
        )  # x.permute(*torch.arange(x.ndim - 1, -1, -1))

        assert torch.isfinite(e_t_guidance).all()
        if get_raw:
            return e_t_guidance
        pred_z0 = (z_t - sigma_t * e_t_guidance) / alpha_t
        return e_t_guidance, pred_z0

    def get_dds_loss(
        self,
        z_source: T,
        z_target: T,
        text_emb_source: T,
        text_emb_target: T,
        eps=None,
        reduction="mean",
        symmetric: bool = False,
        calibration_grad=None,
        timestep: Optional[int] = None,
        guidance_scale_source=7.5,
        guidance_scale_target=7.5,
        raw_log=False,
        generator=None,
    ) -> TS:
        with torch.inference_mode():
            z_t_source, eps, timestep, alpha_t, sigma_t = self.noise_input(
                z_source, eps, timestep, generator
            )
            z_t_target, _, _, _, _ = self.noise_input(
                z_target, eps, timestep, generator
            )
            eps_pred, _ = self.get_eps_prediction(
                torch.cat((z_t_source, z_t_target)),
                torch.cat((timestep, timestep)),
                torch.cat((text_emb_source, text_emb_target)),
                torch.cat((alpha_t, alpha_t)),
                torch.cat((sigma_t, sigma_t)),
                guidance_scale_source=guidance_scale_source,
                guidance_scale_target=guidance_scale_target,
            )
            eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            grad = eps_pred_target - eps_pred_source
        return grad

    def __init__(
        self,
        device,
        pipe: StableDiffusionPipeline,
        dtype=torch.float32,
        t_min=50,
        t_max=950,
    ):
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(
            device, dtype, pipe.unet, pipe.scheduler
        )
        self.prediction_type = pipe.scheduler.prediction_type


model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
).to(device)
pipeline.scheduler = DDPMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
)

print(f"using model: {model_id}")


def image_optimization(
    pipe: StableDiffusionPipeline,
    image: np.ndarray,
    text_source: str,
    text_target: str,
    use_dds=True,
    guidance_scale_source=7.5,
    guidance_scale_target=7.5,
    lr=2000,
    num_iters=10,
    folder_name=None,
    file_name=None,
    linear_timestep_annealing=True,
    t_min=0,
    t_max=670,
    lr_ef=True,
    generator=None,
    output_dir="output",
) -> None:
    dds_loss = DDSLoss(device, pipe, t_min=t_min, t_max=t_max)

    pipe.scheduler: DDPMScheduler = pipe.scheduler  # type: ignore

    image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = image_source.unsqueeze(0).to(device)

    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)["latent_dist"].mean * 0.18215
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text = get_text_embeddings(pipeline, text_source)
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    z_target = z_source.clone()

    if linear_timestep_annealing:
        timesteps = (
            torch.linspace(
                start=min(dds_loss.t_max, 1000) - 1, end=dds_loss.t_min, steps=num_iters
            )
            .int()
            .to(z_target.device)
        )
        pipe.scheduler.set_timesteps(timesteps=timesteps.cpu())
    else:
        timesteps = [None] * num_iters  # will use DDS random timesteps

    for i, timestep in zip(range(num_iters), timesteps):
        if use_dds:
            diff = dds_loss.get_dds_loss(
                z_source,
                z_target,
                embedding_source,
                embedding_target,
                guidance_scale_source=guidance_scale_source,
                guidance_scale_target=guidance_scale_target,
                timestep=timestep.unsqueeze(0) if timestep is not None else timestep,
                generator=generator,
            )
        else:
            loss, log_loss = dds_loss.get_sds_loss(
                z_target, embedding_target, guidance_scale=guidance_scale_source
            )

        if lr_ef:
            alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep]
            prev_t = pipe.scheduler.previous_timestep(timestep.to("cpu"))
            alpha_prod_t_prev = (
                pipe.scheduler.alphas_cumprod[prev_t]
                if prev_t >= 0
                else pipe.scheduler.one
            )
            sqrt_alpha_prod = alpha_prod_t**0.5
            sqrt_one_minus_alpha_prod = (
                1 - pipe.scheduler.alphas_cumprod[timestep]
            ) ** 0.5
            current_alpha_t = (
                alpha_prod_t / alpha_prod_t_prev
            )  # NOTE: current_alpha_t is different than pipe.scheduler.alphas[timestep] since we are not using 1000 inference timesteps
            lr_t = (1 - current_alpha_t) / (sqrt_alpha_prod * sqrt_one_minus_alpha_prod)
            lr = lr_t

        z_target = z_target - lr * diff

    out = decode(z_target, pipeline)
    path = f"{output_dir}/{folder_name}"
    os.makedirs(
        path,
        exist_ok=True,
    )

    out.save(f"{path}/{file_name}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2000)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--t_min", type=int, default=99)
    parser.add_argument("--t_max", type=int, default=899)
    parser.add_argument(
        "--prompts_file", type=str, default="example_images/dataset/dataset.json"
    )
    parser.add_argument("--guidance_scale_source", type=float, default=3.5)

    parser.add_argument("--guidance_scale_target", type=float, default=15)

    parser.set_defaults(linear_timestep_annealing=False)
    parser.add_argument("--linear_timestep_annealing", action="store_true")

    parser.set_defaults(lr_ef=False)
    parser.add_argument("--lr_ef", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    eval_dataset_folder = "/".join(str.split(args.prompts_file, "/")[:-1])

    img_paths_to_prompts = json.load(open(args.prompts_file, "r"))
    img_paths = [
        f"{eval_dataset_folder}/{img_name}" for img_name in img_paths_to_prompts.keys()
    ]

    for i, img_path in enumerate(img_paths):
        img_name = img_path.split("/")[-1]
        prompt = img_paths_to_prompts[img_name]["src_prompt"]
        edit_prompts = img_paths_to_prompts[img_name]["tgt_prompt"]

        image = load_512(img_path)

        for j, edit_prompt in enumerate(edit_prompts):
            image_optimization(
                pipeline,
                image,
                prompt,
                edit_prompt,
                use_dds=True,
                guidance_scale_source=args.guidance_scale_source,
                guidance_scale_target=args.guidance_scale_target,
                lr=args.lr,
                num_iters=args.num_iters,
                folder_name=f"{i+1}",
                file_name=f"{j+1}",
                linear_timestep_annealing=args.linear_timestep_annealing,
                t_min=args.t_min,
                t_max=args.t_max,
                lr_ef=args.lr_ef,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output_dir=args.output_dir,
            )
