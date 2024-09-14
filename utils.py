from typing import Optional, Union
import torch
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

SAMPLING_DEVICE = "cpu"  # "cuda"

device = "cuda" if torch.cuda.is_available() else "cpu"


def deterministic_ddpm_step(
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    scheduler,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep

    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in [
        "learned",
        "learned_range",
    ]:
        model_output, predicted_variance = torch.split(
            model_output, sample.shape[1], dim=1
        )
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    return pred_prev_sample


def normalize(
    z_t,
    i,
    max_norm_zs,
):
    max_norm = max_norm_zs[i]
    if max_norm < 0:
        return z_t, 1

    norm = torch.norm(z_t)
    if norm < max_norm:
        return z_t, 1

    coeff = max_norm / norm
    z_t = z_t * coeff
    return z_t, coeff


def step_save_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
):
    timestep_index = self._timesteps.index(timestep)
    next_timestep_index = timestep_index + 1
    u_hat_t = deterministic_ddpm_step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        scheduler=self,
    )

    x_t_minus_1 = self.x_ts[next_timestep_index]
    self.x_ts_c_predicted.append(u_hat_t)

    z_t = x_t_minus_1 - u_hat_t
    self.latents.append(z_t)

    z_t, _ = normalize(z_t, timestep_index, self._config.max_norm_zs)

    x_t_minus_1_predicted = u_hat_t + z_t

    if not return_dict:
        return (x_t_minus_1_predicted,)

    return DDIMSchedulerOutput(prev_sample=x_t_minus_1, pred_original_sample=None)


def step_use_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
):
    timestep_index = self._timesteps.index(timestep)
    next_timestep_index = timestep_index + 1
    z_t = self.latents[next_timestep_index]  # + 1 because latents[0] is X_T

    _, normalize_coefficient = normalize(
        z_t,
        timestep_index,
        self._config.max_norm_zs,
    )

    x_t_hat_c_hat = deterministic_ddpm_step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        scheduler=self,
    )

    x_t_minus_1_exact = self.x_ts[next_timestep_index]
    x_t_minus_1_exact = x_t_minus_1_exact.expand_as(x_t_hat_c_hat)

    x_t_c_predicted: torch.Tensor = self.x_ts_c_predicted[next_timestep_index]

    x_t_c = x_t_c_predicted[0].expand_as(x_t_hat_c_hat)

    edit_prompts_num = model_output.size(0) // 2
    x_t_hat_c_indices = (
        0,
        edit_prompts_num,
    )
    edit_images_indices = (
        edit_prompts_num,
        (model_output.size(0)),
    )
    x_t_hat_c = torch.zeros_like(x_t_hat_c_hat)
    x_t_hat_c[edit_images_indices[0] : edit_images_indices[1]] = x_t_hat_c_hat[
        x_t_hat_c_indices[0] : x_t_hat_c_indices[1]
    ]

    cross_prompt_term = x_t_hat_c_hat - x_t_hat_c
    cross_trajectory_term = x_t_hat_c - normalize_coefficient * x_t_c
    x_t_minus_1_hat = (
        normalize_coefficient * x_t_minus_1_exact
        + cross_trajectory_term
        + self.w1 * cross_prompt_term
    )

    x_t_minus_1_hat[x_t_hat_c_indices[0] : x_t_hat_c_indices[1]] = x_t_minus_1_hat[
        edit_images_indices[0] : edit_images_indices[1]
    ]  # update x_t_hat_c to be x_t_hat_c_hat

    if not return_dict:
        return (x_t_minus_1_hat,)

    return DDIMSchedulerOutput(
        prev_sample=x_t_minus_1_hat,
        pred_original_sample=None,
    )


def get_ddpm_inversion_scheduler(
    scheduler,
    config,
    timesteps,
    latents,
    x_ts,
):
    def step(
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ):
        # predict and save x_t_c
        res_inv = step_save_latents(
            scheduler,
            model_output[:1, :, :, :],
            timestep,
            sample[:1, :, :, :],
            return_dict,
        )

        res_inf = step_use_latents(
            scheduler,
            model_output[1:, :, :, :],
            timestep,
            sample[1:, :, :, :],
            return_dict,
        )
        res = (torch.cat((res_inv[0], res_inf[0]), dim=0),)
        return res

    scheduler._timesteps = timesteps
    scheduler._config = config
    scheduler.latents = latents
    scheduler.x_ts = x_ts
    scheduler.x_ts_c_predicted = [None]
    scheduler.step = step
    return scheduler


def create_xts(
    noise_shift_delta,
    noise_timesteps,
    generator,
    scheduler,
    timesteps,
    x_0,
):
    if noise_timesteps is None:
        noising_delta = noise_shift_delta * (timesteps[0] - timesteps[1])
        noise_timesteps = [timestep - int(noising_delta) for timestep in timesteps]

    first_x_0_idx = len(noise_timesteps)
    for i in range(len(noise_timesteps)):
        if noise_timesteps[i] <= 0:
            first_x_0_idx = i
            break

    noise_timesteps = noise_timesteps[:first_x_0_idx]

    x_0_expanded = x_0.expand(len(noise_timesteps), -1, -1, -1)
    noise = torch.randn(
        x_0_expanded.size(), generator=generator, device=SAMPLING_DEVICE
    ).to(x_0.device)

    x_ts = scheduler.add_noise(
        x_0_expanded,
        noise,
        torch.IntTensor(noise_timesteps),
    )
    x_ts = [t.unsqueeze(dim=0) for t in list(x_ts)]
    x_ts += [x_0] * (len(timesteps) - first_x_0_idx)
    x_ts += [x_0]
    return x_ts
