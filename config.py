from ml_collections import config_dict
import yaml


def get_num_steps_actual(cfg):
    return (
        cfg.num_steps_inversion - cfg.step_start
        if cfg.timesteps is None
        else len(cfg.timesteps)
    )


def get_config(config_path):
    if config_path and config_path != "":
        with open(config_path, "r") as f:
            cfg = config_dict.ConfigDict(yaml.safe_load(f))

        num_steps_actual = get_num_steps_actual(cfg)

    with cfg.ignore_type():
        if isinstance(cfg.max_norm_zs, (int, float)):
            cfg.max_norm_zs = [cfg.max_norm_zs] * num_steps_actual

    assert (
        len(cfg.max_norm_zs) == num_steps_actual
    ), f"len(cfg.max_norm_zs) ({len(cfg.max_norm_zs)}) != num_steps_actual ({num_steps_actual})"

    assert cfg.noise_timesteps is None or len(cfg.noise_timesteps) == (
        num_steps_actual
    ), f"len(cfg.noise_timesteps) ({len(cfg.noise_timesteps)}) != num_steps_actual ({num_steps_actual})"

    return cfg
