import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardRewardLoggerCallback(BaseCallback):
    """
    Logs mean episode info scalars to TensorBoard, averaged across all environments.
    """

    def __init__(self, log_freq: int = 20, verbose: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.log_freq = log_freq
        self.episode_counts = 0
        self.episode_info_accum = None

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self.episode_info_accum = [{} for _ in range(n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        n_envs = len(infos)

        for env_idx in range(n_envs):
            info = infos[env_idx]
            for k, v in info.items():
                if np.isscalar(v):
                    if k not in self.episode_info_accum[env_idx]:
                        self.episode_info_accum[env_idx][k] = []
                    self.episode_info_accum[env_idx][k].append(v)

        for env_idx, done in enumerate(dones):
            if done:
                self.episode_counts += 1
                for k, v_list in self.episode_info_accum[env_idx].items():
                    if v_list:
                        mean_val = float(np.mean(v_list))
                        self.logger.record(f"custom/mean_{k}", mean_val)
                if self.verbose > 0 and self.episode_counts % self.log_freq == 0:
                    means_str = ", ".join(
                        f"{k}={np.mean(v_list):.3f}"
                        for k, v_list in self.episode_info_accum[env_idx].items()
                    )
                    print(
                        f"Episode {self.episode_counts} (env {env_idx}): mean info: {means_str}"
                    )
                self.episode_info_accum[env_idx] = {}

        return True
