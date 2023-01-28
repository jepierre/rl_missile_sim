import datetime
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from .callbacks import TensorboardRewardLoggerCallback
from .envs.missile_sim import MissileSim
from .utils import NormalizedObservation, RescaleAction, get_git_hash


def train_ppo(
    timesteps: int = 1_100_000,
    n_envs: int = 20,
    seed: int = 42,
    use_cuda: bool = False,
    log_dir_base: str = "logs",
    vec_norm_path: str | None = None,
):
    """
    Train a PPO guidance policy on the wrapped MissileSim environment.
    """
    os.makedirs(log_dir_base, exist_ok=True)
    device = "cuda" if use_cuda else "cpu"
    dir_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_hash = get_git_hash(short=True)
    log_dir = os.path.join(log_dir_base, f"ppo_missile_{dir_timestamp}_{branch_hash}")
    os.makedirs(log_dir, exist_ok=True)
    final_model_path = os.path.join(log_dir, "final_model.zip")

    def env_factory():
        return NormalizedObservation(
            RescaleAction(MissileSim({"render_mode": None}), -1, 1), 0, 1
        )

    env = make_vec_env(
        env_factory,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "model_checkpoints"),
        name_prefix="ppo",
        verbose=1,
    )

    eval_env = make_vec_env(
        env_factory,
        n_envs=1,
        monitor_dir=log_dir,
        seed=seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    tensorboard_logger = TensorboardRewardLoggerCallback(log_freq=20, verbose=1)

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )

    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=[tensorboard_logger, checkpoint_callback, eval_callback],
    )

    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    return model, final_model_path
