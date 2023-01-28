import argparse
import pygame
from stable_baselines3.common.env_util import make_vec_env

from .callbacks import TensorboardRewardLoggerCallback
from .envs.missile_sim import MissileSim
from .evaluation import evaluate
from .guidance import (
    lqr_gain,
    lqr_guidance,
    optimal_guidance_policy,
    pronav_policy,
)
from .training import train_ppo
from .utils import (
    NormalizedObservation,
    RescaleAction,
    get_git_hash,
    init_seed,
    make_wrapped_env,
)

# Alias for backwards compatibility
train = train_ppo

__all__ = [
    "lqr_gain",
    "lqr_guidance",
    "pronav_policy",
    "optimal_guidance_policy",
    "TensorboardRewardLoggerCallback",
    "evaluate",
    "train_ppo",
    "train",
    "init_seed",
    "make_wrapped_env",
    "main",
]


def build_eval_env(policy: str, no_render: bool, seed: int):
    """
    Build environment for evaluation based on selected policy type.
    """
    render_mode = None if no_render else "human"
    if policy == "agent":
        return make_vec_env(
            lambda: make_wrapped_env(render_mode=render_mode),
            n_envs=1,
            seed=seed,
        )
    return MissileSim({"render_mode": render_mode})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate guidance policies on MissileSim."
    )
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train", help="train or eval"
    )
    parser.add_argument(
        "--agent_path", type=str, default=None, help="Path to RL agent for eval"
    )
    parser.add_argument(
        "--policy",
        choices=["random", "agent", "pronav", "lqr", "optimal"],
        default="random",
        help="Policy for eval",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes for eval")
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering in eval"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_100_000, help="Training timesteps"
    )
    parser.add_argument(
        "--n_envs", type=int, default=20, help="Number of parallel envs for training"
    )
    parser.add_argument(
        "--vec_norm_path",
        type=str,
        default=None,
        help="Path for normalize reward and observation files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = 42
    use_cuda = False

    init_seed(seed)
    try:
        pygame.init()
        pygame.display.set_caption("Missile Simulation with Logging")
    except Exception:
        pass

    if args.mode == "train":
        train_ppo(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=seed,
            use_cuda=use_cuda,
            vec_norm_path=args.vec_norm_path,
        )
    elif args.mode == "eval":
        env = build_eval_env(args.policy, args.no_render, seed)
        evaluate(
            env,
            policy_type=args.policy,
            agent_path=args.agent_path,
            episodes=args.episodes,
            render=not args.no_render,
            vectorize=(args.policy == "agent"),
            use_cuda=use_cuda,
        )


if __name__ == "__main__":
    main()
