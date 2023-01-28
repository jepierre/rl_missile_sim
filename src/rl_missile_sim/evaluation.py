import numpy as np
from stable_baselines3 import PPO

from .guidance import lqr_guidance, optimal_guidance_policy, pronav_policy


def evaluate(
    env,
    policy_type: str = "random",
    agent_path: str | None = None,
    episodes: int = 5,
    render: bool = True,
    vectorize: bool = False,
    use_cuda: bool = False,
):
    """
    Evaluate a guidance policy or trained RL agent on the environment.

    Parameters:
        env: Gymnasium or vectorized SB3 environment instance.
        policy_type: One of 'random', 'agent', 'pronav', 'lqr', 'optimal'.
        agent_path: Filesystem path to trained SB3 model (.zip).
        episodes: Number of evaluation episodes to run.
        render: Whether to call env.render() during evaluation.
        vectorize: True if env is a VecEnv (reset/step return shapes differ).
    """
    if policy_type == "agent":
        device = "cuda" if use_cuda else "cpu"
        if agent_path:
            model = PPO.load(agent_path, device=device)
        else:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                device=device,
            )
    else:
        model = None

    for ep in range(episodes):
        if vectorize:
            obs = env.reset()
        else:
            obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if policy_type == "random":
                action = env.action_space.sample()
            elif policy_type == "agent":
                action, _ = model.predict(obs, deterministic=True)
            elif policy_type == "pronav":
                action = pronav_policy(obs)
            elif policy_type == "lqr":
                missile_pos = obs[:2]
                missile_vel = obs[3:5]
                target_pos = obs[6:8]
                target_vel = obs[9:11]
                action = lqr_guidance(missile_pos, missile_vel, target_pos, target_vel)
            elif policy_type == "optimal":
                action = optimal_guidance_policy(obs)
            else:
                raise ValueError(f"Unknown policy_type: {policy_type}")

            if vectorize:
                obs, reward, done, info = env.step(action)
                total_reward += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            else:
                obs, reward, done, _, info = env.step(action)
                total_reward += float(reward)

            if render:
                env.render()

        print(f"Episode {ep+1} reward: {total_reward}")
        print(f"episode {ep+1} info: {info}")
    env.close()
