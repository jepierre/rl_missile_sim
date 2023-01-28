import time
from .envs.missile_sim import MissileSim
from .guidance import pronav_policy


def main():
    env = MissileSim({"render_mode": "human"})
    obs, info = env.reset()
    done = False

    for _ in range(500):
        action = pronav_policy(obs)
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(0.03)  # Slow down for visibility
        if done:
            print("Episode finished. Resetting environment.")
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()