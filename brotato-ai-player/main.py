import keyboard
import time

from stable_baselines3 import PPO

from train_ppo import MODEL_FILE
from brotato_env import BrotatoEnv

def play():
    env = BrotatoEnv()
    model = PPO.load(MODEL_FILE)

    obs, info = env.reset()
    while not keyboard.is_pressed('q'):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            time.sleep(3)
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    play()
