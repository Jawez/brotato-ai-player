import sys
import os
from datetime import datetime
import time

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from brotato_env import BrotatoEnv

MODEL_NAME = "ppo_brotato"
MODEL_DIR = "models"
LOG_DIR = "logs"

MODEL_FILE = os.path.join(MODEL_DIR, MODEL_NAME + '.zip')

BATCH_SIZE = 256
N_STEPS = 2048
N_EPOCHS = 10

ONE_HOUR_STEPS = N_STEPS * 12

TOTAL_TIMESTEPS = ONE_HOUR_STEPS * 6
MODEL_SAVE_FREQ = ONE_HOUR_STEPS


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.paused = False
        self.custom_env = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Wrapping the env with a `Monitor` wrapper
        # Wrapping the env in a DummyVecEnv.
        # Wrapping the env in a VecTransposeImage.
        # model.get_env() -> stable_baselines3.common.vec_env.vec_transpose.VecTransposeImage object
        # model.get_env().venv -> stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object
        # model.get_env().venv.envs[0] -> Monitor<BrotatoEnv instance>  # from stable_baselines3.common.monitor import Monitor
        # model.get_env().venv.envs[0].env -> <BrotatoEnv instance>
        self.custom_env = self.training_env.venv.envs[0].env
        print(f"training start, device: {self.model.device}, lr: {self.model.learning_rate}, env: {self.custom_env}")

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.paused:
            self.custom_env.resume()
            self.paused = False
        print(f"rollout start, pause: {self.paused}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if not self.paused:
            self.custom_env.pause()
            self.paused = True
        print(f"rollout end, pause: {self.paused}")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("training end")

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{MODEL_NAME}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    env = BrotatoEnv()
    # check_env(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    if os.path.exists(MODEL_FILE):
        print(f'load: {MODEL_FILE}')

        learning_rate = 3e-5    # 1.5e-4
        # gamma = 0.9
        # clip_range = 0.2
        custom_objects = {
            'learning_rate': learning_rate,
            # 'gamma': gamma,
            # 'clip_range': clip_range,

            'device': device,
        }
        model = PPO.load(MODEL_FILE, env, custom_objects=custom_objects)
        model.ent_coef = 0.1

        # model = PPO.load(MODEL_FILE, env)
    else:
        print(f'new ppo')

        model = PPO("CnnPolicy",
                    env,

                    learning_rate = 3e-4,   # 1e-4,   #

                    ent_coef = 0.1,    # 0.01, # 0.0,
                    vf_coef = 0.5,
                    gamma = 0.99,
                    gae_lambda = 0.95,

                    clip_range = 0.3,   # 0.2,

                    batch_size = BATCH_SIZE,   # 64,
                    n_steps = N_STEPS,   # 2048
                    n_epochs = N_EPOCHS,  # 10,

                    device = device,
                    verbose = 1,
                    tensorboard_log = LOG_DIR,
                )

    checkpoint_callback = CheckpointCallback(save_freq=MODEL_SAVE_FREQ,
                                             save_path=MODEL_DIR,
                                             name_prefix=MODEL_NAME)
    custom_callback = CustomCallback()

    # Create the callback list
    callback = CallbackList([checkpoint_callback, custom_callback])

    print(f'start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')

    original_stdout = sys.stdout
    with open(log_path, 'a', encoding='utf-8') as f:
        sys.stdout = f
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    sys.stdout = original_stdout

    print(f'end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')

    model.save(MODEL_FILE)

    env.close()

if __name__ == "__main__":
    train()
