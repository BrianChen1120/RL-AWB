import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import cv2

from utils.Algorithm import RGB_estimation


class IlluminantEstimationEnv(gym.Env):
    def __init__(self, max_steps, dataset_name='NCC', img_dir=None, training=False, init_action=None):
        super().__init__()

        self.max_steps = max_steps
        self.dataset_name = dataset_name

        if img_dir is None:
            img_dir = f'./dataset/{dataset_name}dataset/img'
        self.img_dir = img_dir

        low = np.array([-0.6, -4], dtype=np.float32)
        high = np.array([0.6, 4], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        history_time = 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10800 * 1 + 10 + 1,), dtype=np.float32)
        self.current_step = 0

        self.history_EvaLum = np.zeros((history_time, 3), dtype=np.float32)
        self.history_action = np.zeros((history_time, 2), dtype=np.float32)
        self.history_WBsRGB = np.zeros((1, 10800), dtype=np.float32)

        self.training = training

        self.init_action = init_action
        if self.init_action is None:
            if dataset_name == 'NCC':
                self.init_action = np.array([0.50, 3.5, 0.045, 0.3, 10, 3, 0, 0.90, 7, 3], dtype=np.float32)
            elif dataset_name == 'LEVI':
                self.init_action = np.array([0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3], dtype=np.float32)
            elif dataset_name == 'Gehler':
                self.init_action = np.array([0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            print(f'Init action is None, default action is {self.init_action}')
        else:
            print(f'Init action is {self.init_action}')

    def reset(self, image_index=None, **kwargs):
        self.now_action = self.init_action.copy()

        if image_index is None:
            raise ValueError("image_index must be specified for inference")
        else:
            self.image_index = int(image_index)

        print(f"Processing image index: {self.image_index}")

        self.init_arr, self.init_arr_rep, self.init_EvaLum, gt, [for_RL, y], feature_vec = RGB_estimation(
            self.image_index, self.init_action, dataset=self.dataset_name
        )

        img_path = f"{self.img_dir}/{self.image_index}.png"
        print(f"Loading image: {img_path}")
        self.img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(self.img, (377, 252), interpolation=cv2.INTER_NEAREST)

        self.history_EvaLum.fill(0)
        self.history_action.fill(0)
        self.history_WBsRGB.fill(0)

        self.history_EvaLum[:-1] = self.history_EvaLum[1:]
        self.history_EvaLum[-1] = self.init_EvaLum

        self.history_action[:-1] = self.history_action[1:]
        self.history_action[-1] = [1.7, 10]

        self.history_WBsRGB[:-1] = self.history_WBsRGB[1:]
        self.history_WBsRGB[-1] = feature_vec

        obs = np.concatenate([
            self.history_WBsRGB.flatten(),
            self.history_action.flatten(),
            np.array([self.current_step])
        ], dtype=np.float32)

        self.current_step = 0

        return obs, {
            "arr": self.init_arr,
            "arr_rep": self.init_arr_rep,
            "EvaLum": self.init_EvaLum,
            "GtLum": gt,
            "for_RL": for_RL
        }

    def step(self, action):
        self.current_step += 1

        self.now_action[1] = np.clip(self.now_action[1] + action[0], a_min=1.0, a_max=2.0)
        self.now_action[4] = np.clip(self.now_action[4] + action[1], a_min=1, a_max=40)

        arr, arr_rep, EvaLum, GtLum, [for_RL, y], feature_vec = RGB_estimation(
            self.image_index, self.now_action, dataset=self.dataset_name
        )

        self.history_EvaLum[:-1] = self.history_EvaLum[1:]
        self.history_EvaLum[-1] = EvaLum

        self.history_action[:-1] = self.history_action[1:]
        self.history_action[-1] = [self.now_action[1], self.now_action[4]]

        self.history_WBsRGB[:-1] = self.history_WBsRGB[1:]
        self.history_WBsRGB[-1] = feature_vec

        diffs = np.linalg.norm(np.diff(self.history_EvaLum, axis=0), axis=1)

        lambda_ = 0.1
        alpha = 0.6

        if self.dataset_name == 'NCC':
            data_mean_arr = 3.1814
            data_max_err = 26.9662
        elif self.dataset_name == 'LEVI':
            data_mean_arr = 3.5369
            data_max_err = 15.3580
        elif self.dataset_name == 'Gehler':
            data_mean_arr = 3.6615
            data_max_err = 23.672049
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        reward = ((self.init_arr - arr) / (self.init_arr)) * (self.init_arr / data_mean_arr) ** alpha - \
                 lambda_ * ((action[0] / 0.6) ** 2 + (action[1] / 4) ** 2) ** 0.5 * (1 - self.init_arr / data_max_err)

        terminated = np.all(diffs[-3:] < 0.05)

        if terminated:
            eps = 1e-12
            rho = arr / max(self.init_arr, eps)
            if rho < 0.8:
                reward += 50
            elif rho < 0.9:
                reward += 30
            elif rho < 0.95:
                reward += 20
            elif rho < 1.0:
                reward += 10
            else:
                reward -= 10

        truncated = (self.current_step >= self.max_steps)

        obs = np.concatenate([
            self.history_WBsRGB.flatten(),
            self.history_action.flatten(),
            np.array([self.current_step])
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, {
            "arr": arr,
            "arr_rep": arr_rep,
            "EvaLum": EvaLum,
            "GtLum": GtLum,
            "for_RL": for_RL
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]