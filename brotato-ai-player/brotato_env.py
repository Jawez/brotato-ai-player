import gymnasium as gym
import numpy as np

import brotato
import brotato_action
from capture import Capture
import cv2

from ultralytics import YOLO
from ocr import OCR

import re
import math

from datetime import datetime
import time
import os

GAME_WIDTH = brotato.WIDTH
GAME_HEIGHT = brotato.HEIGHT
GAME_MAP_LEFT = brotato.MAP_AREA_XYWH[0]
GAME_MAP_TOP = brotato.MAP_AREA_XYWH[1]
GAME_MAP_RIGHT = brotato.MAP_AREA_XYWH[0] + brotato.MAP_AREA_XYWH[2]
GAME_MAP_BOTTOM = brotato.MAP_AREA_XYWH[1] + brotato.MAP_AREA_XYWH[3]

OBSERVATION_SCALE = (1 / 4)
OBSERVATION_WIDTH = int(brotato.MAP_AREA_XYWH[2] * OBSERVATION_SCALE)
OBSERVATION_HEIGHT = int(brotato.MAP_AREA_XYWH[3] * OBSERVATION_SCALE)
OBSERVATION_CHANNELS = brotato.N_CHANNELS   # 1

LAST_WAVE = 20
WAVE_TIMER_DEFAULT = 20
TOTAL_HP_DEFAULT = 10

TOTAL_HP_CHANGE_RANGE = 3
TIMER_CHANGE_RANGE = 3

CONF_THRESHOLD = 0.2   # OCR 部分数字识别确信度较低

YOLO_MODEL_PATH = "models/brotato-cls.onnx"

# # for debug
# OBS_DIR = "obs"

def normalize(value, max, min=0):
    return (value - min) / (max - min)

class BrotatoEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = gym.spaces.Discrete(brotato_action.N_DISCRETE_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(OBSERVATION_HEIGHT, OBSERVATION_WIDTH, OBSERVATION_CHANNELS),
                                                dtype=np.uint8)

        # capture init
        self.cap = Capture()

        # models init
        self.model_cls = YOLO(YOLO_MODEL_PATH)
        self.ocr = OCR()

        # data init
        self.global_step_count = 0
        self.reset_count = 0
        self.reset_time = time.time()
        self.__reset_data()

        # os.makedirs(OBS_DIR, exist_ok=True)

    def __reset_data(self):
        self.step_count = 0
        self.step_elapsed_sum = 0
        self.step_average_elapsed = 0.0

        self.current_wave = 1
        self.current_wave_timer = WAVE_TIMER_DEFAULT

        self.prev_observation = None
        self.prev_scene = brotato.Scene.UNKNOWN
        self.prev_countdown = self.current_wave_timer
        self.prev_hp = 0
        self.prev_total_hp = 0
        self.prev_material = 0
        self.init_material = 0

        self.last_material_reward_step = 0
        self.material_reward_coefficient = 1.0
        self.hp_step_count = 0
        self.time_reward_sum = 0.0
        self.hp_reward_sum = 0.0
        self.hp_step_reward_sum = 0.0
        self.material_reward_sum = 0.0
        self.reward_sum = 0.0

        self.end_text = ""

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        self.global_step_count += 1
        self.step_count += 1

        start_time = time.time()

        # do action
        if self.prev_scene == brotato.Scene.WAVE:
            self.__do_action(action)

        action_time = time.time()

        # get observation
        observation = self.__get_observation()

        obs_time = time.time()

        # identify scene
        scene = self.__identify_scene(observation)
        while scene == brotato.Scene.PAUSE_MENU:
            print("pause menu")
            time.sleep(3)
            observation = self.__get_observation()
            scene = self.__identify_scene(observation)

        scene_time = time.time()

        if scene == brotato.Scene.WAVE or scene == brotato.Scene.WAVE_END:
            hp, total_hp = self.__get_hp(observation)
            countdown = self.__get_timer(observation)

            info = {
                "timer": countdown,
                "hp": hp,
                "total_hp": total_hp,
                # "material": material,
            }

            # 场景误判处理，倒计时 0 有时会识别失败，因此判断大于 1
            if scene == brotato.Scene.WAVE_END and hp > 0 and countdown > 1:
                print("set to wave")
                scene = brotato.Scene.WAVE
            # elif scene == brotato.Scene.WAVE and countdown <= 0:
            #     print(f"countdown: {countdown}, wait wave end")
            #     return self.__resize_observation(observation), reward, terminated, truncated, info

            if scene == brotato.Scene.WAVE:
                material = self.__get_material(observation)

                # calc reward
                reward = self.__calc_reward(hp, material)

                # Note: after calc reward
                if countdown > 0:   # 倒计时为 0 时保留 prev_countdown，用于 WAVE_END 中计算 reward
                    self.prev_countdown = countdown
                self.prev_hp = hp
                self.prev_total_hp = total_hp
                self.prev_material = material

                info["material"] = material
            elif scene == brotato.Scene.WAVE_END:
                wave_result = self.__get_wave_result(observation)
                if wave_result != brotato.WaveResult.UNKNOWN:
                    terminated = True
                    reward = self.__calc_reward(hp, self.prev_material, wave_result)

                self.prev_countdown = countdown
                self.prev_hp = hp
                self.prev_total_hp = total_hp

                info["total_material"] = self.prev_material - self.init_material
                info["end_text"] = self.end_text
        elif (scene == brotato.Scene.ITEM_FOUND) or (scene == brotato.Scene.LEVEL_UP) or (scene == brotato.Scene.SHOP):
            terminated = True
        elif scene == brotato.Scene.RUN_END:
            terminated = True
        elif scene == brotato.Scene.UNKNOWN:
            pass
        else:
            terminated = True
            pass

        self.prev_observation = observation
        self.prev_scene = scene

        end_time = time.time()

        time_elapsed = end_time - start_time
        action_elapsed = action_time - start_time
        obs_elapsed = obs_time - action_time
        scene_elapsed = scene_time - obs_time
        handle_elapsed = end_time - scene_time

        self.step_elapsed_sum += time_elapsed
        self.step_average_elapsed = self.step_elapsed_sum / self.step_count

        # local_time = time.localtime(end_time)
        # ms = int((end_time - int(end_time)) * 1000)
        # time_info = f"{local_time.tm_hour:02d}:{local_time.tm_min:02d}:{local_time.tm_sec:02d}.{ms}"
        time_info = f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
        step_info = f"scene: {scene.value}, terminated: {terminated}, action: {action}, reward: {reward:.2f}, info: {info}"
        debug_info = f"{self.step_count:03d}: start: {start_time:.4f}, elapsed: {time_elapsed:.4f}, ave: {self.step_average_elapsed:.4f}"
        debug_info += f", action: {action_elapsed:.4f}, obs: {obs_elapsed:.4f}, sce: {scene_elapsed:.4f}, hdl: {handle_elapsed:.4f}"
        print(f"{time_info}-{self.global_step_count:06d}: {step_info}")
        # print(f"                {debug_info}")
        return self.__resize_observation(observation), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        print("wait for reset ...")

        observation = self.__get_observation()
        scene = self.__identify_scene(observation)
        while scene != brotato.Scene.WAVE:
            if scene == brotato.Scene.CONFIRM_MENU or scene == brotato.Scene.WAVE_END:
                brotato_action.press_key('enter')   # retry failed waves
            time.sleep(0.5)
            observation = self.__get_observation()
            scene = self.__identify_scene(observation)

        self.__reset_data()

        self.current_wave = self.__get_wave(observation)
        # if self.current_wave >= 10:
        #     self.material_reward_coefficient = 1.0
        # else:
        #     self.material_reward_coefficient = (10 - self.current_wave) + 1

        self.current_wave_timer = self.__get_timer(observation, WAVE_TIMER_DEFAULT)
        self.prev_countdown = self.current_wave_timer

        self.prev_observation = observation
        self.prev_scene = scene

        self.prev_hp, self.prev_total_hp = self.__get_hp(observation, True)
        self.init_material = self.__get_material(observation, True)
        self.prev_material = self.init_material

        info = {
            "wave": self.current_wave,

            "timer": self.current_wave_timer,
            "hp": self.prev_hp,
            "total_hp": self.prev_total_hp,
            "material": self.prev_material,
            # "material_reward_coefficient": self.material_reward_coefficient,
        }

        self.reset_count += 1
        self.reset_time = time.time()

        # end_time = time.time()
        # ms = int((end_time - int(end_time)) * 1000)
        # local_time = time.localtime(end_time)
        # time_info = f"{local_time.tm_hour:02d}:{local_time.tm_min:02d}:{local_time.tm_sec:02d}.{ms}"
        time_info = f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
        print(f"{time_info}: reset count: {self.reset_count}, info: {info}")
        return self.__resize_observation(observation), info

    def render(self):
        obs = self.prev_observation
        if obs is not None:
            self.cap.show(obs)

    # def close(self):
    #     pass

    def pause(self):
        observation = self.__get_observation()
        scene = self.__identify_scene(observation)
        if scene != brotato.Scene.PAUSE_MENU:
            brotato_action.pause()

    def resume(self):
        observation = self.__get_observation()
        scene = self.__identify_scene(observation)
        if scene == brotato.Scene.PAUSE_MENU:
            brotato_action.resume()

    def __resize_observation(self, observation):
        image = cv2.resize(observation, (GAME_WIDTH, GAME_HEIGHT))
        # # for debug
        # cv2.imshow("image", image)
        # cv2.waitKey(1)
        # image_path = os.path.join(OBS_DIR, f'{self.global_step_count:06d}_image.jpg')
        # cv2.imwrite(image_path, image)

        image = image[GAME_MAP_TOP:GAME_MAP_BOTTOM, GAME_MAP_LEFT:GAME_MAP_RIGHT]

        # x, y, w, h = 0, 0, 50, 50
        # region = image[y:y+h, x:x+w]
        # # 对提取的区域应用高斯模糊
        # # 你可以根据需要调整ksize的大小
        # blurred_region = cv2.GaussianBlur(region, (21, 21), 0)
        # # 将模糊后的区域放回原图
        # image[y:y+h, x:x+w] = blurred_region

        image = cv2.resize(image, (OBSERVATION_WIDTH, OBSERVATION_HEIGHT))

        # for debug
        # cv2.imshow("obs", image)
        # cv2.waitKey(1)
        # image_path = os.path.join(OBS_DIR, f'{self.global_step_count:06d}_obs.jpg')
        # cv2.imwrite(image_path, image)

        # # gray_image = cv2.resize(observation, (int(brotato.WIDTH * 0.5), int(brotato.HEIGHT * 0.5)))
        # gray_image = image
        # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        # gray_image_path = os.path.join(OBS_DIR, f'gray_{self.global_step_count:06d}.jpg')
        # cv2.imwrite(gray_image_path, gray_image)

        # gray_image_3d = np.expand_dims(gray_image, axis=-1)

        # return gray_image_3d
        return image

    # Action
    def __do_action(self, action):
        if action == brotato_action.ACTION_UP:
            brotato_action.move_up()
        elif action == brotato_action.ACTION_DOWN:
            brotato_action.move_down()
        elif action == brotato_action.ACTION_LEFT:
            brotato_action.move_left()
        elif action == brotato_action.ACTION_RIGHT:
            brotato_action.move_right()

    # Capture Window
    def __get_observation(self):
        observation = self.cap.capture()
        while observation is None:
            print(f"no window: '{self.cap.get_window_name()}'")
            time.sleep(1)
            observation = self.cap.capture()
        return observation

    # Reward
    def __calc_reward(self, hp, material, wave_result: brotato.WaveResult = None):
        TIME_REWARD_COEFFICIENT = 0.1
        HP_REWARD_COEFFICIENT = 0.15
        HP_STEP_REWARD_COUNT = 10
        HP_STEP_REWARD_COEFFICIENT = 0.015
        MATERIAL_REWARD_COEFFICIENT = 0.02
        MAX_NO_MATERIAL_REWARD_STEP = 50

        material_coefficient = MATERIAL_REWARD_COEFFICIENT

        reward = 0.0

        time_reward = 0.0
        hp_reward = 0.0
        hp_step_reward = 0.0    # 持续未扣血辅助奖励
        material_reward = 0.0

        # 升级等情况下的 hp 提升也计算 reward，暂时不做限制
        # hp reward
        if hp < self.prev_hp:
            hp_reward = - ((self.prev_hp - hp) * HP_REWARD_COEFFICIENT)
            self.hp_step_count = 0

            material_coefficient = MATERIAL_REWARD_COEFFICIENT
        else:
            if hp > self.prev_hp:
                # hp_reward += hp - self.prev_hp
                hp_reward = 1 * HP_REWARD_COEFFICIENT

            self.hp_step_count += 1

            hp_step_reward = int(self.hp_step_count / HP_STEP_REWARD_COUNT) * HP_STEP_REWARD_COEFFICIENT
            if hp_step_reward > 0.1:
                hp_step_reward = 0.1

                material_coefficient = 0.1

            if self.step_count > self.last_material_reward_step + MAX_NO_MATERIAL_REWARD_STEP:
                hp_step_reward = 0

        if wave_result is None:
            # material reward
            # 通过时有收获加成增加材料，不能计算奖励
            if material > self.prev_material:
                material_reward = (material - self.prev_material) * material_coefficient    # * self.material_reward_coefficient

                self.last_material_reward_step = self.step_count

        elif wave_result == brotato.WaveResult.COMPLETED:
            # time_reward += self.current_wave_timer * TIME_REWARD_COEFFICIENT

            hp = hp or 1
            total_hp = self.prev_total_hp or TOTAL_HP_DEFAULT
            hp_reward += math.pow(total_hp, hp / total_hp) * HP_REWARD_COEFFICIENT

            # # wave 11 收集材料数可以达到 200 以上，以及散落 100 左右
            # # wave 19 收集材料数可以达到 600 以上，以及散落 100 多
            # material_reward += self.material_reward_sum / self.current_wave
        elif wave_result == brotato.WaveResult.LOST:
            countdown = self.prev_countdown or 1
            wave_timer = self.current_wave_timer or WAVE_TIMER_DEFAULT
            time_reward -= (math.pow(wave_timer, countdown / wave_timer) * TIME_REWARD_COEFFICIENT)
            # time_reward -= math.pow(wave_timer * TIME_REWARD_COEFFICIENT, countdown / wave_timer)
        elif wave_result == brotato.WaveResult.WON:
            # # 倒计时 countdown 这里始终为 0，prev_countdown 通过 __get_timer 的处理限制为变为 1 前的数值
            # # time_reward = (self.current_wave_timer + self.prev_countdown) * TIME_REWARD_COEFFICIENT
            # time_reward += self.prev_countdown * TIME_REWARD_COEFFICIENT   # 快速击杀 boss 奖励

            hp = hp or 1
            total_hp = self.prev_total_hp or TOTAL_HP_DEFAULT
            hp_reward += math.pow(total_hp, hp / total_hp) * HP_REWARD_COEFFICIENT

            # material_reward += self.material_reward_sum / self.current_wave

        # total reward
        reward = time_reward + hp_reward + hp_step_reward + material_reward

        self.time_reward_sum += time_reward
        self.hp_reward_sum += hp_reward
        self.hp_step_reward_sum += hp_step_reward
        self.material_reward_sum += material_reward
        self.reward_sum += reward

        # print(f"reward: {reward:.2f}, time: {time_reward:.2f}, hp: {hp_reward:.2f}, hp step: {hp_step_reward:.2f}, material: {material_reward:.2f}, hp step count: {self.hp_step_count}, material coef: {material_coefficient}")
        # print(f"   sum: {self.reward_sum:.2f}, time: {self.time_reward_sum:.2f}, hp: {self.hp_reward_sum:.2f}, hp step: {self.hp_step_reward_sum:.2f}, material: {self.material_reward_sum:.2f}")

        return reward

    # OCR
    def __recognize_text(self, roi) -> tuple[str, float]:
        text = ""
        conf = 0.0

        # 尝试多次识别， conf 均相同，没必要 retry
        results, elapse = self.ocr.recognize(roi)
        # print(f'ocr results: {results}, elapse: {elapse}')
        if results and results[0]:
            result = results[0]
            conf = result[1]
            if conf >= CONF_THRESHOLD:
                text = result[0]

        return text, conf

    def __match_text(self, observation, roi_xyxy, pattern): # -> (Match[str] | None)
        x, y, x1, y1 = roi_xyxy

        # # for debug
        # cv2.rectangle(observation, (x, y), (x1, y1), (0, 0, 255), 1)

        roi = observation[y:y1, x:x1]
        text, conf = self.__recognize_text(roi)
        if text:
            return re.match(pattern, text)

        return None

    def __get_wave_result(self, observation) -> brotato.WaveResult:
        wave_result = brotato.WaveResult.UNKNOWN

        match_result = self.__match_text(observation, brotato.BOX_WAVE_RESULT_XYXY[0], r'(\S*)')
        end_text = (match_result and match_result.group(1)) or ""
        if end_text and len(end_text) >= brotato.WAVE_TEXT_MATCH_LEN:
            self.end_text = end_text

            text = end_text[:brotato.WAVE_TEXT_MATCH_LEN]
            if text == brotato.WAVE_COMPLETED_TEXT:
                wave_result = brotato.WaveResult.COMPLETED
            elif text == brotato.WAVE_LOST_TEXT:
                wave_result = brotato.WaveResult.LOST
            elif text == brotato.WAVE_WON_TEXT:
                wave_result = brotato.WaveResult.WON

        return wave_result

    def __match_material_num(self, observation, box_index) -> int:
        material = self.prev_material

        if box_index > len(brotato.BOX_MATERIAL_XYXY):
            return material

        x, y, x1, y1 = brotato.BOX_MATERIAL_XYXY[box_index]

        # # for debug
        # cv2.rectangle(observation, (x, y), (x1, y1), (0, 0, 255), 1)

        roi = observation[y:y1, x:x1]
        text, conf = self.__recognize_text(roi)
        if text:
            pattern = r'^\D*(\d+)'
            result = re.match(pattern, text)
            if result:
                # 处理 reset 时 0 后面出现误判数字的情况，如'02'直接返回 0
                material_text = result.group(1)
                if material_text and material_text[0] == '0':
                    material = 0
                else:
                    next_material = int(material_text)
                    # 处理 2/3 误判为 5
                    if next_material == 5 and material <= 3 and conf < 0.6:
                        pass
                    # 处理 0 误判为 6
                    elif next_material == 6 and material == 0 and conf < 0.6:
                        pass
                    else:
                        material = next_material

        return material

    # Note: OCR 存在0、3误判为6，10误判为16，3误判为5、13，2、3、1之间误判，4、5误判为1，11连续多次误判为1等情况
    def __get_material(self, observation, reset=False):
        if reset:
            box_index = 3
        else:
            # TODO: optimize
            if self.prev_material >= 1000:
                box_index = 3   # '9999'
            elif self.prev_material >= 100:
                box_index = 2   # '999'
            elif self.prev_material >= 10:
                box_index = 1   # '99'
            else:
                box_index = 0   # '9'

        material = self.__match_material_num(observation, box_index)

        # 波次中材料数不会变少，始终大于等于前一次的检测值
        if material < self.prev_material:
            print(f"less material: {material}, prev_material: {self.prev_material}")
            # 日志记录到 4 开始就变为一直识别 1 的情况；86 开始 一直识别为 10 或 11 等
            if (box_index == 0 and self.prev_material >= 4 and material <= 2) or \
               (box_index == 1 and self.prev_material >= 85 and material <= 12) or \
               (box_index == 2 and self.prev_material >= 980 and material <= 102):
                box_index += 1
                material = self.__match_material_num(observation, box_index)
                print(f"re match index: {box_index}, material: {material}")
        elif material >= (self.prev_material * 10):
            # 非初始状态下，识别到的 material 为 prev_material 的 10 倍，认为是识别错误
            if self.prev_material > 0:
                print(f"error material: {material}, prev_material: {self.prev_material}")
                # if material >= (self.prev_material * 100):
                #     material = int(material / 100)
                # else:
                #     material = int(material / 10)
                material = self.prev_material

        if material < self.prev_material:
            material = self.prev_material

        return material

    # Note: 扣血过程中（血条背景色变为白色）会出现识别错误的情况
    def __get_hp(self, observation, reset=False):
        hp = self.prev_hp
        total_hp = self.prev_total_hp

        # TODO: optimize
        xyxy = brotato.BOX_HP_XYXY[0]
        if (not reset) and self.prev_total_hp > 0 and self.prev_total_hp < 100:
            xyxy = brotato.BOX_HP_XYXY[1]

        pattern = r'^\s*(\d+)\s*/\s*(\d+)\s*'
        result = self.__match_text(observation, xyxy, pattern)
        if result:
            hp = int(result.group(1))
            total_hp = int(result.group(2))

        # 识别到的 total_hp 变化一定值（升级/特定道具/特定会使 total_hp 增加或减少），认为是识别错误
        if total_hp < self.prev_total_hp - TOTAL_HP_CHANGE_RANGE:
            print(f"error total_hp: {total_hp}, prev_total_hp: {self.prev_total_hp}")
            hp = self.prev_hp   # total_hp 识别错误时，hp 可能也识别错误    # TODO: optimize
            total_hp = self.prev_total_hp
        elif total_hp > self.prev_total_hp + TOTAL_HP_CHANGE_RANGE:
            # 非初始状态下
            if self.prev_total_hp > 0:
                print(f"error total_hp: {total_hp}, prev_total_hp: {self.prev_total_hp}")
                hp = self.prev_hp
                # total_hp = (int(self.prev_total_hp / 10) * 10) + (total_hp % 10)  # 89->90 的情况未处理到
                total_hp = self.prev_total_hp

        if hp > total_hp:
            hp = self.prev_hp

        return hp, total_hp

    def __get_wave(self, observation):
        wave = 1

        pattern = r'^第\s*(\d+)\s*波'
        result = self.__match_text(observation, brotato.BOX_WAVE_XYXY[0], pattern)
        if result:
            wave = int(result.group(1)) or 1

        return wave

    # 识别 0 和 3 出现确信度较低的情况，且会在数字前面识别出其他符号
    def __get_timer(self, observation, reset_timer=None):
        timer = reset_timer or self.prev_countdown

        roi_xyxy = brotato.BOX_TIMER_XYXY[0]
        if timer < 10:
            roi_xyxy = brotato.BOX_TIMER_XYXY[1]

        pattern = r'^\D*(\d+)'
        result = self.__match_text(observation, roi_xyxy, pattern)
        if result:
            # 处理倒计时从 10 到 9 时，出现 9 后面多误判数字的情况，如 9 识别为 94/95
            timer_text = result.group(1)
            if timer_text and timer_text[0] == '9' and self.prev_countdown == 10:
                timer = 9
            else:
                timer = int(timer_text)

        # 非初始状态下
        if reset_timer is None:
            # 识别到的 timer 变化一定值，认为是识别错误
            # 这里的判断条件会使胜利时的 timer 变化不生效
            if timer < self.prev_countdown - TIMER_CHANGE_RANGE:
                # 最后一关击杀 boss 时 timer 会变为 1 然后才到 0，这里跳到 1 时不更新，避免产生时间奖励
                if self.current_wave == LAST_WAVE and timer == 0:
                        pass
                elif self.current_wave == LAST_WAVE and timer == 1:
                        print(f"won timer: {timer}, prev_countdown: {self.prev_countdown}")
                        timer = self.prev_countdown
                        time.sleep(0.2) # wait timer to 0
                else:
                    real_elapsed = int(time.time() - self.reset_time)
                    timer_elapsed = self.current_wave_timer - timer
                    if timer_elapsed <= (real_elapsed + 1) and timer_elapsed >= (real_elapsed - 1):
                        print(f"calibrate timer: {timer}, prev_countdown: {self.prev_countdown}, real_elapsed: {real_elapsed}, timer_elapsed: {timer_elapsed}")
                    else:
                        print(f"error timer: {timer}, prev_countdown: {self.prev_countdown}, real_elapsed: {real_elapsed}, timer_elapsed: {timer_elapsed}")
                        timer = self.prev_countdown
            elif timer > self.prev_countdown:
                    print(f"error timer: {timer}, prev_countdown: {self.prev_countdown}")
                    timer = self.prev_countdown
        else:
            if timer <= 0:
                print(f"error timer: {timer}, reset: {reset_timer}")
                timer = reset_timer

        return timer

    # Image Classification
    def __identify_scene(self, observation):
        scene = brotato.Scene.UNKNOWN

        results = self.model_cls(observation, verbose=False)
        if results and results[0]:
            probs = results[0].probs
            try:
                # print(f"top 1: {probs.top1}, {probs.top1conf.item():.4f}, top 5: {probs.top5}, {probs.top5conf.tolist()}")
                top1_confidence = probs.top1conf    # get confidence of top 1 class
                if top1_confidence.item() > CONF_THRESHOLD:
                    scene = brotato.Scene(probs.top1)

            except ValueError:
                print(f"invalid probs: {probs}")

        return scene
