import brotato as game
from window import Window
from decimal import Decimal, ROUND_HALF_UP
import cv2

import keyboard
import time
import os

CAPTURE_DIR = "captured"

class Capture:
    def __init__(self):
        self.window_name = game.WINDOW_NAME
        self.game_window = Window(self.window_name, game.ASPECT_RATIO)

        self.prev_image = None
        self.image_count = 0

        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def get_window_name(self):
        return self.window_name

    def capture(self, save=False):
        observation = self.game_window.grab()
        if observation is not None:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
            observation = cv2.resize(observation, (game.WIDTH, game.HEIGHT))
            if save:
                self.__save_diff_image(observation)
        return observation

    def __save_diff_image(self, image):
        # 检查图像尺寸是否相同
        if (self.prev_image is not None) and (image.shape == self.prev_image.shape):
            # 比较两个图像的每个像素
            difference = cv2.subtract(image, self.prev_image)
            b, g, r = cv2.split(difference)

            # 如果所有通道的像素差异都是0，则图像完全一致
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                return

        self.image_count += 1
        image_path = os.path.join(CAPTURE_DIR, f'{self.image_count:06d}.jpg')
        cv2.imwrite(image_path, image)
        print(f"save: {image_path}")

        self.prev_image = image


    def show(self, image):
        # screen_scale = self.game_window.get_screen_scale()
        # decimal_number = Decimal(str(screen_scale))
        # scale_integer = int(decimal_number.quantize(Decimal('1'), rounding=ROUND_HALF_UP))

        # desired_width = int((image.shape[1] / scale_integer) * screen_scale)
        # desired_height = int((image.shape[0] / scale_integer) * screen_scale)
        # # print(f'shape: {image.shape}, desired size: {desired_width, desired_height}')

        # image = cv2.resize(image, (desired_width, desired_height))

        cv2.imshow('image', image)
        cv2.waitKey(1)

if __name__ == "__main__":
    cap = Capture()

    while not keyboard.is_pressed('q'):
        observation = cap.capture(True)
        if observation is not None:
            cap.show(observation)
        else:
            print("no obs")

        time.sleep(0.5)
