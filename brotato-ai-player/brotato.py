from enum import Enum

WINDOW_NAME = "Brotato"
ASPECT_RATIO = 16 / 9
N_CHANNELS = 3
HEIGHT = 540
WIDTH = 960
MAP_AREA_XYWH = (30, 80, 900, 430)

BOX_HP_XYXY = [
    [64, 17, 119, 29],      # '226/226'
    [70, 16, 112, 30],      # '40/40'
]

BOX_MATERIAL_XYXY = [
    [44, 70, 61, 91],       # '9'
    [44, 70, 75, 91],       # '99'
    [44, 70, 89, 91],       # '999'
    [44, 70, 103, 91]       # '9999'
]

BOX_WAVE_XYXY = [
    [447, 9, 511, 32],      # '第19波'
    [452, 9, 506, 32],      # '第3波'
]

BOX_TIMER_XYXY = [
    [461, 43, 497, 68],     # '60'
    [470, 43, 488, 68],     # [473, 47, 488, 66],     # '3'
]

BOX_WAVE_RESULT_XYXY = [
    [426, 66, 535, 130]     # '通过', '胜利', '战败'
]

WAVE_COMPLETED_TEXT = '通'  # '通过'    # [431, 81, 506, 117]
WAVE_WON_TEXT       = '胜'  # '胜利'    # [426, 66, 535, 130]   # [427, 70, 532, 128]
WAVE_LOST_TEXT      = '战'  # '战败'    # [426, 66, 535, 130]
WAVE_TEXT_MATCH_LEN = len(WAVE_COMPLETED_TEXT)

class WaveResult(Enum):
    COMPLETED = 0
    WON = 1
    LOST = 2

    UNKNOWN = 99

class Scene(Enum):
    MAIN_MENU = 0
    CHARACTER_SELECTION = 1
    WEAPON_SELECTION = 2
    DIFFICULTY_SELECTION = 3

    WAVE = 4
    WAVE_END = 5
    SHOP = 6
    LEVEL_UP = 7
    ITEM_FOUND = 8
    RUN_END = 9

    PAUSE_MENU = 10
    OPTIONS_MENU = 11
    GENERAL_MENU = 12
    GAMEPLAY_MENU = 13
    CONFIRM_MENU = 14

    UNKNOWN = 99
