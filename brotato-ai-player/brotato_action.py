# import pydirectinput
import ctypes
from ctypes import wintypes

import time

N_DISCRETE_ACTIONS = 4

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

PRESS_KEEP_TIME = 0.075  # 0.1  #


# ref pydirectinput
SendInput = ctypes.windll.user32.SendInput

# KeyBdInput Flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_UNICODE = 0x0004

# Keyboard Scan Code Mappings
KEYBOARD_MAPPING = {
    'w': 0x11,
    'a': 0x1E,
    's': 0x1F,
    'd': 0x20,

    'esc': 0x01,
    'enter': 0x1C,
}

# C struct redefinitions

PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def key_down(key):
    # pydirectinput.keyDown(key)

    keybdFlags = KEYEVENTF_SCANCODE

    hexKeyCode = KEYBOARD_MAPPING[key]
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, keybdFlags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def key_up(key):
    # pydirectinput.keyUp(key)

    keybdFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP

    hexKeyCode = KEYBOARD_MAPPING[key]
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, keybdFlags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)

    # SendInput returns the number of event successfully inserted into input stream
    # https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput#return-value
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def press_key(key, keep_time=PRESS_KEEP_TIME):
    # pydirectinput.press(key)
    key_down(key)
    time.sleep(keep_time)
    key_up(key)

def move_up():
    press_key('w')

def move_down():
    press_key('s')

def move_left():
    press_key('a')

def move_right():
    press_key('d')

def pause():
    press_key('esc')

def resume():
    press_key('enter')
