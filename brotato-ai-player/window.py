import win32gui, win32ui, win32con
import ctypes
import numpy as np

def get_window_handle(window_name):
    return win32gui.FindWindow(None, window_name)

def get_screen_dpi():
    # 定义所需的Windows API常量
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # 设置进程DPI感知级别，2 - PROCESS_PER_MONITOR_DPI_AWARE
    # 获取系统DPI设置
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    # 获取主显示器的逻辑DPI
    hdc = user32.GetDC(0)
    dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
    dpi_y = ctypes.windll.gdi32.GetDeviceCaps(hdc, 90)  # LOGPIXELSY
    user32.ReleaseDC(0, hdc)
    # 返回DPI值
    return dpi_x, dpi_y

def calc_screen_scale() -> float:
    # 获取DPI
    dpi_x, dpi_y = get_screen_dpi()

    # # 计算缩放百分比
    # scale_percentage = round((dpi_x / 96.0) * 100)  # 96 DPI（dots per inch，每英寸点数）
    # print(f"Screen Scale Percentage: {scale_percentage}%")

    return dpi_x / 96.0

class Window():
    def __init__(self, window_name: str, aspect_ratio: float | None = None):
        self.window_name = window_name
        self.hwnd = None

        self.aspect_ratio = aspect_ratio
        self.screen_scale = calc_screen_scale()

    def reset(self):
        self.hwnd = None
        self.screen_scale = calc_screen_scale()

    def get_screen_scale(self):
        return self.screen_scale

    def __calc_image_rect(self):
        if not self.hwnd:
            return None

        # 获取窗口客户区矩形
        left, top, width, height = win32gui.GetClientRect(self.hwnd)

        if self.aspect_ratio is not None:
            target_width = int(self.aspect_ratio * height)
            target_height = int(width / self.aspect_ratio)
            if width > target_width:
                left += int((width - target_width) / 2)
                width = target_width
            elif width < target_width:
                top += int((height - target_height) / 2)
                height = target_height

        # 将窗口客户区的左上角坐标转换为屏幕坐标
        left, top = win32gui.ClientToScreen(self.hwnd, (left, top))

        # print(f"image rect: {left, top, width, height}")
        return left, top, width, height

    def ___handle_scale(self, width, height, left_off, top_off):
        # # 用于历史游戏版本 1.0.x
        # # 根据缩放比例调整捕获区域
        # screen_scale = self.screen_scale
        # width = round(width / screen_scale)
        # height = round(height / screen_scale)
        # left_off = round(left_off / screen_scale)
        # top_off = round(top_off / screen_scale)

        return width, height, left_off, top_off

    # 返回 BGRA numpy 数组
    def grab(self):
        if not self.hwnd:
            self.hwnd = get_window_handle(self.window_name)
            if not self.hwnd:
                return None

        # 用桌面窗口句柄时只有当要捕获的程序窗口在顶端显示时才能捕获到，用程序窗口的句柄即使程序不在顶端显示也能捕获到画面
        hwnd = self.hwnd    # win32gui.GetDesktopWindow()

        try:
            window_state = win32gui.GetWindowPlacement(hwnd)
            if window_state[1] == win32con.SW_SHOWMINIMIZED:
                print("minimized")
                return None

            # 每次获取时都重新计算才能保证调整窗口位置和大小后依然得到正确的值
            image_left, image_top, image_width, image_height = self.__calc_image_rect()

            # 获取窗口矩形
            window_left, window_top, window_width, window_height = win32gui.GetWindowRect(hwnd)
            # 获取客户区左上角相对于窗口左上角的坐标
            left_off = image_left - window_left
            top_off = image_top - window_top

            # 处理系统缩放
            width, height, left_off, top_off = self.___handle_scale(image_width, image_height, left_off, top_off)
            # print(f"size and off: {width, height, left_off, top_off}")

            # 创建一个设备上下文
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            # 创建一个位图对象
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bmp)

            # 将窗口内容复制到位图中
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (left_off, top_off), win32con.SRCCOPY)   # 采集的图像有偏移，不确定是不是窗口边框有影响

            # 获取位图数据并转换为PIL图像
            bmp_bits = bmp.GetBitmapBits(True)

            # 将位图数据转换为 NumPy 数组
            bmp_array = np.frombuffer(bmp_bits, dtype='uint8')
            bmp_array.shape = (height, width, 4)  # 4 表示 BGRA

            # 清理资源
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            win32gui.DeleteObject(bmp.GetHandle())

            return bmp_array

        except Exception as e:
            print(e)
            self.reset()

        return None
