'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-05-22 09:56:25
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-06-17 17:54:36
FilePath: /cv_Course_Design/app.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
'''
import tkinter as tk
from time import time
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.kcftracker import KCFTracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1 # 初始化矩形框坐标
w, h = 0, 0

inteval = 1
duration = 0.01
# mouse callback function
done = False


def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h, done

    # 画矩形框
    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y # 起始坐标
        cx, cy = x, y # 当前坐标

    # 移动鼠标，画出矩形框
    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y # 当前坐标

    # 放开鼠标，完成矩形框
    elif event == cv2.EVENT_LBUTTONUP: 
        selectingObject = False
        if(abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)   # w:矩形框的宽，h:矩形框的高
            ix, iy = min(x, ix), min(y, iy)   # ix,iy:矩形框左上角的坐标
            initTracking = True
            done = True  # 用户完成选择
        else:
            onTracking = False
            done = True  # 用户完成选择

    # 右键按下，取消选取矩形框
    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if(w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True

class App:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root)
        self.canvas.pack()
        self.image_path = None
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        # 按钮用来打开图片
        open_button = tk.Button(root, text="打开图片", command=self.open_image)
        open_button.pack()

        # 新增 "开始追踪" 按钮
        track_button = tk.Button(root, text="开始追踪", command=self.start_tracking)
        track_button.pack()

        # 新增 "使用摄像头" 按钮
        cam_button = tk.Button(root, text="使用摄像头", command=self.use_camera)
        cam_button.pack()

    def open_image(self):
        # 请用户选择一个图片文件
        self.image_path = filedialog.askopenfilename()
        if not self.image_path:
            return

        self.image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas.config(width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

        # 绑定鼠标点击、移动、释放事件
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        # 存储矩形起始点坐标
        self.start_x = event.x
        self.start_y = event.y

        # 创建矩形
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red')

    def on_move_press(self, event):
        # 更新矩形的大小
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        # 鼠标释放后，存储结束点坐标
        self.end_x, self.end_y = event.x, event.y

    def start_tracking(self):
        # 检查用户是否已经选择了一个区域
        if self.start_x is None or self.start_y is None or self.end_x is None or self.end_y is None:
            messagebox.showinfo('Info', 'You have not selected a rectangular area yet.')
            return

        # 请用户选择一个视频文件
        video_path = filedialog.askopenfilename()
        if not video_path:
            return

        # 传递视频路径和矩形坐标给 track 函数
        self.track(video_path, (self.start_x, self.start_y, self.end_x, self.end_y), cv2.imread(self.image_path))

    def compare(self, frame, image):
        global initTracking,selectingObject
        frame = cv2.resize(frame, (image.shape[1], image.shape[0]))
        # 比较两个图像是否相同
        difference = np.sum((image.astype("float") - frame.astype("float")) ** 2)
        difference /= float(image.shape[0] * image.shape[1] * image.shape[2])
        print(difference)
        if difference < 100:
            print("找到你的图片所在帧了！")
            initTracking = True
            selectingObject = False

    def use_camera(self):
        # 使用摄像头，传入空的视频路径和零坐标
        global selectingObject, initTracking, onTracking
        selectingObject = False
        initTracking = False
        onTracking = False
        self.track('', (0, 0, 0, 0), '')

    def track(self, video_path, coords, image):
        # 这里应该添加实际的追踪代码
        global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h, done,duration

        if  video_path:  # 检查是否在命令行中输入了视频文件路径
            ix, iy, cx, cy = coords[0], coords[1], coords[2], coords[3]
            selectingObject = True  
            cap = cv2.VideoCapture(video_path)
            cv2.namedWindow('tracking')
            print("用户框选的区域是：",ix, iy, cx, cy)


        else:
            cap = cv2.VideoCapture(0) 
            cv2.namedWindow('tracking')                        # 创建一个窗口
            cv2.setMouseCallback('tracking', draw_boundingbox) # 设置鼠标事件的回调函数


        tracker = KCFTracker(True, True, True)  # hog, fixed_window, multiscale
        # 如果使用hog特征，第一次框完之后会有一个短暂的停顿，这是因为Numba的缘故。

        while(cap.isOpened()):
            ret, frame = cap.read() # 读取一帧的图像
            if not ret: # 如果读取失败，退出程序
                break

            if(selectingObject):
                # 画出矩形框
                if video_path:
                    self.compare(frame, image)
                else:
                    cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)

            elif(initTracking):
                # 初始化跟踪器
                w, h = abs(cx - ix), abs(cy - iy)
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
                print("起始追踪的坐标：", ix, iy, w, h)
                tracker.init([ix, iy, w, h], frame) # 初始化跟踪器
                

                initTracking = False
                onTracking = True

            elif(onTracking):
                # 跟踪
                t0 = time()
                boundingbox = tracker.update(frame) # 更新跟踪器
                t1 = time()

                boundingbox = list(map(int, boundingbox))
                print(boundingbox)
                cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), 
                                (0, 255, 255), 1) # 

                duration = 0.8 * duration + 0.2 * (t1 - t0)
                #duration = t1-t0
                cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('tracking', frame)
            c = cv2.waitKey(inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f'Video path: {video_path}')
        print(f'Tracking coordinates: {coords}')

root = tk.Tk()
app = App(root)
root.mainloop()

