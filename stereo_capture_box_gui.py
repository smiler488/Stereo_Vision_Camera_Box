# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os, time, json, math, threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # pip install pillow

# ================= 相机与标定参数 =================
left_camera_matrix = np.array([[416.841180253704, 0.0, 338.485167779639],
                               [0., 416.465934495134, 230.419201769346],
                               [0., 0., 1.]], dtype=np.float64)
left_distortion = np.array([-0.0170280933781798, 0.0643596519467521, -0.00161785356900972, -0.00330684695473645, 0], dtype=np.float64)

right_camera_matrix = np.array([[417.765094485395, 0.0, 315.061245379892],
                                [0., 417.845058291483, 238.181766936442],
                                [0., 0., 1.]], dtype=np.float64)
right_distortion = np.array([-0.0394089328586398, 0.131112076868352, -0.00133793245429668, -0.00188957913931929, 0], dtype=np.float64)

R = np.array([[0.999962872853149, 0.00187779299260463, -0.00840992323112715],
              [-0.0018408858041373, 0.999988651353238, 0.00439412154902114],
              [0.00841807904053251, -0.00437847669953504, 0.999954981430194]], dtype=np.float64)
T = np.array([[-120.326603502087], [0.199732192805711], [-0.203594457929446]], dtype=np.float64)

IMAGE_SIZE = (640, 480)  # (width, height)
CAM_ID = 0               # 拼接流相机设备号

# 立体校正与重映射
R1, R2, P1, P2, Q, valid1, valid2 = cv2.stereoRectify(
    left_camera_matrix, left_distortion,
    right_camera_matrix, right_distortion,
    IMAGE_SIZE, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, IMAGE_SIZE, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, IMAGE_SIZE, cv2.CV_16SC2)

# SGBM + WLS
window_size = 5
num_disp = 16*10
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=12,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
use_wls = False
try:
    import cv2.ximgproc as ximgproc
    right_matcher = ximgproc.createRightMatcher(left_matcher)
    wls_filter = ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    use_wls = True
except Exception:
    pass

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ================== 采集线程 ==================
class StereoApp:
    def __init__(self, root):
        self.root = root
        root.title("CottonLeaf Stereo Capture")

        self.sample_id = tk.StringVar()
        self.preview_running = False
        self.thread = None
        self.stop_event = threading.Event()

        # 顶部控制栏
        ctrl = ttk.Frame(root)
        ctrl.pack(fill="x", padx=10, pady=8)
        ttk.Label(ctrl, text="样品编号:").pack(side="left")
        self.entry = ttk.Entry(ctrl, textvariable=self.sample_id, width=20)
        self.entry.pack(side="left", padx=6)
        ttk.Button(ctrl, text="开始预览", command=self.start_preview).pack(side="left", padx=4)
        ttk.Button(ctrl, text="保存当前样品 (s)", command=self.save_current).pack(side="left", padx=4)
        ttk.Button(ctrl, text="下一样品", command=self.next_sample).pack(side="left", padx=4)
        ttk.Button(ctrl, text="退出", command=self.on_quit).pack(side="right")

        # 画面区域
        view = ttk.Frame(root)
        view.pack(fill="both", expand=True, padx=10, pady=8)

        self.lbl_left  = ttk.Label(view, text="Left")
        self.lbl_right = ttk.Label(view, text="Right")
        self.lbl_depth = ttk.Label(view, text="Depth")

        self.lbl_left.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_right.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_depth.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Bind mouse click event to depth label
        self.lbl_depth.bind("<Button-1>", self.on_depth_click)

        # 相机
        self.cap = cv2.VideoCapture(CAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0]*2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            root.after(200, root.destroy)

        # 状态
        self.last_frames = (None, None, None)  # left_bgr, right_bgr, disp_float

    def start_preview(self):
        if self.preview_running:
            return
        if not self.sample_id.get().strip():
            messagebox.showwarning("提示", "请先输入样品编号")
            return
        self.preview_running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                continue
            # 分割拼接流
            left_raw  = frame[0:IMAGE_SIZE[1], 0:IMAGE_SIZE[0]]
            right_raw = frame[0:IMAGE_SIZE[1], IMAGE_SIZE[0]:IMAGE_SIZE[0]*2]
            # 矫正
            left_bgr  = cv2.remap(left_raw,  left_map1,  left_map2,  cv2.INTER_LINEAR)
            right_bgr = cv2.remap(right_raw, right_map1, right_map2, cv2.INTER_LINEAR)
            # 灰度 + CLAHE
            imgL = clahe.apply(cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY))
            imgR = clahe.apply(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY))
            # 视差
            dispL = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0
            if use_wls:
                dispR = right_matcher.compute(imgR, imgL).astype(np.float32) / 16.0
                disp  = wls_filter.filter(dispL, left_bgr, disparity_map_right=dispR)
            else:
                disp  = dispL
            # 可视化
            disp_u8 = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)

            # 更新GUI图像
            self.show_image(self.lbl_left,  left_bgr)
            self.show_image(self.lbl_right, right_bgr)
            self.show_image(self.lbl_depth, disp_color)

            # 存最近帧用于保存
            self.last_frames = (left_bgr, right_bgr, disp)

        # 线程退出时清空状态
        self.preview_running = False

    def show_image(self, label, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im  = Image.fromarray(rgb)
        im  = im.resize((int(IMAGE_SIZE[0]*0.6), int(IMAGE_SIZE[1]*0.6)))
        imgtk = ImageTk.PhotoImage(image=im)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def save_current(self):
        sample = self.sample_id.get().strip()
        if not sample:
            messagebox.showwarning("提示", "请先输入样品编号")
            return
        left_bgr, right_bgr, disp = self.last_frames
        if left_bgr is None or disp is None:
            messagebox.showwarning("提示", "没有可保存的帧，请先开始预览")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join("data", sample)
        for sub in ["left", "right", "depth", "pointcloud"]:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

        left_path  = os.path.join(base, "left",  f"{sample}_{ts}_left.png")
        right_path = os.path.join(base, "right", f"{sample}_{ts}_right.png")
        depth_png  = os.path.join(base, "depth", f"{sample}_{ts}_depth.png")
        depth_npy  = os.path.join(base, "depth", f"{sample}_{ts}_disp.npy")
        ply_path   = os.path.join(base, "pointcloud", f"{sample}_{ts}.ply")
        meta_path  = os.path.join(base, f"{sample}_{ts}_metadata.json")

        disp_u8 = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)

        cv2.imwrite(left_path, left_bgr)
        cv2.imwrite(right_path, right_bgr)
        cv2.imwrite(depth_png, disp_color)
        np.save(depth_npy, disp.astype(np.float32))

        # 点云（不过滤，保持和命令行版一致）
        points_3d = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
        mask = (disp > 0) & np.isfinite(points_3d[:,:,2])
        pts  = points_3d[mask]
        cols = left_bgr[mask]
        with open(ply_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts, cols):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[2])} {int(c[1])} {int(c[0])}\n")

        # 元数据
        meta = {
            "sample_id": sample,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera": {
                "cam_id": CAM_ID,
                "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "gain": self.cap.get(cv2.CAP_PROP_GAIN),
            },
            "stereo": {"num_disp": num_disp, "window_size": window_size, "use_wls": use_wls}
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        messagebox.showinfo("完成", f"已保存到：{base}")

    def next_sample(self):
        # 清空输入框，等待输入下一个编号
        self.sample_id.set("")
        self.entry.focus_set()

    def on_quit(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def on_depth_click(self, event):
        # 计算点击位置在原始深度图中的坐标
        # 因为深度图显示被缩放了0.6倍
        x = int(event.x / 0.6)
        y = int(event.y / 0.6)
        _, _, disp = self.last_frames
        if disp is None:
            messagebox.showwarning("提示", "没有有效的深度数据")
            return
        if x < 0 or x >= disp.shape[1] or y < 0 or y >= disp.shape[0]:
            messagebox.showwarning("提示", "点击位置超出范围")
            return
        disparity = disp[y, x]
        if disparity <= 0 or not np.isfinite(disparity):
            messagebox.showinfo("距离", "该点视差无效，无法计算距离")
            return

        # 计算距离 Z = f*B / disparity
        # f 和 B 从相机参数中计算
        f = P1[0,0]  # 焦距，单位像素
        B = abs(T[0,0])/1000.0  # 基线，单位米（T单位为mm，转换为m）
        Z = (f * B) / disparity  # 单位米

        messagebox.showinfo("距离", f"点击点深度距离约为 {Z:.3f} 米")

def main():
    root = tk.Tk()
    app = StereoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()