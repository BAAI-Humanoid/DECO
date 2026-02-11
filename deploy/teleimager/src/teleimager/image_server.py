import os
import argparse
import glob
import cv2
import numpy as np
import uvc
import yaml
import time
from . import zmq_msg
from . import webrtc_msg
import threading
import signal
import functools
import subprocess
import platform
import logging_mp
import pyzed.sl as sl
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# Resolve the absolute path of cam_config_server.yaml relative to this script
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "cam_config_server.yaml"
)
CONFIG_PATH = os.path.normpath(CONFIG_PATH)


def reload_uvc_driver():
    try:
        subprocess.run("sudo modprobe -r uvcvideo", shell=True, check=True)
        time.sleep(1)
        subprocess.run("sudo modprobe uvcvideo debug=0", shell=True, check=True)
        time.sleep(1)
        logger_mp.info("UVC driver reloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger_mp.error(f"Failed to reload driver: {e}")

# ========================================================
# camera and camera discovery
# ========================================================
class CameraFinder:
    """
    Discover connected cameras and their properties.
    vpath: /dev/videoX
    ppath: physical path in /sys/class/video4linux, e.g. /sys/devices/pci0000:00/0000:00:14.0/usb1/1-11/1-11.2/1-11.2:1.0
    uid: USB unique ID, e.g. "001:002"
    dev_info: extra info from uvc
    sn: serial number of the camera
    """
    def __init__(self, realsense_enable=False, verbose=False):
        self.verbose = verbose
        # uvc
        reload_uvc_driver()
        self.uvc_devices = uvc.device_list()
        self.uid_map = {dev["uid"]: dev for dev in self.uvc_devices}
        # all video devices
        self.video_paths = self._list_video_paths()
        # realsense
        if realsense_enable:
            self.rs_serial_numbers = self._list_realsense_serial_numbers()
            self.rs_video_paths = self._list_realsense_video_paths()
            self.rs_rgb_video_paths = [p for p in self.rs_video_paths if self._is_like_rgb(p)]
        else:
            self.rs_serial_numbers = []
            self.rs_video_paths = []
            self.rs_rgb_video_paths = []
        # rgb & uvc
        self.uvc_rgb_video_paths = self._list_uvc_rgb_video_paths()
        self.uvc_rgb_video_ids = [int(v.replace("/dev/video", "")) for v in self.uvc_rgb_video_paths]
        self.uvc_rgb_physical_paths = [self._get_ppath_from_vpath(v) for v in self.uvc_rgb_video_paths]
        self.uvc_rgb_uids = [self._get_uid_from_ppath(p) for p in self.uvc_rgb_physical_paths]
        self.uvc_rgb_dev_info = [self.uid_map.get(uid) for uid in self.uvc_rgb_uids]
        self.uvc_rgb_serial_numbers = [dev_info.get("serialNumber") if dev_info else None for dev_info in self.uvc_rgb_dev_info]
        # all uvc cameras
        self.uvc_rgb_cameras = {}
        for vpath, vid, ppath, uid, dev_info, sn in zip(
            self.uvc_rgb_video_paths,
            self.uvc_rgb_video_ids,
            self.uvc_rgb_physical_paths,
            self.uvc_rgb_uids,
            self.uvc_rgb_dev_info,
            self.uvc_rgb_serial_numbers,
        ):
            self.uvc_rgb_cameras[vpath] = {
                "video_id": vid,
                "physical_path": ppath,
                "uid": uid,
                "dev_info": dev_info,
                "serial_number": sn
            }
        if self.verbose:
            self.info()

    # utils
    def _list_video_paths(self):
        base = "/sys/class/video4linux/"
        if not os.path.exists(base):
            return []
        return [f"/dev/{x}" for x in sorted(os.listdir(base)) if x.startswith("video")]

    def _list_uvc_rgb_video_paths(self):
        return [p for p in self.video_paths if self._is_like_rgb(p) and p not in self.rs_video_paths]

    def _list_realsense_video_paths(self):
        def _read_text(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
            except Exception:
                return None

        def _parent_usb_device_sysdir(video_sysdir):
            d = os.path.realpath(os.path.join(video_sysdir, "device"))
            for _ in range(10):
                if d is None or d == "/" or not os.path.isdir(d):
                    break
                id_vendor = _read_text(os.path.join(d, "idVendor"))
                id_product = _read_text(os.path.join(d, "idProduct"))
                if id_vendor and id_product:
                    return d
                d_next = os.path.dirname(d)
                if d_next == d:
                    break
                d = d_next
            return None

        ports = []
        for devnode in sorted(glob.glob("/dev/video*")):
            sysdir = f"/sys/class/video4linux/{os.path.basename(devnode)}"
            name = _read_text(os.path.join(sysdir, "name"))
            usb_dir = _parent_usb_device_sysdir(sysdir)
            vendor_id = _read_text(os.path.join(usb_dir, "idVendor")) if usb_dir else None

            # Match RealSense by name and Intel vendor ID
            if name and "realsense" in name.lower() and (vendor_id or "").lower() in ("8086", "32902"):
                ports.append(devnode)

        return ports
    
    def get_realsense_module(self) -> object:
        try:
            import pyrealsense2 as rs
            return rs
        except ImportError:
            arch = platform.machine()
            system = platform.system()
            print(f"[RealSense] Platform: {system} / {arch}")

            if system == "Linux" and arch.startswith("aarch64"):
                # Jetson NX / arm64
                msg = (
                    "[RealSense] pyrealsense2 not installed. please build from source:\n"
                    "    cd ~\n"
                    "    git clone https://github.com/IntelRealSense/librealsense.git\n"
                    "    cd librealsense\n"
                    "    git checkout v2.50.0\n"
                    "    mkdir build && cd build\n"
                    "    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3)\n"
                    "    make -j$(nproc)\n"
                    "    sudo make install\n"
                )
            else:
                # x86/x64
                msg = (
                    "[RealSense] pyrealsense2 not installed. You can try:\n"
                    "    pip install pyrealsense2\n"
                )
            raise RuntimeError(msg)

    def _list_realsense_serial_numbers(self):
        rs = self.get_realsense_module()
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = []
        for dev in devices:
            try:
                serials.append(dev.get_info(rs.camera_info.serial_number))
            except Exception:
                continue
        return serials

    def _get_ppath_from_vpath(self, video_path):
        sysfs_path = f"/sys/class/video4linux/{os.path.basename(video_path)}/device"
        return os.path.realpath(sysfs_path)

    def _get_uid_from_ppath(self, physical_path):
        def read_file(path):
            return open(path).read().strip() if os.path.exists(path) else None

        busnum_file = os.path.join(physical_path, "busnum")
        devnum_file = os.path.join(physical_path, "devnum")

        if not (os.path.exists(busnum_file) and os.path.exists(devnum_file)):
            parent = os.path.dirname(physical_path)
            busnum_file = os.path.join(parent, "busnum")
            devnum_file = os.path.join(parent, "devnum")

        if os.path.exists(busnum_file) and os.path.exists(devnum_file):
            bus = read_file(busnum_file)
            dev = read_file(devnum_file)
            return f"{bus}:{dev}"
        return None

    def _is_like_rgb(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None and frame.ndim == 3 and frame.shape[2] == 3

    # --------------------------------------------------------
    # public api
    # --------------------------------------------------------
    def is_rs_serial_exist(self, serial_number):
        return str(serial_number) in self.rs_serial_numbers

    def is_vpath_exist(self, vpath):
        return vpath in self.video_paths
    
    def is_ppath_exist(self, physical_path):
        for cam in self.uvc_rgb_cameras.values():
            if cam.get("physical_path") == physical_path:
                return True
        return False
    
    def get_uid_by_sn(self, serial_number):
        matches = [
            cam for cam in self.uvc_rgb_cameras.values()
            if cam.get("serial_number") == str(serial_number)
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Multiple cameras found with serial number {serial_number}")
        return matches[0].get("uid")

    def get_uid_by_ppath(self, physical_path):
        for cam in self.uvc_rgb_cameras.values():
            if cam.get("physical_path") == physical_path:
                return cam.get("uid")
        return None
    
    def get_uid_by_vpath(self, video_path):
        cam = self.uvc_rgb_cameras.get(video_path)
        if cam:
            return cam.get("uid")
        return None
    
    def get_vpath_by_sn(self, serial_number):
        matches = []
        for cam in self.uvc_rgb_cameras.values():
            if cam.get("serial_number") == str(serial_number):
                vpath = f"/dev/video{cam.get('video_id')}"
                matches.append(vpath)
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Multiple video devices found for serial number {serial_number}: {matches}. ")
        return matches[0]

    def get_vpath_by_ppath(self, physical_path):
        base = "/sys/class/video4linux/"
        matches = []
        for v in os.listdir(base):
            sys_path = os.path.realpath(os.path.join(base, v, "device"))
            if sys_path == physical_path:
                vpath = f"/dev/{v}"
                if self._is_like_rgb(vpath):
                    matches.append(vpath)
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"Multiple video devices found for physical path {physical_path}: {matches}. ")
        return matches[0]
    

    def info(self):
        logger_mp.info("======================= Camera Discovery Start ==================================")
        logger_mp.info("Found video devices: %s", self.video_paths)
        logger_mp.info("Found RGB video devices: %s", self.uvc_rgb_video_paths)

        if self.rs_serial_numbers:
            logger_mp.info("----------------------- Realsense Cameras ----------------------------------")
            logger_mp.info(f"RealSense serial numbers: {self.rs_serial_numbers}")
            logger_mp.info(f"RealSense video paths: {self.rs_video_paths}")
            logger_mp.info(f"RealSense RGB-like video paths: {self.rs_rgb_video_paths}")

        for idx, (vpath, cam) in enumerate(self.uvc_rgb_cameras.items(), start=1):
            logger_mp.info("----------------------- OpenCV / UVC Camera %d -----------------------------", idx)
            logger_mp.info("video_path    : %s", vpath)
            logger_mp.info("video_id      : %s", cam.get("video_id"))
            logger_mp.info("serial_number : %s", cam.get("serial_number") or "unknown")
            logger_mp.info("physical_path : %s", cam.get("physical_path"))
            logger_mp.info("extra_info:")

            dev_info = cam.get("dev_info")
            uid = cam.get("uid")

            if dev_info:
                for k, v in dev_info.items():
                    logger_mp.info("    %s: %s", k, v)
                try:
                    cap = uvc.Capture(uid)
                    for fmt in cap.available_modes:
                        logger_mp.info("    format: %dx%d@%d %s", fmt.height, fmt.width, fmt.fps, fmt.format_name)
                    cap.close()
                    cap = None
                except Exception as e:
                    logger_mp.warning("    failed to get formats: %s", e)
            else:
                logger_mp.info("    no uvc extra info available")

        logger_mp.info("=========================== Camera Discovery End ================================")

class BaseCamera:
    def __init__(self, cam_topic, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666):
        self._ready = threading.Event()
        self._cam_topic = cam_topic
        self._img_shape = img_shape # (H, W)
        self._fps = fps
        self._enable_zmq = enable_zmq
        self._zmq_port = zmq_port
        if self._enable_zmq:
            self._zmq_buffer = zmq_msg.TripleRingBuffer()
        else:
            self._zmq_buffer = None

        self._enable_webrtc = enable_webrtc
        self._webrtc_port = webrtc_port
        if self._enable_webrtc:
            self._webrtc_buffer = webrtc_msg.TripleRingBuffer()
        else:
            self._webrtc_buffer = None

    def __str__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return self.__str__()

    def _update_frame(self):
        """Return a jepg frame as bytes, and a bgr frame as numpy array"""
        raise NotImplementedError
    
    def wait_until_ready(self, timeout=None):
        """Block until the camera is ready (first frame is available) or timeout occurs."""
        return self._ready.wait(timeout=timeout)

    def enable_webrtc(self):
        return self._enable_webrtc
    
    def enable_zmq(self):
        return self._enable_zmq

    def get_jpeg_bytes(self):
        jpeg_bytes = self._zmq_buffer.read() if self._enable_zmq and self._zmq_buffer else None
        return jpeg_bytes

    def get_bgr_frame(self):
        bgr_numpy = self._webrtc_buffer.read() if self._enable_webrtc and self._webrtc_buffer else None
        return bgr_numpy

    def get_depth_frame(self):
        """Return a depth frame as bytes, or None if not supported. 
           Before call this function, must first call get_frame() to update the latest depth data."""
        return None

    def get_point_cloud(self):
        """Return a point cloud as bytes, or None if not supported.
           Before call this function, must first call get_frame() to update the latest point cloud data."""
        return None

    def get_zmq_port(self):
        """Return the zmq port number the camera is serving on."""
        return self._zmq_port
    
    def get_webrtc_port(self):
        """Return the webrtc port number the camera is serving on."""
        return self._webrtc_port

    def get_fps(self):
        """Return the camera FPS setting."""
        return self._fps

    def release(self):
        """Release camera resources."""
        raise NotImplementedError

class RealSenseCamera(BaseCamera):
    def __init__(self, cam_topic, serial_number, img_shape, fps, 
                 enable_zmq=True, zmq_port = 55555, enable_webrtc=False, webrtc_port=66666, enable_depth=False):
        rs = self.check_pyrealsense2_install()
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port)
        self._serial_number = serial_number
        self._enable_depth = enable_depth
        self._latest_depth = None
        try:
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self._serial_number)

            config.enable_stream(rs.stream.color, self._img_shape[1], self._img_shape[0], rs.format.bgr8, self._fps)
            if self._enable_depth:
                config.enable_stream(rs.stream.depth, self._img_shape[1], self._img_shape[0], rs.format.z16, self._fps)

            profile = self.pipeline.start(config)
            self._device = profile.get_device()
            if self._device is None:
                logger_mp.error('[RealSenseCamera] pipe_profile.get_device() is None .')
            if self._enable_depth:
                assert self._device is not None
                depth_sensor = self._device.first_depth_sensor()
                self.g_depth_scale = depth_sensor.get_depth_scale()

            self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            logger_mp.info(str(self))
        except Exception as e:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except:
                    pass
            raise RuntimeError(f"[RealSenseCamera] Failed to initialize RealSense camera {self._serial_number}: {e}")

    def __str__(self):
        return (
            f"[RealSenseCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS.\n"
            f"ZMQ: {'enabled, zmq_port=' + str(self._zmq_port) if self.enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc_port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )

    def check_pyrealsense2_install(self):
        try:
            import pyrealsense2 as rs
            return rs
        except Exception as e:
            raise ImportError(
                "pyrealsense2 not installed. Install Intel RealSense SDK and pyrealsense2 Python bindings."
            ) from e
    
    def _update_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return None

        if self._enable_depth:   
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                self._latest_depth = np.asanyarray(depth_frame.get_data())
            else:
                self._latest_depth = None

        bgr_numpy = np.asanyarray(color_frame.get_data())

        if self._enable_webrtc:
            self._webrtc_buffer.write(bgr_numpy)

        if self.enable_zmq:
            ok, buf = cv2.imencode(".jpg", bgr_numpy)
            if ok:
                self._zmq_buffer.write(buf.tobytes())
        
        if not self._ready.is_set():
            self._ready.set()
    
    def get_depth_frame(self):
        if self._latest_depth is None:
            return None
        return self._latest_depth.tobytes()

    def release(self):
        try:
            if hasattr(self.pipeline, "stop") and getattr(self.pipeline, "_running", False):
                try:
                    self.pipeline.stop()
                except Exception as e:
                    logger_mp.warning(f"[RealSenseCamera] pipeline.stop() failed: {e}")
        except Exception:
            pass
        self.pipeline = None
        logger_mp.info(f"[RealSenseCamera] Released {self._cam_topic}")

class UVCCamera(BaseCamera):
    def __init__(self, cam_topic, uid, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666):
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port)
        self.uid = uid
        self.cap = None
        try:
            self.cap = uvc.Capture(self.uid)
        except Exception as e:
            self.cap = None
            raise RuntimeError(f"[UVCCamera] Failed to open camera {self._cam_topic}: {e}")

        try:
            self.cap.frame_mode = self._choose_mode(self.cap, width=self._img_shape[1], height=self._img_shape[0], fps=self._fps)
            logger_mp.info(str(self))
        except Exception as e:
            self.cap = None
            raise RuntimeError(f"[UVCCamera] Failed to set mode for {self._cam_topic}: {e}")

    def __str__(self):
        return (
            f"[UVCCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS, MJPG.\n"
            f"ZMQ: {'enabled, zmq port=' + str(self._zmq_port) if self.enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )

    def _choose_mode(self, cap, width=None, height=None, fps=None):
        for m in cap.available_modes:
            if m.width == width and m.height == height and m.fps == fps and m.format_name == "MJPG":
                return m
        raise ValueError("[UVCCamera] No matching uvc mode found")

    def _update_frame(self):
        if self.cap is not None:
            frame = self.cap.get_frame(timeout=50) # get_frame_robust()
            if frame is not None:
                if self.enable_zmq:
                    if frame.jpeg_buffer is not None:
                        self._zmq_buffer.write(bytes(frame.jpeg_buffer))

                if self._enable_webrtc:
                    if frame.bgr is not None:
                        self._webrtc_buffer.write(frame.bgr)

                if not self._ready.is_set():
                    self._ready.set()
            else:
                raise RuntimeError

    def release(self):
        # if usbhub is plugged out, calling stop_streaming and close may hang forever.
        # try:
        #     self.cap.stop_streaming()
        # except Exception:
        #     pass
        # try:
        #     self.cap.close()
        # except Exception:
        #     pass
        # self.cap = None
        logger_mp.info(f"[UVCCamera] Released {self._cam_topic}")

class OpenCVCamera(BaseCamera):
    def __init__(self, cam_topic, video_path, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666):
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port)
        self._video_path = video_path

        self.cap = cv2.VideoCapture(self._video_path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)

        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 5)  # 0-8, default=4
        self.cap.set(cv2.CAP_PROP_GAIN, 5)        # 0-8, default=4

        # Test if the camera can read frames BEFORE starting the reader thread
        if not self._can_read_frame():
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"[OpenCVCamera] Camera {self._cam_topic} failed to initialize or read frames.")
        
        # Start the frame reader thread
        self._current_frame = None
        self._frame_lock = threading.Lock()
        self._frame_reader_stop = threading.Event()
        self._frame_reader_thread = threading.Thread(target=self._frame_reader_loop, daemon=True)
        self._frame_reader_thread.start()

    def __str__(self):
        return (
            f"[OpenCVCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS.\n"
            f"ZMQ: {'enabled, zmq port=' + str(self._zmq_port) if self.enable_zmq else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}"
        )
        
    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def _frame_reader_loop(self):
        while not self._frame_reader_stop.is_set():
            if self.cap is not None:
                ret, bgr_numpy = self.cap.read()
                if ret and bgr_numpy is not None:
                    with self._frame_lock:
                        self._current_frame = bgr_numpy.copy()
                else:
                    time.sleep(0.01)
            else:
                break
            time.sleep(0.001)
    
    def _update_frame(self):
        """Use the latest frame from the continuous reader thread."""
        with self._frame_lock:
            bgr_numpy = self._current_frame
        
        if bgr_numpy is not None:
            if self._enable_webrtc:
                self._webrtc_buffer.write(bgr_numpy)
            if self.enable_zmq:
                ok, buf = cv2.imencode(".jpg", bgr_numpy)
                #cv2.imshow("OpenCVCamera", bgr_numpy)
                #cv2.waitKey(1)
                if ok:
                    self._zmq_buffer.write(buf.tobytes())
            if not self._ready.is_set():
                self._ready.set()
        else:
            # No frame available yet, wait a bit
            time.sleep(0.01)

    def release(self):
        self._frame_reader_stop.set()
        if self._frame_reader_thread.is_alive():
            self._frame_reader_thread.join(timeout=1.0)
        
        # Release the capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger_mp.info(f"[OpenCVCamera] Released {self._cam_topic}")

class ZedCamera(BaseCamera):
    def __init__(self, cam_topic, serial_number, img_shape, fps, 
                 enable_zmq=True, zmq_port=55555, enable_webrtc=False, webrtc_port=66666, 
                 enable_depth=False, enable_point_cloud=False,
                 depth_port=None, pointcloud_port=None):
        super().__init__(cam_topic, img_shape, fps, enable_zmq, zmq_port, enable_webrtc, webrtc_port)
        self._serial_number = serial_number
        self._enable_depth = enable_depth
        self._enable_point_cloud = enable_point_cloud
        self._latest_depth = None
        self._latest_point_cloud = None
        
        # 深度图和点云的ZMQ端口
        self._depth_port = depth_port
        self._pointcloud_port = pointcloud_port
        # 为深度图和点云创建独立的buffer
        if self._enable_depth and self._depth_port:
            self._depth_buffer = zmq_msg.TripleRingBuffer()
        else:
            self._depth_buffer = None
        if self._enable_point_cloud and self._pointcloud_port:
            self._pointcloud_buffer = zmq_msg.TripleRingBuffer()
        else:
            self._pointcloud_buffer = None

        # 初始化ZED相机
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        
        # 根据请求的分辨率设置
        # 注意：ZED双目相机，image_shape可能是拼接后的尺寸（宽度×2）
        # 需要判断单目分辨率来设置camera_resolution
        single_width = img_shape[1] // 2  # 拼接模式下单目宽度
        
        if img_shape[0] == 1080 and single_width == 1920:
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif img_shape[0] == 720 and single_width == 1280:
            init_params.camera_resolution = sl.RESOLUTION.HD720
        elif img_shape[0] == 1080 and img_shape[1] == 1920:  # 未拼接的HD1080
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif img_shape[0] == 720 and img_shape[1] == 1280:   # 未拼接的HD720
            init_params.camera_resolution = sl.RESOLUTION.HD720
        else:
            init_params.camera_resolution = sl.RESOLUTION.HD720
            logger_mp.warning(f"[ZedCamera] Unsupported resolution {img_shape}, using HD720 (1280x720)")
        
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE if (enable_depth or enable_point_cloud) else sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        
        # 如果指定了serial_number，设置它
        # 注意：serial_number必须是有效的数字，不能是 "unknown" 或 None
        if serial_number and str(serial_number).lower() not in ['none', 'null', 'unknown', '']:
            try:
                init_params.set_from_serial_number(int(serial_number))
                logger_mp.info(f"[ZedCamera] Using serial number: {serial_number}")
            except (ValueError, TypeError) as e:
                logger_mp.warning(f"[ZedCamera] Invalid serial number '{serial_number}', will use first available camera: {e}")
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"[ZedCamera] Failed to open camera {self._cam_topic}: {err}")

        # 设置相机参数：亮度和对比度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 5)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 5)
        logger_mp.info(f"[ZedCamera] Camera settings: Brightness=5, Contrast=5")

        # 3*3 camera intrinsics
        # [ fx, 0, cx ],
        # [ 0, fy, cy ],
        # [ 0, 0, 1 ]
        # 1*4 camera params [ fx, fy, cx, cy ]
        # each variable is dict with key left and right
        self._camera_intrinsics = {}
        self._camera_params = {}
        self._camera_distortion_coeffs = {}
        self._camera_params = self._get_camera_intrinsics()
        logger_mp.info(f"[ZedCamera] Camera intrinsics: {self._camera_intrinsics}")
        logger_mp.info(f"[ZedCamera] Camera params: {self._camera_params}")
        logger_mp.info(f"[ZedCamera] Camera distortion coeffs: {self._camera_distortion_coeffs}")
        
        # 创建运行时参数
        self.runtime_params = sl.RuntimeParameters()
        
        logger_mp.info(str(self))

    def _get_camera_intrinsics(self):
        """
        获取ZED左右眼相机内参
        存储格式为字典，key为'left'和'right'
        - self._camera_params: {'left': [fx, fy, cx, cy], 'right': [fx, fy, cx, cy]}
        - self._camera_intrinsics: {'left': 3x3 matrix, 'right': 3x3 matrix}
        - self._camera_distortion_coeffs: {'left': [...], 'right': [...]}
        """
        try:
            calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
            left_cam = calibration_params.left_cam
            right_cam = calibration_params.right_cam
            
            # 左眼参数
            # apriltag格式: [fx, fy, cx, cy]
            self._camera_params['left'] = np.array([left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy], dtype=np.float32)
            
            # 3x3相机内参矩阵
            self._camera_intrinsics['left'] = np.array([
                [left_cam.fx, 0.0, left_cam.cx],
                [0.0, left_cam.fy, left_cam.cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # 畸变系数
            self._camera_distortion_coeffs['left'] = np.array(left_cam.disto, dtype=np.float32)
            
            # 右眼参数
            # apriltag格式: [fx, fy, cx, cy]
            self._camera_params['right'] = np.array([right_cam.fx, right_cam.fy, right_cam.cx, right_cam.cy], dtype=np.float32)
            
            # 3x3相机内参矩阵
            self._camera_intrinsics['right'] = np.array([
                [right_cam.fx, 0.0, right_cam.cx],
                [0.0, right_cam.fy, right_cam.cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # 畸变系数
            self._camera_distortion_coeffs['right'] = np.array(right_cam.disto, dtype=np.float32)
            
            logger_mp.info(f"[ZedCamera] Left camera intrinsics: fx={left_cam.fx:.2f}, fy={left_cam.fy:.2f}, "
                          f"cx={left_cam.cx:.2f}, cy={left_cam.cy:.2f}")
            logger_mp.info(f"[ZedCamera] Right camera intrinsics: fx={right_cam.fx:.2f}, fy={right_cam.fy:.2f}, "
                          f"cx={right_cam.cx:.2f}, cy={right_cam.cy:.2f}")
            
            return self._camera_params
        except Exception as e:
            logger_mp.warning(f"[ZedCamera] Failed to get camera intrinsics: {e}")
            return None
    
    def get_camera_params(self):
        """
        返回apriltag格式的相机参数
        Returns:
            dict: {'left': [fx, fy, cx, cy], 'right': [fx, fy, cx, cy]}
        """
        return self._camera_params
    
    def get_camera_matrix(self):
        """
        返回3x3相机内参矩阵
        Returns:
            dict: {'left': 3x3 matrix, 'right': 3x3 matrix}
        """
        return self._camera_intrinsics
    
    def get_distortion_coeffs(self):
        """
        返回畸变系数
        Returns:
            dict: {'left': [...], 'right': [...]}
        """
        return self._camera_distortion_coeffs
    
    def __str__(self):
        return (
            f"[ZedCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS.\n"
            f"ZMQ: {'enabled, zmq_port=' + str(self._zmq_port) if self.enable_zmq() else 'disabled'}; "
            f"WebRTC: {'enabled, webrtc_port=' + str(self._webrtc_port) if self._enable_webrtc else 'disabled'}; "
            f"Depth: {'enabled, depth_port=' + str(self._depth_port) if self._enable_depth else 'disabled'}; "
            f"PointCloud: {'enabled, pc_port=' + str(self._pointcloud_port) if self._enable_point_cloud else 'disabled'}; "
        )
    
    def _update_frame(self):
        """采集图像、深度图和点云数据"""
        if self.zed is None:
            return
        
        # 采集新帧
        start_time = time.time()
        logger_mp.info(f"[ZedCamera] Grabbing new frame")
        err = self.zed.grab(self.runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            return
    
        sbs_image = sl.Mat()
        self.zed.retrieve_image(sbs_image, sl.VIEW.SIDE_BY_SIDE)
        sbs_data = sbs_image.get_data()
        
        # ZED SDK retrieve_image()可能返回BGRA（4通道）或BGR（3通道），取决于SDK版本和配置
        # 检查并转换为BGR格式（如果需要）
        if sbs_data.shape[2] == 4:  # BGRA格式
            bgr_numpy = cv2.cvtColor(sbs_data, cv2.COLOR_BGRA2BGR)
        else:  # 已经是BGR格式
            bgr_numpy = sbs_data
        
        # 获取深度图
        if self._enable_depth:
            depth = sl.Mat()
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_numpy = depth.get_data()
            
            self._latest_depth = depth_numpy
            
            # 将深度图写入buffer (序列化为bytes)
            if self._depth_buffer:
                self._depth_buffer.write(depth_numpy.tobytes())
        
        # 获取点云
        if self._enable_point_cloud:
            point_cloud = sl.Mat()
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            point_cloud_numpy = point_cloud.get_data()  # Shape: (H, W, 4)
            
            # 注意：点云不应该resize！保持原始分辨率以保持3D坐标准确性
            # point_cloud_numpy[:,:,0:3] = XYZ坐标（米）
            # point_cloud_numpy[:,:,3] = RGBA颜色（打包的float32）
            self._latest_point_cloud = point_cloud_numpy
            
            # 将点云写入buffer (序列化为bytes)
            # 传输格式: (H, W, 4) 的 float32 数组
            if self._pointcloud_buffer:
                self._pointcloud_buffer.write(point_cloud_numpy.tobytes())
        
        # 更新RGB缓冲区
        if self._enable_webrtc:
            self._webrtc_buffer.write(bgr_numpy)
        
        end_time = time.time()
        logger_mp.info(f"[ZedCamera] Time taken to update frame: {end_time - start_time} seconds, fps: {1.0 / (end_time - start_time)}")

        if self.enable_zmq():
            ok, buf = cv2.imencode(".jpg", bgr_numpy)
            if ok:
                self._zmq_buffer.write(buf.tobytes())
        
        if not self._ready.is_set():
            self._ready.set()

    def get_depth_frame(self):
        """返回深度图数据 (用于非ZMQ场景)"""
        if self._latest_depth is None:
            return None
        return self._latest_depth.tobytes()
    
    def get_point_cloud(self):
        """返回点云数据 (用于非ZMQ场景)"""
        if self._latest_point_cloud is None:
            return None
        return self._latest_point_cloud.tobytes()
    
    def get_depth_bytes(self):
        """从buffer获取深度图bytes (用于ZMQ传输)"""
        if self._depth_buffer is None:
            return None
        return self._depth_buffer.read()
    
    def get_pointcloud_bytes(self):
        """从buffer获取点云bytes (用于ZMQ传输)"""
        if self._pointcloud_buffer is None:
            return None
        return self._pointcloud_buffer.read()
    
    def get_depth_port(self):
        """返回深度图ZMQ端口"""
        return self._depth_port
    
    def get_pointcloud_port(self):
        """返回点云ZMQ端口"""
        return self._pointcloud_port

    def release(self):
        if self.zed is not None:
            try:
                self.zed.close()
            except Exception as e:
                logger_mp.warning(f"[ZedCamera] Failed to close: {e}")
            self.zed = None
        
        logger_mp.info(f"[ZedCamera] Released {self._cam_topic}")
# ========================================================
# image server
# ========================================================
class ImageServer:
    def __init__(self, cam_config, realsense_enable=False, camera_finder_verbose=False):
        self._cam_config = cam_config
        self._realsense_enable = realsense_enable
        self._stop_event = threading.Event()
        self._cameras: dict[str, BaseCamera] = {}
        self._cam_finder = CameraFinder(realsense_enable, camera_finder_verbose)
        self._responser = zmq_msg.Responser(self._cam_config)
        self._zmq_publisher_manager = zmq_msg.PublisherManager.get_instance()
        self._webrtc_publisher_manager = webrtc_msg.PublisherManager.get_instance()
        self._publisher_threads = []  # keep references for graceful join

        try:
            # Load cameras from self.cam_config
            for cam_topic, cam_cfg in self._cam_config.items():
                if not cam_cfg.get("enable_zmq", False) and not cam_cfg.get("enable_webrtc", False):
                    continue

                enable_zmq = cam_cfg.get("enable_zmq", False)
                zmq_port = cam_cfg.get("zmq_port", None)
                enable_webrtc = cam_cfg.get("enable_webrtc", False)
                webrtc_port = cam_cfg.get("webrtc_port", None)
                cam_type = cam_cfg.get("type", "uvc").lower()
                img_shape = cam_cfg.get("image_shape", None)
                fps = cam_cfg.get("fps", 30)
                video_id = cam_cfg.get("video_id", "0")
                video_path = f"/dev/video{video_id}" if video_id else None
                physical_path = str(cam_cfg.get("physical_path")) if cam_cfg.get("physical_path") else None
                serial_number = str(cam_cfg.get("serial_number")) if cam_cfg.get("serial_number") else None

                if cam_type == "opencv":
                    if physical_path is not None:
                        vpath = self._cam_finder.get_vpath_by_ppath(physical_path)
                        if vpath is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find OpenCVCamera for {cam_topic} with physical path {physical_path}")
                        else:
                            self._cameras[cam_topic] = OpenCVCamera(cam_topic, vpath, img_shape, fps, 
                                                                    enable_zmq, zmq_port, enable_webrtc, webrtc_port)
                            continue

                    if serial_number is not None:
                        vpath = self._cam_finder.get_vpath_by_sn(serial_number)
                        if vpath is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find OpenCVCamera for {cam_topic} with serial number {serial_number}")
                        else:
                            self._cameras[cam_topic] = OpenCVCamera(cam_topic, vpath, img_shape, fps, 
                                                                    enable_zmq, zmq_port, enable_webrtc, webrtc_port)
                        # once you specify either `physical_path` or `serial_number`, the system will no longer fall back to searching by `video_id`.
                        # ——— even if no camera matches the given path/serial.
                        continue
                    
                    if not self._cam_finder.is_vpath_exist(video_path):
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Cannot find OpenCVCamera for {cam_topic} with video_id {video_id}")
                    else:
                        self._cameras[cam_topic] = OpenCVCamera(cam_topic, video_path, img_shape, fps,
                                                                enable_zmq, zmq_port, enable_webrtc, webrtc_port)
                        

                elif cam_type == "realsense":
                    if not self._realsense_enable:
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Please start image server with the '--rs' flag to support Realsense {cam_topic}.")
                    elif not self._cam_finder.is_rs_serial_exist(serial_number):
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Cannot find RealSenseCamera for {cam_topic}")
                    else:
                        self._cameras[cam_topic] = RealSenseCamera(cam_topic, serial_number, img_shape, fps,
                                                                   enable_zmq, zmq_port, enable_webrtc, webrtc_port)

                elif cam_type == "zed":
                    enable_depth = cam_cfg.get("enable_depth", False)
                    enable_point_cloud = cam_cfg.get("enable_point_cloud", False)
                    depth_port = cam_cfg.get("depth_port", None)
                    pointcloud_port = cam_cfg.get("pointcloud_port", None)
                    try:
                        zed_camera = ZedCamera(
                            cam_topic, serial_number, img_shape, fps,
                            enable_zmq, zmq_port, enable_webrtc, webrtc_port,
                            enable_depth, enable_point_cloud,
                            depth_port, pointcloud_port
                        )
                        self._cameras[cam_topic] = zed_camera
                        
                        # 将相机内参添加到cam_config中，以便client可以获取
                        camera_params = zed_camera.get_camera_params()
                        camera_matrix = zed_camera.get_camera_matrix()
                        distortion_coeffs = zed_camera.get_distortion_coeffs()
                        
                        if camera_params is not None:
                            # apriltag格式: {'left': [fx, fy, cx, cy], 'right': [fx, fy, cx, cy]}
                            cam_cfg['camera_params'] = {
                                'left': camera_params['left'].tolist(),
                                'right': camera_params['right'].tolist()
                            }
                        if camera_matrix is not None:
                            # 3x3相机矩阵: {'left': 3x3 matrix, 'right': 3x3 matrix}
                            cam_cfg['camera_matrix'] = {
                                'left': camera_matrix['left'].tolist(),
                                'right': camera_matrix['right'].tolist()
                            }
                        if distortion_coeffs is not None:
                            # 畸变系数: {'left': [...], 'right': [...]}
                            cam_cfg['distortion_coeffs'] = {
                                'left': distortion_coeffs['left'].tolist(),
                                'right': distortion_coeffs['right'].tolist()
                            }
                            
                        logger_mp.info(f"[Image Server] Added camera intrinsics (left & right) to config for {cam_topic}")
                    except Exception as e:
                        self._cameras[cam_topic] = None
                        logger_mp.error(f"[Image Server] Failed to initialize ZedCamera for {cam_topic}: {e}")

                elif cam_type == "uvc":
                    uid = None
                    if physical_path is not None:
                        uid = self._cam_finder.get_uid_by_ppath(physical_path)
                        if uid is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find UVCCamera for {cam_topic} with physical path {physical_path}")
                        else:
                            self._cameras[cam_topic] = UVCCamera(cam_topic, uid, img_shape, fps, 
                                                                 enable_zmq, zmq_port, enable_webrtc, webrtc_port)
                            continue

                    if serial_number is not None:
                        uid = self._cam_finder.get_uid_by_sn(serial_number)
                        if uid is None:
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find UVCCamera for {cam_topic} with serial number {serial_number}")
                        else:
                            self._cameras[cam_topic] = UVCCamera(cam_topic, uid, img_shape, fps, 
                                                                 enable_zmq, zmq_port, enable_webrtc, webrtc_port)
                        # once you specify either `physical_path` or `serial_number`, the system will no longer fall back to searching by `video_id`.
                        # ——— even if no camera matches the given path/serial.
                        continue

                    if video_id is not None:
                        if not self._cam_finder.is_vpath_exist(video_path):
                            self._cameras[cam_topic] = None
                            logger_mp.error(f"[Image Server] Cannot find UVCCamera for {cam_topic} with video_id {video_id}")
                        else:
                            uid = self._cam_finder.get_uid_by_vpath(video_path)
                            if uid is None:
                                self._cameras[cam_topic] = None
                                logger_mp.error(f"[Image Server] Cannot find UVCCamera for {cam_topic} with uid from video_id {video_id}")
                            else:
                                self._cameras[cam_topic] = UVCCamera(cam_topic, uid, img_shape, fps, 
                                                                     enable_zmq, zmq_port, enable_webrtc, webrtc_port)
                else:
                    logger_mp.error(f"[Image Server] Unknown camera type {cam_type} for {cam_topic}, skipping...")
                    continue
        except Exception as e:
            logger_mp.error(f"[Image Server] Initialization failed: {e}")
            self._clean_up()
            raise

        logger_mp.info("[Image Server] Image server has started, waiting for client connections...")

    def _update_frames(self, cam_topic: str, camera: BaseCamera):
        try:
            interval = 1.0 / camera.get_fps()
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    camera._update_frame()
                except Exception as e:
                    logger_mp.error(f"[Image Server] Error updating frame for {cam_topic} camera")
                    self._stop_event.set()
                    break
                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Failed to update frames for {cam_topic} camera: {e}")
            self._stop_event.set()

    def _zmq_pub(self, cam_topic: str, camera: BaseCamera):
        try:
            interval = 1.0 / camera.get_fps()
            next_frame_time = time.monotonic()

            while not self._stop_event.is_set():
                # 发布RGB图像
                jpeg_bytes = camera.get_jpeg_bytes()
                if jpeg_bytes is not None:
                    self._zmq_publisher_manager.publish(jpeg_bytes, camera.get_zmq_port())
                else:
                    logger_mp.warning(f"[Image Server] {cam_topic} returned no frame.")
                    self._stop_event.set()
                    break
                
                # 如果是ZedCamera，同时发布深度图和点云（同一帧数据）
                if isinstance(camera, ZedCamera):
                    # 发布深度图
                    if hasattr(camera, 'get_depth_port') and camera.get_depth_port():
                        depth_bytes = camera.get_depth_bytes()
                        if depth_bytes is not None:
                            self._zmq_publisher_manager.publish(depth_bytes, camera.get_depth_port())
                    
                    # 发布点云
                    if hasattr(camera, 'get_pointcloud_port') and camera.get_pointcloud_port():
                        pc_bytes = camera.get_pointcloud_bytes()
                        if pc_bytes is not None:
                            self._zmq_publisher_manager.publish(pc_bytes, camera.get_pointcloud_port())

                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Failed to publish zmq data from {cam_topic} camera: {e}")
            self._stop_event.set()
    
    def _webrtc_pub(self, cam_topic: str, camera: BaseCamera):
        try:
            interval = 1.0 / camera.get_fps()
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                bgr_frame = camera.get_bgr_frame()

                if bgr_frame is not None:
                    self._webrtc_publisher_manager.publish(bgr_frame, camera.get_webrtc_port())
                else:
                    logger_mp.info(f"[Image Server] {cam_topic} returned no frame.")
                    self._stop_event.set()
                    break

                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Failed to publish rtc frame from {cam_topic} camera.")
            self._stop_event.set()

    def _clean_up(self):
        self._responser.stop()
        for t in self._publisher_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._publisher_threads.clear()
        
        try:
            self._zmq_publisher_manager.close()
        except Exception:
            pass
        try:
            self._webrtc_publisher_manager.close()
        except Exception:
            pass

        for cam in self._cameras.values():
            if cam:
                try:
                    cam.release()
                except Exception as e:
                    logger_mp.error(f"[Image Server] Error releasing camera {cam._cam_topic}: {e}")
        logger_mp.info("[Image Server] Clean up completed. Server stopped.")

    # --------------------------------------------------------
    # public api
    # --------------------------------------------------------
    def start(self):
        for camera_topic, camera in self._cameras.items():
            if camera is None:
                logger_mp.error(f"[Image Server] Camera {camera_topic} failed to initialize previously, cannot start.")
                self._stop_event.set()
                self._clean_up()
                return
            t = threading.Thread(target=self._update_frames, args=(camera_topic, camera), daemon=True)
            t.start()
            self._publisher_threads.append(t)
        
        for camera_topic, camera in self._cameras.items():
            ready = camera.wait_until_ready(timeout=5.0)
            if not ready:
                logger_mp.error(f"[Image Server] {camera_topic} ready timeout.")
                self._stop_event.set()
                self._clean_up()
            logger_mp.info(f"[Image Server] {camera_topic} is ready.")
        
        for camera_topic, camera in self._cameras.items():
            if camera.enable_webrtc():
                t = threading.Thread(target=self._webrtc_pub, args=(camera_topic, camera), daemon=True)
                t.start()
                self._publisher_threads.append(t)

            if camera.enable_zmq():
                t = threading.Thread(target=self._zmq_pub, args=(camera_topic, camera), daemon=True)
                t.start()
                self._publisher_threads.append(t)
                
                # 日志记录ZedCamera的额外端口
                if isinstance(camera, ZedCamera):
                    ports_info = f"RGB port={camera.get_zmq_port()}"
                    if hasattr(camera, 'get_depth_port') and camera.get_depth_port():
                        ports_info += f", Depth port={camera.get_depth_port()}"
                    if hasattr(camera, 'get_pointcloud_port') and camera.get_pointcloud_port():
                        ports_info += f", PointCloud port={camera.get_pointcloud_port()}"
                    logger_mp.info(f"[Image Server] Started ZMQ publisher for {camera_topic} ({ports_info})")

    def wait(self):
        self._stop_event.wait()
        self._clean_up()

    def stop(self):
        self._stop_event.set()

# ========================================================
# utility functions
# ========================================================
def signal_handler(server, signum, frame):
    logger_mp.info(f"[Image Server] Received signal {signum}, initiating graceful shutdown...")
    server.stop()

def main():
    logger_mp.info(
        "\n====================== Image Server Startup Guide ======================\n"
        "Please first read this repo's README.md to learn how to configure and use the teleimager.\n"
        "To discover connected cameras, run the following command:\n"
        "\n"
        "    teleimager-server --cf\n"
        "\n"
        "The '--cf' flag means 'camera find'.\n"
        "This will list all detected cameras and their details (video paths, serial numbers and physical path etc.).\n"
        "Use that information to fill in your 'cam_config_server.yaml' file.\n"
        "Once configured, you can start the image server with:\n"
        "\n"
        "    teleimager-server\n"
        "\n"
        "Note:\n"
        " - If you have RealSense cameras, add the '--rs' flag to enable RealSense support.\n"
        " - Make sure you have proper permissions to access the camera devices (e.g., run with sudo or set udev rules).\n"
        "=========================================================================="
    )

    # command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', action = 'store_true', help = 'Enable camera found mode, print all connected cameras info')
    parser.add_argument('--rs', action = 'store_true', help = 'Enable RealSense camera mode. Otherwise only find UVC/OpenCV cameras.')
    args = parser.parse_args()

    # if enable camera finder mode, just print cameras info and exit
    if args.cf:
        CameraFinder(realsense_enable=args.rs, verbose=True)
        exit(0)

    # Load config file, start image server
    try:
        with open(CONFIG_PATH, "r") as f:
            cam_config = yaml.safe_load(f)
    except Exception as e:
        logger_mp.error(f"Failed to load configuration file at {CONFIG_PATH}: {e}")
        exit(1)

    # start image server
    server = ImageServer(cam_config, realsense_enable=args.rs, camera_finder_verbose=False)
    server.start()

    # graceful shutdown handling
    signal.signal(signal.SIGINT, functools.partial(signal_handler, server))
    signal.signal(signal.SIGTERM, functools.partial(signal_handler, server))

    logger_mp.info("[Image Server] Running... Press Ctrl+C to exit.")
    server.wait()

    # usbhub plugout may cause block process exit, no better solution for now
    time.sleep(0.5)
    os.killpg(os.getpgrp(), 9)

if __name__ == "__main__":
    main()
