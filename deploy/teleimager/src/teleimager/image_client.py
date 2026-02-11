import cv2
import time
import os
from . import zmq_msg
import numpy as np
import logging_mp
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

class ImageClient:
    def __init__(self, host="192.168.123.164", request_port=60000):
        """
        Args:
            server_address:   IP address of image host server
            requset_port:     Port for request camera configuration
        """
        self._host = host
        self._request_port = request_port

        # subscriber and requester setup
        self._subscriber_manager = zmq_msg.SubscriberManager.get_instance()
        self._requester  = zmq_msg.Requester(self._host, self._request_port)
        self._cam_config = self._requester.request()

        if self._cam_config is None:
            raise RuntimeError("Failed to get camera configuration.")
        
        if "head_camera" in self._cam_config:
            if self._cam_config['head_camera']['enable_zmq']:
                self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['zmq_port'])
            if not self._cam_config['head_camera']['enable_zmq'] and not self._cam_config['head_camera']['enable_webrtc']:
                logger_mp.warning("[Image Client] NOTICE! Head camera is not enabled on both ZMQ and WebRTC.")

            if "enable_depth" in self._cam_config['head_camera']:
                if self._cam_config['head_camera']['enable_depth']:
                    self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['depth_port'], decode_jpeg=False)
            if "enable_point_cloud" in self._cam_config['head_camera']:
                if self._cam_config['head_camera']['enable_point_cloud']:
                    self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['pointcloud_port'], decode_jpeg=False)
            
        if "left_wrist_camera" in self._cam_config:
            if self._cam_config['left_wrist_camera']['enable_zmq']:
                self._subscriber_manager.subscribe(self._host, self._cam_config['left_wrist_camera']['zmq_port'])
        if "right_wrist_camera" in self._cam_config:
            if self._cam_config['right_wrist_camera']['enable_zmq']:
                self._subscriber_manager.subscribe(self._host, self._cam_config['right_wrist_camera']['zmq_port'])


    # --------------------------------------------------------
    # public api
    # --------------------------------------------------------
    def get_cam_config(self):
        return self._cam_config

    def get_head_frame(self):
        return self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['zmq_port'])
    
    def get_head_info(self):
        # 0 is data, 1 is fps
        img = self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['zmq_port'])[0]
        depth = None
        if "enable_depth" in self._cam_config['head_camera']:
            if self._cam_config['head_camera']['enable_depth']:
                depth = self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['depth_port'], decode_jpeg=False)[0]
        pointcloud = None
        if "enable_point_cloud" in self._cam_config['head_camera']:
            if self._cam_config['head_camera']['enable_point_cloud']:
                pointcloud = self._subscriber_manager.subscribe(self._host, self._cam_config['head_camera']['pointcloud_port'], decode_jpeg=False)[0]
        return img, depth, pointcloud

    def get_left_wrist_frame(self):
        return self._subscriber_manager.subscribe(self._host, self._cam_config['left_wrist_camera']['zmq_port'])
    
    def get_right_wrist_frame(self):
        return self._subscriber_manager.subscribe(self._host, self._cam_config['right_wrist_camera']['zmq_port'])
        
    def close(self):
        self._subscriber_manager.close()
        logger_mp.info("Image client has been closed.")

    def decode_pointcloud(self, pointcloud_bytes):
        '''
        input  poitncloud are (H, W, 4) float32 array, where 4 are xyz+rgba
        output pointcluod should be (H,W,7) 7 are XYZRGBA, respectively
        '''
        start_time = time.time()
        # 获取相机配置
        cam_config = self._cam_config['head_camera']
        h = cam_config['image_shape'][0]
        w = cam_config['image_shape'][1] // 2  # 左图宽度
        
        # 转换bytes为numpy数组 (H, W, 4) - XYZRGBA格式
        pc_array = np.frombuffer(pointcloud_bytes, dtype=np.float32).reshape(h, w, 4)
        
        # 创建输出数组 (H, W, 7) - XYZ + RGBA分离
        pc_decoded = np.zeros((h, w, 7), dtype=np.float32)
        
        # 复制XYZ坐标
        pc_decoded[:, :, :3] = pc_array[:, :, :3]
        
        # 向量化解码RGBA颜色（官方struct方法的高性能版本）
        # 将float32重新解释为uint32，然后提取RGBA分量
        rgba_packed = pc_array[:, :, 3]
        rgba_uint = rgba_packed.view(np.uint32)
        
        # 使用位运算向量化提取RGBA（与struct.unpack等价，但快1000倍）
        pc_decoded[:, :, 3] = (rgba_uint & 0xFF) / 255.0         # R
        pc_decoded[:, :, 4] = ((rgba_uint >> 8) & 0xFF) / 255.0  # G
        pc_decoded[:, :, 5] = ((rgba_uint >> 16) & 0xFF) / 255.0 # B
        pc_decoded[:, :, 6] = ((rgba_uint >> 24) & 0xFF) / 255.0 # A
        
        # 调试信息
        logger_mp.debug(f"Point cloud decoded: shape={pc_decoded.shape}, "
                       f"XYZ range=[{np.nanmin(pc_decoded[:,:,:3]):.2f}, {np.nanmax(pc_decoded[:,:,:3]):.2f}], "
                       f"RGB range=[{np.nanmin(pc_decoded[:,:,3:6]):.2f}, {np.nanmax(pc_decoded[:,:,3:6]):.2f}]")
        end_time = time.time()
        logger_mp.info(f"Point cloud decoded time: {end_time - start_time:.2f} seconds")
        
        return pc_decoded


def main():
    # command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='192.168.123.167', help='IP address of image server')
    parser.add_argument('--no-pointcloud', action='store_true', help='Disable point cloud visualization')
    parser.add_argument('--save-interval', type=int, default=30, help='Save point cloud every N frames (0 to disable)')
    parser.add_argument('--save-dir', type=str, default='/home/mani2/lxk_code/xr_teleoperate/teleop/teleimager/saved_pointclouds', 
                        help='Directory to save point cloud files')
    args = parser.parse_args()

    # Example usage with three camera streams
    client = ImageClient(host=args.host)
    cam_config = client.get_cam_config()

    # 点云保存相关
    frame_counter = 0
    save_interval = args.save_interval
    save_dir = args.save_dir
    if save_interval > 0:
        os.makedirs(save_dir, exist_ok=True)
        logger_mp.info(f"[Point Cloud] Saving every {save_interval} frames to: {save_dir}")
    else:
        logger_mp.info("[Point Cloud] Auto-save disabled (--save-interval 0)")

    running = True
    while running:
        if "head_camera" in cam_config:
            if cam_config['head_camera']['enable_zmq']:
                #head_img, head_fps = client.get_head_frame()
                head_img, head_depth_bytes, head_pointcloud_bytes = client.get_head_info()
                if head_img is not None:
                    start_time = time.time()
                    #logger_mp.info(f"head_img shape: {head_img.shape}")
                    #logger_mp.info(f"Head Camera FPS: {head_fps:.2f}")
                    #logger_mp.info(f"Head Camera Shape: {cam_config['head_camera']['image_shape']}")
                    cv2.imshow("Head Camera", head_img)
                    #cv2.imwrite(os.path.join(save_dir, f"head_img_frame_{frame_counter:06d}.jpg"), head_img)
                    
                    # 显示深度图
                    if head_depth_bytes is not None:
                        # 转换bytes为numpy数组
                        head_depth = np.frombuffer(head_depth_bytes, dtype=np.float32)
                        #logger_mp.info(f"head_depth shape: {head_depth.shape}")
                        h = cam_config['head_camera']['image_shape'][0]
                        w = cam_config['head_camera']['image_shape'][1] // 2  # 左图宽度
                        head_depth = head_depth.reshape(h, w)
                        #logger_mp.info(f"head_depth reshape shaped shape: {head_depth.shape}")
                        # 可视化深度图
                        depth_vis = np.nan_to_num(head_depth, nan=10.0)
                        depth_vis = np.clip(depth_vis, 0, 10.0)
                        depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
                        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        #cv2.imshow("Head Depth", depth_colormap)
                        #cv2.imwrite(os.path.join(save_dir, f"depth_frame_{frame_counter:06d}.jpg"), depth_colormap)
                    
                    if head_pointcloud_bytes is not None:
                        pointcloud = np.frombuffer(head_pointcloud_bytes, dtype=np.float32)
                        #logger_mp.info(f"pointcloud shape: {pointcloud.shape}")
                        pointcloud = pointcloud.reshape(cam_config['head_camera']['image_shape'][0], cam_config['head_camera']['image_shape'][1]//2, 4)
                        #logger_mp.info(f"pointcloud reshape shape: {pointcloud.shape}")
                        #np.save(os.path.join(save_dir, f"pointcloud_frame_{frame_counter:06d}.npy"), pointcloud)

                    end_time = time.time()
                    #logger_mp.info(f"Head Camera Save and show time: {end_time - start_time:.2f} seconds")
        if "left_wrist_camera" in cam_config:
            if cam_config['left_wrist_camera']['enable_zmq']:
                left_wrist_img, left_wrist_fps = client.get_left_wrist_frame()
                if left_wrist_img is not None:
                    logger_mp.info(f"Left Wrist Camera FPS: {left_wrist_fps:.2f}")
                    logger_mp.debug(f"Left Wrist Camera Shape: {cam_config['left_wrist_camera']['image_shape']}")
                    cv2.imshow("Left Wrist Camera", left_wrist_img)

        if "right_wrist_camera" in cam_config:
            if cam_config['right_wrist_camera']['enable_zmq']:
                right_wrist_img, right_wrist_fps = client.get_right_wrist_frame()
                if right_wrist_img is not None:
                    logger_mp.info(f"Right Wrist Camera FPS: {right_wrist_fps:.2f}")
                    logger_mp.debug(f"Right Wrist Camera Shape: {cam_config['right_wrist_camera']['image_shape']}")
                    cv2.imshow("Right Wrist Camera", right_wrist_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger_mp.info("Exiting image client on user request.")
            running = False
            # clean up
            client.close()
            cv2.destroyAllWindows()
            if vis is not None:
                vis.destroy_window()
                logger_mp.info("[Image Client] Open3D visualizer closed")
        # Small delay to prevent excessive CPU usage
        frame_counter += 1
        time.sleep(0.002)

if __name__ == "__main__":
    main()