import pyzed.sl as sl

# 创建相机对象
zed = sl.Camera()

# 设置初始化参数
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720

# 打开相机
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"相机打开失败: {err}")
    exit(1)

# 获取相机标定参数
calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

# 左相机内参
left_cam = calibration_params.left_cam
print("=== 左相机内参 ===")
print(f"fx: {left_cam.fx}")
print(f"fy: {left_cam.fy}")
print(f"cx: {left_cam.cx}")
print(f"cy: {left_cam.cy}")
print(f"畸变系数: {left_cam.disto}")

# 右相机内参
right_cam = calibration_params.right_cam
print("\n=== 右相机内参 ===")
print(f"fx: {right_cam.fx}")
print(f"fy: {right_cam.fy}")
print(f"cx: {right_cam.cx}")
print(f"cy: {right_cam.cy}")
print(f"畸变系数: {right_cam.disto}")

# 基线距离
print(f"\n基线距离 (baseline): {calibration_params.get_camera_baseline()} mm")

zed.close()
