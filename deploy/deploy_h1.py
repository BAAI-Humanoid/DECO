import os
import sys
import time
import cv2
import math
import json
import yaml
import torch
import argparse
import numpy as np
import threading
from multiprocessing import Lock, Array, Lock
import logging_mp
logger_mp = logging_mp.get_logger(__name__)
logger_mp.setLevel(logging_mp.INFO)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleimager.image_client import ImageClient
from deploy.robot_control.robot_arm_ik import H1_2_ArmIK
from deploy.robot_control.robot_arm import H1_2_ArmController
from deploy.robot_control.robot_hand_inspire_deploy import Inspire_Controller_FTP
from deploy.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from deploy.dynamixel.active_cam import DynamixelAgent, DynamixelRobotConfig
from inference import predict_action, modeling, ACTTemporalEnsembler

from sshkeyboard import listen_keyboard, stop_listening

def load_episode_actions(json_path):
    """
    read episode actions from json file
    and flatten it to [left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6) + head_cam(2)] 28 dimensions
    """
    # episode_dir = os.path.join(data_dir, f"episode_{str(episode_id).zfill(4)}")
    # json_path = os.path.join(episode_dir, "data.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Episode json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        episode_data = json.load(f)

    data_items = episode_data.get("data", [])
    data_items = sorted(data_items, key=lambda x: x.get("idx", 0))

    actions_seq = []
    for item in data_items:
        act_dict = item.get("actions", {})
        left_arm = act_dict.get("left_arm", {}).get("qpos", [])
        right_arm = act_dict.get("right_arm", {}).get("qpos", [])
        left_ee = act_dict.get("left_ee", {}).get("qpos", [])
        right_ee = act_dict.get("right_ee", {}).get("qpos", [])
        head = act_dict.get("head", {}).get("qpos", [])
        # 預期長度：7 + 7 + 6 + 6 + 2 = 28
        action = left_arm + right_arm + left_ee + right_ee + head
        actions_seq.append(np.array(action, dtype=np.float64))

    logger_mp.info(f"Loaded {len(actions_seq)} actions from {json_path}")
    return actions_seq

START          = False  # Enable to start robot policy
STOP           = False  # Enable to begin system exit procedure
RESETING          = False  # During RESETING, robot goes to initial position, otherwise it use the action from policy
def on_press(key):
    global STOP, START, RESETING
    if key == 'r':
        START = True
    elif key == 'q':
        STOP = True
    elif key == 's':
        RESETING = not RESETING
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def main(args):
    listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                daemon=True)
    listen_keyboard_thread.start()

    try:
        yaml_config = yaml.safe_load(open(args.yaml, 'r'))
        chunk_size = yaml_config['model']['chunk_size']
        model_name = yaml_config['model_name']
        temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensembler_alpha, chunk_size)
        temporal_ensembler.reset()
        temporal_ensembler_flag = args.temporal_ensembler
        record_video = args.record_video
        receding_horizon = False
        action_receding = []
        model = modeling(yaml_config)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        if record_video:
            videowriter = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"XVID"), 25, (1280, 720), True)


        if model_name == 'deco' or model_name == 'dp':
            n_action_select = args.select_action
            receding_horizon = True
            temporal_ensembler_flag = False

        img_client = ImageClient(host=args.img_server_ip)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        if args.motion:
            loco_wrapper = LocoClientWrapper(networkInterface=args.network_interface)
            logger_mp.info(f"运行在运动模式(Motion Mode)，下肢将保持站立")
        else:
            motion_switcher = MotionSwitcher(networkInterface=args.network_interface)
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"进入调试模式(Debug Mode): {'成功' if status == 0 else '失败'}，下肢将进入阻尼状态")

        # arm 
        arm_ik = H1_2_ArmIK()
        arm_ctrl = H1_2_ArmController(motion_mode=args.motion, networkInterface=args.network_interface)

        # end-effector
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        dual_hand_action_array[:] = [0.0] * 12
        hand_ctrl = Inspire_Controller_FTP(dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, networkInterface=args.network_interface)

        # active camera
        # first is yaw, second is pitch, no roll
        config = DynamixelRobotConfig(
            joint_ids=(1, 2),
            joint_offsets=(
                np.pi, 
                np.pi/2, 
            ),
            joint_signs=(1, 1),
            gripper_config=None,
        )
        active_cam = DynamixelAgent(port="/dev/ttyUSB0", dynamixel_config=config)
        active_cam._robot.set_torque_mode(True)
        active_cam._robot.command_joint_state([0, 0])

        ## --initialize robot states-- ##
        episode_actions = load_episode_actions(json_path=args.json_path)
        action = episode_actions[0]
        # arm
        current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
        tau = arm_ik.solve_tau(current_lr_arm_q, current_lr_arm_dq)
        action_arm = action[:14]
        arm_ctrl.ctrl_dual_arm(action_arm, tau)
        # hand
        action_hand = action[14:26]
        dual_hand_action_array[:] = action_hand
        action_head = action[26:28]

        active_cam._robot.command_joint_state(action_head)
        time.sleep(3)
        arm_ctrl.speed_gradual_max()
        index = 0

        logger_mp.info("---------------------🚀press r to start program🚀-------------------------")
        while not START:
            if STOP:
                raise Exception("Stop program")
            time.sleep(0.01)

        while not STOP:
            index += 1
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                head_img, head_img_fps = img_client.get_head_frame()
                
            ## all observations
            states = torch.zeros(28)

            img1 = head_img[:, :camera_config['head_camera']['image_shape'][1]//2]
            img2 = head_img[:, camera_config['head_camera']['image_shape'][1]//2:]  
            img_show = cv2.hconcat([img1, img2])
            cv2.imshow("Head Camera", img_show)
            cv2.waitKey(1)
            if index < 10:
                loop_time = time.time() - start_time
                time.sleep(max(0, (1 / args.fps) - loop_time))
                continue
            if record_video:
                img_record = cv2.resize(img1, (1280, 720))
                videowriter.write(img_record)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # states
            with dual_hand_data_lock:
                left_hand_state = dual_hand_state_array[:6]
                right_hand_state = dual_hand_state_array[-6:]
                states[7:13] = torch.tensor(left_hand_state)
                states[20:26] = torch.tensor(right_hand_state)
                
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            left_arm_state = current_lr_arm_q[:7].tolist()
            right_arm_state = current_lr_arm_q[-7:].tolist()
            states[:7] = torch.tensor(left_arm_state)
            states[13:20] = torch.tensor(right_arm_state)
            head_state = active_cam._robot.get_joint_state().tolist()
            states[26:28] = torch.tensor(head_state)

            # tactile
            with dual_hand_data_lock:
                left_tac = np.array(hand_ctrl.left_hand_tactile_array)
                right_tac = np.array(hand_ctrl.right_hand_tactile_array)
            
            if RESETING:
                # use first action
                temporal_ensembler.reset()
                action_receding = []
                initial_action = episode_actions[0]
                # left arm
                action[:7] = initial_action[:7]
                # left hand
                action[7:13] = initial_action[14:20]
                # right arm
                action[13:20] = initial_action[7:14]
                # right hand
                action[20:26] = initial_action[20:26]
                # head
                action[26:28] = initial_action[26:28]
            else:
                #use policy
                if len(action_receding) == 0:
                    # action : 28-dim, left arm(7), left hand(6), right arm(7), right hand(6) head_cam(2)
                    action = predict_action(model, device, yaml_config, img1, img2, obs=states, task_idx=args.task_idx, tac1=left_tac, tac2=right_tac) 
                    if temporal_ensembler_flag:
                        action = temporal_ensembler.update(action.unsqueeze(0))
                        action = action.squeeze(0).numpy()
                    elif receding_horizon:
                        action = action.numpy()  # (chunk, dim)
                        action_receding = action[1:n_action_select, :]
                        action = action[0, :]
                    else:   
                        action = action[0, :].numpy()
                else:
                    action = action_receding[0]
                    action_receding = action_receding[1:, :]
            

            ## control
            # arm
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            tau = arm_ik.solve_tau(current_lr_arm_q, current_lr_arm_dq)
            left_arm_action = action[:7].tolist()
            right_arm_action = action[13:20].tolist()
            action_arm = left_arm_action + right_arm_action
            arm_ctrl.ctrl_dual_arm(action_arm, tau)
            # hand
            left_hand_action = action[7:13].tolist()
            right_hand_action = action[20:26].tolist()
            dual_hand_action_array[:] = left_hand_action + right_hand_action
            # head
            head_action = action[26:28].tolist()
            # print("head_action: ", head_action)
            active_cam._robot.command_joint_state(head_action) 
            ## fps control
            loop_time = time.time() - start_time
            # if loop_time > 1 / args.fps:
            #     logger_mp.warning(f"Loop time is greater than fps: {loop_time} > {1 / args.fps}")
            time.sleep(max(0, (1 / args.fps) - loop_time))

    except Exception as e:
        logger_mp.error(f"Error: {e}")
        raise e

    finally:
        if record_video:
            videowriter.release()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', action='store_true', help='enable motion control')
    parser.add_argument('--network-interface', type=str, default='enx9c69d30201e2', help='Network interface for DDS communication')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.167', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--fps', type=float, default=30, help='fps of the program')
    parser.add_argument('--yaml', type=str, default='./config/deco.yaml', help='path to the trained model file, set pretrain_model_path in yaml to your trained model path')
    parser.add_argument('--task_idx', type=int, default=0, help='onehot index of the sub-task')
    parser.add_argument('--select-action', type=int, default=32, help='select first n actions from the predicted action sequence in diffusion models')
    parser.add_argument('--json_path', type=str, default='~/DECO-50/task1/data-t1-1/episode_0000/data.json', help='initalize the robot state from the recorded data')
    parser.add_argument('--temporal-ensembler', type=bool, default=True, help='whether to use temporal ensembler for action smoothing')
    parser.add_argument('--temporal-ensembler-alpha', type=float, default=0.1, help='alpha for temporal ensembler')
    parser.add_argument('--record-video', type=bool, default=True, help='whether to record video')
    parser.add_argument('--save-video', type=str, default='/home/mani2/deploy_record/save.avi', help='path to save video')
    args = parser.parse_args()

    main(args)