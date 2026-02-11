'''
for deployment, controller only do 2 things:
    1. get actions from shared memory and send command to hand hardware
    2. get state from hand hardware and send to shared memory
'''
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

import numpy as np
import threading
import time
from multiprocessing import Process, Array

import logging_mp

logger_mp = logging_mp.get_logger(__name__)

# number of motors for inspire hand
Inspire_Num_Motors = 6

# basic info for inspire hand 
kTopicInspireFTPLeftCommand   = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand  = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState  = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"
kTopicInspireFTPLeftTactile = "rt/inspire_hand/touch/l"
kTopicInspireFTPRightTactile = "rt/inspire_hand/touch/r"

# get tactile data name and corresponding range
_touch_regions = [
    ("fingerone_tip_touch", 3 * 3),
    ("fingerone_top_touch", 12 * 8),
    ("fingerone_palm_touch", 10 * 8),
    ("fingertwo_tip_touch", 3 * 3),
    ("fingertwo_top_touch", 12 * 8),
    ("fingertwo_palm_touch", 10 * 8),
    ("fingerthree_tip_touch", 3 * 3),
    ("fingerthree_top_touch", 12 * 8),
    ("fingerthree_palm_touch", 10 * 8),
    ("fingerfour_tip_touch", 3 * 3),
    ("fingerfour_top_touch", 12 * 8),
    ("fingerfour_palm_touch", 10 * 8),
    ("fingerfive_tip_touch", 3 * 3),
    ("fingerfive_top_touch", 12 * 8),
    ("fingerfive_middle_touch", 3 * 3),
    ("fingerfive_palm_touch", 12 * 8),
    ("palm_touch", 8 * 14),
]
_touch_start = 0
tactile_data_index = {}
for region_name, region_size in _touch_regions:
    tactile_data_index[region_name] = [_touch_start, _touch_start + region_size]
    _touch_start += region_size
assert _touch_start == 1062, "Total tactile data length should be 1062"

class Inspire_Controller_FTP:
    def __init__(
        self,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        networkInterface=None,
    ):
        logger_mp.info("Initialize Inspire_Controller...")

        self.fps = fps

        # initialize channel factory
        if networkInterface is None:
            print("Using default network interface")
            ChannelFactoryInitialize(0)
        else:
            print(f"Using network interface: {networkInterface}")
            ChannelFactoryInitialize(0, networkInterface=networkInterface)

       # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()
        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()
        # Initialize tactile subscribers
        self.LeftHandTactile_subscriber = ChannelSubscriber(kTopicInspireFTPLeftTactile, inspire_dds.inspire_hand_touch)
        self.LeftHandTactile_subscriber.Init()
        self.RightHandTactile_subscriber = ChannelSubscriber(kTopicInspireFTPRightTactile, inspire_dds.inspire_hand_touch)
        self.RightHandTactile_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array  = Array('d', 6, lock=True)
        self.right_hand_state_array = Array('d', 6, lock=True)
        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        # Initialize tactile shared arrays
        self.left_hand_tactile_array = Array('d', 1062, lock=True)
        self.right_hand_tactile_array = Array('d', 1062, lock=True)
        # initialize tactile subscribe thread
        self.subscribe_tactile_thread = threading.Thread(target=self._subscribe_tactile)
        self.subscribe_tactile_thread.daemon = True
        self.subscribe_tactile_thread.start()

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0: # Print every second
                logger_mp.info(f"[Inspire_Controller_FTP] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500: # Timeout after 5 seconds
                logger_mp.warning("[Inspire_Controller_FTP] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        logger_mp.info(f"[Inspire_Controller_FTP] Current hand states: (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})...")
        logger_mp.info("[Inspire_Controller_FTP] Initial hand states received or timeout.")

        # control process
        hand_control_process = Process(
            target=self.control_process,
            args=(
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller_FTP] Subscribe thread started.")
        while True:
            # Left Hand
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(f"[Inspire_Controller_FTP] Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
            # Right Hand
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(f"[Inspire_Controller_FTP] Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")

            time.sleep(0.002)

    def _subscribe_tactile(self):
        logger_mp.info("[Inspire_Controller_FTP] Subscribe thread started.")
        while True:
            # Left Hand
            left_tactile_msg = self.LeftHandTactile_subscriber.Read()
            if left_tactile_msg is not None:
                with self.left_hand_tactile_array.get_lock():
                    for field_name, region in tactile_data_index.items():
                        segment = np.asarray(getattr(left_tactile_msg, field_name), dtype=np.float64)
                        self.left_hand_tactile_array[region[0]:region[1]] = segment
            # Right Hand
            right_tactile_msg = self.RightHandTactile_subscriber.Read()
            if right_tactile_msg is not None:
                with self.right_hand_tactile_array.get_lock():
                    for field_name, region in tactile_data_index.items():
                        segment = np.asarray(getattr(right_tactile_msg, field_name), dtype=np.float64)
                        self.right_hand_tactile_array[region[0]:region[1]] = segment
            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        # Left Hand Command
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # Right Hand Command
        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001 # Mode 1: Angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def control_process(
        self,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        self.running = True
        try:
            while self.running:
                start_time = time.time()

                # get dual hand state
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if dual_hand_state_array is not None and dual_hand_action_array is not None and dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        left_action = dual_hand_action_array[:Inspire_Num_Motors]
                        right_action = dual_hand_action_array[Inspire_Num_Motors:]
                else:
                    logger_mp.warning("[Inspire_Controller_FTP] Dual hand state or action array is not initialized.")
                    logger_mp.warning(f"[Inspire_Controller_FTP] dual_hand_data_lock: {dual_hand_data_lock}")
                    logger_mp.warning(f"[Inspire_Controller_FTP] dual_hand_state_array: {dual_hand_state_array}")
                    logger_mp.warning(f"[Inspire_Controller_FTP] dual_hand_action_array: {dual_hand_action_array}")
                    continue

                # send hand command [0,1] to [0,1000]
                left_cmd_scaled = [int(np.clip(val * 1000.0, 0, 1000)) for val in left_action]
                right_cmd_scaled = [int(np.clip(val * 1000.0, 0, 1000)) for val in right_action]
                self._send_hand_command(left_cmd_scaled, right_cmd_scaled)

                # 频率控制
                sleep_time = max(0.0, (1.0 / self.fps) - (time.time() - start_time))
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")
