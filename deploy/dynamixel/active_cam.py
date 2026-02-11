import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from dynamixel.agent import Agent
from dynamixel.dynamixel_robot import DynamixelRobot
from dynamixel.driver import DynamixelDriver
# from agent import Agent
# from dynamixel_robot import DynamixelRobot


@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of dynamixel robot. Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of robot. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs is -1 for all dynamixel"""

    gripper_config: Tuple[int, int, int]
    """reserved for later work"""

    # it will run after init and work as init check
    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )

# Can put multi robot into the dic, note that the calibration info shoule be put here
PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    #! for camera mounta
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8IT033-if00-port0": DynamixelRobotConfig(
        joint_ids=(1, 2),
        joint_offsets=(
            2*np.pi/2, 
            2*np.pi/2, 
        ),
        joint_signs=(-1, -1),
        gripper_config=None,
    ), 

}

# general we only input port into the class, other info is stored in the dic
class DynamixelAgent(Agent):
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        cap_num: int = 42,
    ):
        #! init dynamixel robot setting
        # use the config to make the robot
        if dynamixel_config is not None:
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        # find the info auto
        else:
            # check port 
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            # use port to gain config
            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(port=port, start_joints=start_joints)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray: 
        return self._robot.get_joint_state()
    
class ActiveCam:
    def __init__(self, ids: np.ndarray, offsets: np.ndarray, limits: np.ndarray, signs: np.ndarray = [1, 1]):
        '''
        N motors
        ids: the ids of the dynamixel motors N
        offsets: the offsets of the dynamixel motors N
        limits: the limits of the dynamixel motors N x 2
        signs: the signs of the dynamixel motors N
        '''
        self.ids = ids
        self.offsets = offsets
        self.limits = limits
        self.signs = signs
        self.driver = DynamixelDriver(ids, port="/dev/ttyUSB0")

        ## init 
        # torque mode
        self.driver.set_torque_mode(False)
        time.sleep(0.1)
        self.driver.set_torque_mode(True)
        # set joint offsets
        self.driver.set_joints(self.offsets)

    def set_action(self, action: np.ndarray):
        action_clip = np.clip(action, self.limits[0], self.limits[1])
        action_clip = action_clip * self.signs
        action_value = action_clip + self.offsets
        self.driver.set_joints(action_value)

    def get_observation(self):
        observation = (self.driver.get_joints() - self.offsets) * self.signs
        return observation

    def release(self):
        self.driver.set_joints(self.offsets)
        self.driver.set_torque_mode(False)
        self.driver.close()
if __name__ == "__main__":

    # 偏航 1.57693225 向左为正
    # 俯仰 3.07102954 向下为正
    config = DynamixelRobotConfig(
        joint_ids=(1, 2),
        joint_offsets=(
            np.pi, 
            np.pi, 
        ),
        joint_signs=(1, 1),
        gripper_config=None,
    )
    agent = DynamixelAgent(port="/dev/ttyUSB0", dynamixel_config=config)

    agent._robot.set_torque_mode(True)

    min_radians = -1.57
    max_radians = np.pi/4
    interval = 0.01

    current_yaw = -np.pi/4
    current_pitch = 0
    while True:
    #while True:
        #action = agent.act(1)
        #print("now action                     : ", [f"{x:.3f}" for x in action])
        agent._robot.command_joint_state([0, 0])
        time.sleep(0.1) 
        true_value = agent._robot.get_joint_state()   
        print("true value                 : ", [f"{x:.3f}" for x in true_value])
        current_yaw += 2*interval
        current_pitch += interval
        if current_yaw > np.pi/4 or current_pitch > np.pi/4:
            break
    '''
    try:
        active_cam = ActiveCam(
            ids=np.array([1, 2]), 
            offsets=np.array([np.pi/2, np.pi]), 
            limits=np.array([[-np.pi/2, np.pi/2], [0, np.pi/2]]), 
            signs=np.array([1, 1]))
        
        start_time = time.time()
        while time.time() - start_time < 5:
            #action = np.array([0, 0])
            #active_cam.set_action(action)
            observation = active_cam.get_observation()
            print("observation: ", [f"{x:.3f}" for x in observation])
            time.sleep(0.1)

        print("finish observation")
        active_cam.release()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        active_cam.release()
        print("Finally Release")
    '''