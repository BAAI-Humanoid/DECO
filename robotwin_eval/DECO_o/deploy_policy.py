import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
from PIL import Image
from deco import modeling
from tokenizer import HFEmbedder
from torchvision.transforms import v2 as transforms


yaml_config = yaml.safe_load(open('./robotwin_eval/DECO_o/DECO.yaml', 'r'))
yaml_config['model']['pretrain_model_path'] = './robotwin.pth'
t5_path = './models/t5_base'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(yaml_config)

def encode_obs(observation):  # Post-Process Observation
    head = Image.fromarray(observation["observation"]["head_camera"]["rgb"]).resize((256, 192))
    left = Image.fromarray(observation["observation"]["left_camera"]["rgb"]).resize((256, 192))
    right = Image.fromarray(observation["observation"]["right_camera"]["rgb"]).resize((256, 192))
    obs_mean, obs_std = torch.tensor(yaml_config['data']['observation_mean']), torch.tensor(yaml_config['data']['observation_std'])
    obs_state = torch.tensor(observation['joint_action']['vector'])
    obs_state = (obs_state - obs_mean) / obs_std
    img_mean = yaml_config['data']['img_mean']
    img_std = yaml_config['data']['img_std']
    transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    head = transform(head).unsqueeze(0)
    left = transform(left).unsqueeze(0)
    right = transform(right).unsqueeze(0)
    composite = torch.stack([head, left, right], dim=1)  # [B, n_view=3, C, H, W]
    composite = composite.to(device=device)
    return composite, obs_state  # return your observation


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    deco = modeling(**yaml_config['model'])
    deco.eval()
    deco.to(device)
    t5 = HFEmbedder(t5_path, max_length=64, torch_dtype=torch.bfloat16)
    t5.to(device)
    return {'model': deco, 't5': t5}  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    act_mean, act_std = torch.tensor(yaml_config['data']['action_mean']), torch.tensor(yaml_config['data']['action_std']).clamp_min(1e-8)
    img, obs_state = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    deco, t5 = model['model'], model['t5']
    obs_state = obs_state.unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [B, state_dim]
    with torch.inference_mode():
        text_embedding, mask = t5([instruction])
        mask = mask.to(torch.bool)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = deco(imgs=img, obs=obs_state, act=None, prompt=text_embedding, prompt_mask=mask, training=False)  
    actions = actions.squeeze(0).cpu()
    actions = (actions * act_std + act_mean).numpy()
    actions = actions[::2]
    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        # observation = TASK_ENV.get_obs()


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass

if __name__ == '__main__':
    a = get_model(1)