"""XPolicyLab policy-server adapter for the DECO model.

The policy server instantiates ``Model(model_cfg)``, then drives it with
``update_obs(obs)`` / ``get_action()`` per control step. ``obs`` is the
xpolicylab observation format:

    {
        "vision": { "cam_head": {"color": HWC uint8}, "cam_left_wrist": ...,
                    "cam_right_wrist": ... },
        "state":  { "left_arm_joint_state", "left_ee_joint_state",
                    "right_arm_joint_state", "right_ee_joint_state" },
        "instruction": "text prompt",
    }

``get_action()`` returns an action chunk; each action is the flat 14-dim qpos
vector [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import v2 as transforms

# DECO's own modules (deco.py / tokenizer.py) use plain absolute imports
# (e.g. `from denoise_schedular import ...`), so the policy directory must be
# importable at the top level.
_POLICY_DIR = Path(__file__).resolve().parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from deco import modeling  # noqa: E402
from tokenizer import HFEmbedder  # noqa: E402

from XPolicyLab.model_template import ModelTemplate  # noqa: E402
from XPolicyLab.utils.checkpoint_resolver import ckpt_name_is_path  # noqa: E402
from XPolicyLab.utils.process_data import (  # noqa: E402
    get_robot_action_dim_info,
    pack_robot_state,
)

# xpolicylab observation camera names -> DECO view names
CAMERA_KEYS = {
    "head": ("cam_head", "head_camera"),
    "left": ("cam_left_wrist", "left_camera"),
    "right": ("cam_right_wrist", "right_camera"),
}

DEFAULT_T5_PATH = os.environ.get("T5_MODEL_PATH")


class Model(ModelTemplate):
    """DECO adapter for the XPolicyLab policy-server interface."""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg.get("action_type") or "joint")
        if self.action_type != "joint":
            raise ValueError(
                f"DECO was trained for joint (qpos) control, got "
                f"action_type={self.action_type!r}. Use --action-type joint."
            )

        with open(_POLICY_DIR / "DECO.yaml", "r", encoding="utf-8") as f:
            self.deco_cfg = yaml.safe_load(f)

        model_kwargs = dict(self.deco_cfg["model"])
        pretrained_path = self._resolve_pretrained_path(model_cfg)
        if pretrained_path is None:
            raise ValueError(
                "Could not locate DECO weights. Fix 'pretrain_model_path' in "
                "DECO.yaml, set an explicit path key (model_path/checkpoint_path/"
                "ckpt_path/model_dir/pretrained_path) in deploy.yml, or pass a "
                ".pth path via --ckpt-name."
            )
        model_kwargs["pretrain_model_path"] = pretrained_path
        print(f"[DECO] loading pretrained weights from {pretrained_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = modeling(**model_kwargs)
        self.model.eval()
        self.model.to(self.device)

        t5_path = (
            model_cfg.get("t5_path")
            or self.deco_cfg.get("t5_path")
            or DEFAULT_T5_PATH
        )
        # GPU matches the original DECO eval (bf16 + autocast); CPU needs fp32
        # because torch.autocast is unavailable there and bf16 tensors would
        # clash with the fp32 linear weights.
        t5_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"[DECO] loading T5 text encoder from {t5_path} ({t5_dtype})")
        self.t5 = HFEmbedder(t5_path, max_length=64, torch_dtype=t5_dtype)
        self.t5.to(self.device)

        if not model_cfg.get("env_cfg_type"):
            raise ValueError(
                "env_cfg_type must be specified (e.g. --env-cfg-type arx_x5)."
            )
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])

        img_size = list(self.deco_cfg["data"].get("img_size", [192, 256]))
        self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        self._obs = None

    # ------------------------------------------------------------------ #
    # checkpoint / weight resolution
    # ------------------------------------------------------------------ #
    def _resolve_pretrained_path(self, model_cfg):
        # 1. explicit path keys carried by deploy.yml / CLI overrides
        for key in (
            "model_path",
            "checkpoint_path",
            "ckpt_path",
            "model_dir",
            "pretrained_path",
        ):
            value = model_cfg.get(key)
            if value:
                path = self._maybe_file(value)
                if path is not None:
                    return str(path)
        # 2. --ckpt-name given as a path to a .pth file
        ckpt_name = model_cfg.get("ckpt_name")
        if ckpt_name and ckpt_name_is_path(ckpt_name):
            path = self._maybe_file(ckpt_name)
            if path is not None:
                return str(path)
        # 3. checkpoints/<ckpt_name>.pth / checkpoints/<ckpt_name>
        if ckpt_name:
            for candidate in (
                _POLICY_DIR / "checkpoints" / f"{ckpt_name}.pth",
                _POLICY_DIR / "checkpoints" / str(ckpt_name),
            ):
                if candidate.is_file():
                    return str(candidate)
        # 4. DECO_MODEL_PATH env / DECO.yaml default
        default = os.environ.get("DECO_MODEL_PATH") or self.deco_cfg.get(
            "model", {}
        ).get("pretrain_model_path")
        if default:
            path = self._maybe_file(default)
            if path is not None:
                return str(path)
        return None

    def _maybe_file(self, value):
        path = Path(os.path.expanduser(str(value)))
        if not path.is_absolute():
            path = _POLICY_DIR / path
        if path.is_file():
            return path
        return None

    # ------------------------------------------------------------------ #
    # observation encoding
    # ------------------------------------------------------------------ #
    def update_obs(self, obs):
        self._obs = self._encode_obs(obs)

    def _encode_obs(self, obs):
        data_cfg = self.deco_cfg["data"]
        vision = obs.get("vision") or {}

        img_mean = torch.tensor(data_cfg["img_mean"], dtype=torch.float32)
        img_std = torch.tensor(data_cfg["img_std"], dtype=torch.float32)
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )

        views = []
        for view_name in ("head", "left", "right"):
            color = self._camera_color(vision, view_name)
            image = Image.fromarray(color).resize((self.img_w, self.img_h))
            views.append(transform(image))
        composite = torch.stack(views, dim=0).unsqueeze(0)  # [1, n_view, C, H, W]
        composite = composite.to(device=self.device)

        state_vec = pack_robot_state(
            obs, self.action_type, self.robot_action_dim_info, source_type="obs"
        )
        obs_mean = torch.tensor(data_cfg["observation_mean"], dtype=torch.float32)
        obs_std = torch.tensor(data_cfg["observation_std"], dtype=torch.float32).clamp_min(
            1e-8
        )
        obs_state = (torch.tensor(state_vec, dtype=torch.float32) - obs_mean) / obs_std

        instruction = obs.get("instruction") or (obs.get("instructions") or [""])[0]
        return composite, obs_state, str(instruction)

    def _camera_color(self, vision, view_name):
        for key in CAMERA_KEYS[view_name]:
            camera = vision.get(key)
            if camera is None:
                continue
            if isinstance(camera, dict):
                for image_key in ("color", "rgb"):
                    if image_key in camera:
                        return np.asarray(camera[image_key])
            return np.asarray(camera)
        raise KeyError(
            f"Missing camera '{view_name}' in observation['vision']; expected one "
            f"of {CAMERA_KEYS[view_name]}, got {sorted(vision.keys())}"
        )

    # ------------------------------------------------------------------ #
    # action generation
    # ------------------------------------------------------------------ #
    def get_action(self):
        if self._obs is None:
            raise RuntimeError("get_action() called before update_obs().")
        composite, obs_state, instruction = self._obs
        data_cfg = self.deco_cfg["data"]
        act_mean = torch.tensor(data_cfg["action_mean"], dtype=torch.float32)
        act_std = torch.tensor(data_cfg["action_std"], dtype=torch.float32).clamp_min(1e-8)

        obs_state = obs_state.unsqueeze(0).to(device=self.device)
        if self.device.type == "cuda":
            obs_state = obs_state.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            text_embedding, mask = self.t5([instruction])
            mask = mask.to(torch.bool)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    actions = self.model(
                        imgs=composite,
                        obs=obs_state,
                        act=None,
                        prompt=text_embedding,
                        prompt_mask=mask,
                        training=False,
                    )
            else:
                actions = self.model(
                    imgs=composite,
                    obs=obs_state,
                    act=None,
                    prompt=text_embedding,
                    prompt_mask=mask,
                    training=False,
                )
        actions = actions.squeeze(0).cpu()
        actions = (actions * act_std + act_mean).numpy()
        actions = actions[::2]
        return [np.asarray(action, dtype=np.float32).reshape(-1) for action in actions]

    def reset(self):
        # DECO is stateless between chunks; clear the cached observation.
        self._obs = None
