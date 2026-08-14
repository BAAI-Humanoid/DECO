import os
import av
import torch
import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
_HAS_TORCHCODEC = False
try:
    from torchcodec.decoders import VideoDecoder
    _HAS_TORCHCODEC = True
except Exception:
    print("torchcodec not available!")
    print("We strongly recommend you install torchcodec! Otherwise, it will be very slow if batch size is large.")
    pass


class leDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, **args):
        self.roots = [
            os.path.join(data_dir, name)
            for name in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, name))
        ]

        self.root = self.roots[0]

        self.transform = transform
        self.chunksize = args.get('chunk_size', 32)
        self.norm_type = args.get('norm_type', 'min_max')

        self.obs_mean = torch.tensor(args['observation_mean'])
        self.obs_std = torch.tensor(args['observation_std']).clamp_min(1e-8)
        self.obs_min = torch.tensor(args['observation_min'])
        self.obs_max = torch.tensor(args['observation_max'])
        self.action_mean = torch.tensor(args['action_mean'])
        self.action_std = torch.tensor(args['action_std']).clamp_min(1e-8)
        self.action_min = torch.tensor(args['action_min'])
        self.action_max = torch.tensor(args['action_max'])

        self._use_torchcodec = _HAS_TORCHCODEC


        with open(os.path.join(self.roots[0], "meta/info.json")) as f:
            self.info = json.load(f)

        self.fps = self.info["fps"]
        self.camera_keys = [
            k for k, v in self.info["features"].items()
            if v.get("dtype") in ("video", "image") and "observation.images" in k
        ]
        if not self.camera_keys:
            self.camera_keys = ["observation.images.cam_high"]
        self._video_template = self.info["video_path"]
        # every 1000 episodes are stored in one chunk; helps compute the video chunk index from the episode index
        self.meta_size = self.info.get("chunks_size", 1000)

        self._data = []  # {"action", "state", "task", "ep", "root"}

        for root in self.roots:
            parquet_files = sorted((Path(root) / "data").rglob("*.parquet"))
            for pf in parquet_files:
                table = pq.read_table(str(pf))
                n = table.num_rows
                actions = np.array(
                    table.column("action").combine_chunks().values
                ).reshape(n, -1).astype(np.float32)
                states = np.array(
                    table.column("observation.state").combine_chunks().values
                ).reshape(n, -1).astype(np.float32)
                ep_indices = table.column("episode_index").combine_chunks().to_numpy().astype(np.int32)
                task_indices = table.column("task_index").combine_chunks().to_numpy().astype(np.int32)

                for ep in np.unique(ep_indices):
                    mask = ep_indices == ep
                    self._data.append({
                        "action": actions[mask],
                        "state": states[mask],
                        "task": task_indices[mask],  # raw task_index
                        "ep": int(ep),
                        "root": root,
                    })

        # ── build the global frame index: (data_list_idx, frame_idx) ──
        total_frames = sum(d["action"].shape[0] for d in self._data)
        self.frame_table_indices = np.empty((total_frames, 2), dtype=np.int32)
        offset = 0
        for ti, d in enumerate(self._data):
            n = d["action"].shape[0]
            self.frame_table_indices[offset:offset + n, 0] = ti
            self.frame_table_indices[offset:offset + n, 1] = np.arange(n, dtype=np.int32)
            offset += n

        rng = np.random.default_rng(42)
        rng.shuffle(self.frame_table_indices)
        split = int(total_frames * 0.8)
        if train:
            self.frame_table_indices = self.frame_table_indices
        else:
            self.frame_table_indices = self.frame_table_indices[split:]  # sample some train data for observe val loss, not bugs

    def _video_path(self, cam, ep_idx, root=None):
        """Construct the absolute video path using the video_path template from info.json."""
        root = root or self.root
        return os.path.join(root, self._video_template.format_map(
            defaultdict(int, video_key=cam, episode_index=ep_idx,
                       episode_chunk=ep_idx // self.meta_size)
        ))

    def _decode_frame(self, video_path, frame_idx):
        if self._use_torchcodec:
            try:
                decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
                return decoder.get_frames_at(indices=[frame_idx]).data[0].float() / 255.0
            except Exception:
                self._use_torchcodec = False
                return self._decode_frame(video_path, frame_idx)

        # pyav fallback
        timestamp = frame_idx / self.fps
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        target_pts = int(timestamp / stream.time_base)
        container.seek(target_pts, stream=stream, any_frame=False)
        target = None
        last_frame = None
        for frame in container.decode(stream):
            last_frame = frame
            if frame.pts >= target_pts:
                target = frame.to_ndarray(format="rgb24")
                break
        container.close()
        if target is None and last_frame is not None:
            target = last_frame.to_ndarray(format="rgb24")
        if target is None:
            raise RuntimeError(f"No frames found in {video_path}")
        return torch.from_numpy(target).permute(2, 0, 1).float() / 255.0

    def __len__(self):
        return len(self.frame_table_indices)

    def __getitem__(self, index):
        # ti: episode index, fi: frame index
        ti, fi = self.frame_table_indices[index].tolist()
        d = self._data[ti]
        ep_idx = d["ep"]
        action_np = d["action"]
        state_np = d["state"]

        # decode multiple cameras in parallel
        if len(self.camera_keys) == 1:
            cam = self.camera_keys[0]
            video_path = self._video_path(cam, ep_idx, d["root"])
            img = self._decode_frame(video_path, fi)
            if self.transform is not None:
                img = self.transform(img)
            imgs = img.unsqueeze(0)
        else:
            video_paths = [
                self._video_path(cam, ep_idx, d["root"])
                for cam in self.camera_keys
            ]
            frames = [self._decode_frame(vp, fi) for vp in video_paths]
            imgs = torch.stack(frames, dim=0)  # [n_view, C, H, W]
            if self.transform is not None:
                # torchvision v2 applies the same random parameters to a batch ([N,C,H,W]) to keep multi-view augmentation consistent
                imgs = self.transform(imgs)

        obs_state = torch.from_numpy(state_np[fi]).float()

        # action chunk with padding
        n_rows = action_np.shape[0]
        action_len = min(n_rows - fi, self.chunksize)
        action_slice = action_np[fi:fi + action_len]
        if action_len < self.chunksize:
            pad = np.tile(action_np[-1], (self.chunksize - action_len, 1))
            action_slice = np.concatenate([action_slice, pad], axis=0)
            mask = torch.zeros(self.chunksize, dtype=torch.bool)
            mask[:action_len] = True
        else:
            mask = torch.ones(self.chunksize, dtype=torch.bool)
        action = torch.from_numpy(action_slice).float()

        if self.norm_type == 'min_max':
            obs_state = 2 * (obs_state - self.obs_min) / (self.obs_max - self.obs_min) - 1
            action = 2 * (action - self.action_min[None, :]) / (self.action_max[None, :] - self.action_min[None, :]) - 1
        else:
            obs_state = (obs_state - self.obs_mean) / self.obs_std
            action = (action - self.action_mean[None, :]) / self.action_std[None, :]

        prompt_path = os.path.join(d["root"], "prompt", f'{str(d["task"][fi]).zfill(4)}.pt')
        prompt_file = torch.load(prompt_path, map_location="cpu")
        prompt, prompt_mask = prompt_file['tokens'], prompt_file['mask']
        return imgs, obs_state, action, mask, prompt, prompt_mask.bool()


if __name__ == "__main__":
    import yaml
    data_path = '/home/sunyu/yusun/dataset/robotwin/rand_ori'
    json_path = '/home/sunyu/galbot/codes/robotwin/config/DECO_robotwin.yaml'
    with open(json_path, 'r') as f:
        config = yaml.safe_load(f)
        
    test_transform = transforms.Compose([
        transforms.Resize(config['data']['img_size']),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=config['data']['img_mean'],
            std=config['data']['img_std'])
        ])

    ds = leDataset(data_path, train=True, transform=test_transform, **config['data'])
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=8, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    print(len(loader))
    for imgs, obs, act, mask, prompt, prompt_mask in loader:
        print(f"imgs: {imgs.shape}, obs: {obs.shape}, act: {act.shape}, mask: {mask.shape}, prompt: {prompt.shape}, prompt_mask: {prompt_mask.shape}")

