import os
import shutil

"""
merge the original data into a new folder, and rename the episodes in a unified format: episode_000000, episode_000001, ...
"""
ori_data_path = '/root/total_data/data_task2/original_data'
save_data_path = '/root/total_data/data_task2/data_merged'
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)
    
episode_id = 0

for subtask in os.listdir(ori_data_path):
    subtask_path = os.path.join(ori_data_path, subtask)
    for episodes in os.listdir(subtask_path):
        episode_path = os.path.join(subtask_path, episodes)
        episode_save_path = os.path.join(save_data_path, f"episode_{str(episode_id).zfill(6)}")
        shutil.move(episode_path, episode_save_path)
        episode_id += 1