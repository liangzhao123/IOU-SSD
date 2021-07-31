import numpy as np
import os
from new_train.config import cfg
import random


def split_train_val():
    """
    split the 7481 frame to 80% for train and 20% for val
    """
    split_dir_train = os.path.join(cfg.DATA_DIR, "ImageSets", "train.txt")
    split_dir_val = os.path.join(cfg.DATA_DIR, "ImageSets", "val.txt")
    sample_idx_list = [x.strip() for x in open(split_dir_train).readlines()]
    sample_idx_list += [x.strip() for x in open(split_dir_val).readlines()]
    # sample_idx_list = np.array(sample_idx_list)
    train_size = int(len(sample_idx_list) * 0.8)
    temp_idx = list(np.arange(0, len(sample_idx_list)))
    train_idx = random.sample(temp_idx, train_size)

    sample_idx_list = np.array(sample_idx_list)
    train_idx = np.array(train_idx)
    temp_idx = np.array(temp_idx)
    train_split_80_id = sample_idx_list[train_idx]
    test_idx = np.delete(temp_idx, train_idx)
    val_split_20_id = sample_idx_list[test_idx]
    for i in val_split_20_id:
        if i in train_split_80_id:
            print("warning")
    save_path_train = os.path.join(cfg.DATA_DIR, "ImageSets", "train_80%.txt")
    save_path_val = os.path.join(cfg.DATA_DIR, "ImageSets", "val_20%.txt")
    fl = open(save_path_train, "w")
    sep = '\n'
    fl.write(sep.join(train_split_80_id))
    fl.close()
    fl = open(save_path_val, "w")
    sep = "\n"
    fl.write(sep.join(val_split_20_id))
    fl.close()
    print("done")


if __name__ == '__main__':
    # split_train_val()
    split_dir_train = os.path.join(cfg.DATA_DIR, "ImageSets", "train_80%.txt")
    sample_idx_list = [x.strip() for x in open(split_dir_train).readlines()]
    split_dir_val = os.path.join(cfg.DATA_DIR, "ImageSets", "val_20%.txt")
    sample_idx_list_1 = [x.strip() for x in open(split_dir_val).readlines()]
    print("done")