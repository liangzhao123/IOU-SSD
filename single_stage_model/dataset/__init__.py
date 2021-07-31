from single_stage_model.dataset.kitti import KittiDataset
import sys
sys.path.append("/media/ubuntu-502/pan1/liang/PVRCNN-V1.1/single_stage_model/dataset")
try:
    from single_stage_model.dataset.leishen_dataset.LSDataset import LeiShenDataset
except:
    from ..dataset.leishen_dataset.LSDataset import LeiShenDataset

import torch
from torch.utils.data import DataLoader

# from single_stage_model.configs.single_stage_config import cfg

def build_data_loader(cfg,batch_size=1,num_workers=2,training=True,split="train",logger=None,args=None,train_all=None,dist=False):
    if cfg.get("LEISHEN",None) is None:
        dataset = KittiDataset(datapath=cfg.DATA_DIR,
                               class_name=cfg.CLASS_NAMES,
                               training=training,
                               split=split,
                               #split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
                               logger= logger,
                               args =args,
                               train_all=train_all,
                               )
    else:
        dataset = LeiShenDataset(datapath=cfg.DATA_DIR,
                               class_name=cfg.CLASS_NAMES,
                               training=training,
                               split=split,
                               #split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
                               logger= logger,
                               args =args,
                               train_all=train_all,
                               )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=num_workers,
                            shuffle=(sampler is None) and training,
                            collate_fn=dataset.collate_batch,
                            drop_last=False,
                            sampler=sampler,
                            timeout=0
                            )
    return dataset,dataloader,sampler