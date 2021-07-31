from pvdet.dataset.Dataset.dataset import KittiDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from pvdet.tools.config import cfg

def build_data_loader(dist=False,data_path= cfg.DATA_DIR,batch_size=1,num_workers=2,training=True,split="train",logger=None,args=None):
    dataset = KittiDataset(datapath=data_path,
                           class_name=cfg.CLASS_NAMES,
                           training=training,
                           split=split,
                           #split = cfg.MODEL["TRAIN" if training else "TEST"].SPLIT,
                           logger= logger,
                           args=args
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


