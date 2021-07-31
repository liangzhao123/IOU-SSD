from single_stage_model.configs.single_stage_config import cfg,log_cfg_to_file
import torch.nn as nn
import torch
import os
import datetime
from pvdet.dataset.utils import common_utils
from tensorboardX import SummaryWriter
from single_stage_model.dataset import build_data_loader
from pvdet.model.detectors.F_PVRCNN import FPVdet
import torch.distributed as dist
from pvdet.tools.optimization import build_optimizer, build_scheduler
import glob
from pvdet.tools.train_utils.train_utils import model_fn_decorator
from pvdet.tools.train_utils.train_utils import train_model
from easydict import EasyDict as edict
from pvdet.model.detectors.part2net import Part2net
import sys
import argparse

from single_stage_model.detector.light_detector import LightDetector

if cfg.using_remote:
    sys.path.append("/media/ubuntu-502/pan1/liang/PVRCNN-V1.1")
else:
    from pathlib import Path
    sys_path = str((Path(__file__).resolve().parent / '../').resolve())
    sys.path.append("/home/liang/for_ubuntu502/PVRCNN-V1.1/single_stage_model")
#0.0.6 is the best model
def parse_config():
    using_IED =True
    if using_IED:
        args = edict()
        args.TAG = "single_stage_model"
        args.local_rank = 0
        args.batch_size = 3
        args.epochs = 80
        args.workers = 15
        args.extra_tag = "0.0.6"
        args.ckpt = os.path.join(cfg.CODE_DIR, "ckpt", args.TAG, args.extra_tag)
        args.pretrained_model = None
        args.version = args.extra_tag
        args.ckpt_save_interval = 1
        args.max_ckpt_save_num = 30
        args.tcp_port = 18886
        args.start_epoch = 0
        args.save_to_file = True
        args.split = "train"
        args.launcher = "none"
        args.train_all = False #test on offical kitti benchmark
        args.detector_name = args.TAG
    else:
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
        parser.add_argument('--TAG', type=str, default="single_stage_model", help='specify the config for training')
        parser.add_argument('--batch_size', type=int, default=9, required=False, help='batch size for training')
        parser.add_argument('--epochs', type=int, default=80, required=False, help='number of epochs to train for')
        parser.add_argument('--workers', type=int, default=3, help='number of workers for dataloader')
        parser.add_argument('--extra_tag', type=str, default='0.0.2', help='extra tag for this experiment')
        # parser.add_argument('--ckpt', type=str, default=os.path.join(cfg.CODE_DIR,"ckpt",cfg.TAG,parser.extra_tag), help='checkpoint to start from')
        parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
        parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
        parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
        parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
        parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
        parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

        parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
        parser.add_argument('--start_epoch', type=int, default=0, help='')
        parser.add_argument('--save_to_file', action='store_true', default=False, help='')
        parser.add_argument('--split', type=str, default="train", help='train or test')
        parser.add_argument('--train_all', action='store_true', default=False, help='whether to train in all training fold including 7481')
        parser.add_argument('--detector_name', type=str, default="single_stage_model", help='select the detector model')
        args = parser.parse_args()
        args.ckpt = os.path.join(cfg.CODE_DIR,"ckpt",args.TAG,args.extra_tag)


    return args



def main():
    args = parse_config()
    dist_train = False  # 非分布式运行
    if args.launcher=="pytorch":
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    output_dir = os.path.join(cfg.CODE_DIR,"output",args.TAG,args.split,args.extra_tag)
    os.makedirs(output_dir,exist_ok=True)
    ckpt_dir = args.ckpt
    os.makedirs(ckpt_dir,exist_ok=True)

    log_file = os.path.join(output_dir,("log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    logger = common_utils.create_logger(log_file)

    logger.info("*********************start logging********************************")
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ.keys() else "ALL"
    log_cfg_to_file(cfg,logger=logger)

    tb_log_path = os.path.join(output_dir,"tensorboard")
    os.makedirs(tb_log_path,exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_log_path if args.local_rank ==0 else None)


    dataset,data_loader,sampler= build_data_loader(dist=dist_train,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   training=True,
                                                   logger=logger,
                                                   split=args.split,
                                                   args = args,
                                                   train_all=args.train_all,
                                                   cfg =cfg)
    if args.detector_name=="LZnet":
        model = FPVdet(dataset=dataset,logger=logger)
    elif args.detector_name=="pvrcnn":
        model = Part2net(num_class=len(cfg.CLASS_NAMES),dataset=dataset)
    elif args.detector_name == "single_stage_model":
        # model = LightDetector(dataset=dataset,logger=logger)
        model = LightDetector(logger=logger,config=cfg.model,cfg=cfg)

    else:
        raise NotImplementedError
    model.cuda()

    optimizer = build_optimizer(model, cfg.model.optimization)

    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model,to_cpu=dist_train)

    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*checkpoint_epoch_*.pth"))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getatime)
        it, start_epoch = model.load_params_with_optimizer(ckpt_list[-1], to_cpu=dist,
                                                           optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1

    model.train()
    #logger.info(model)
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])

    lr_scheduler,lr_warmup_scheduler = build_scheduler(optimizer,
                                                       total_iters_each_epoch=len(data_loader),
                                                       total_epochs=args.epochs,
                                                       last_epoch=last_epoch,
                                                       optim_cfg=cfg.model.optimization,
                                                       )
    logger.info("************start training*************")
    train_model(
        model,
        optimizer,
        data_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.model.optimization,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=args.local_rank,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=sampler,
        lr_warmup_scheduler = lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval ,
        max_ckpt_save_num=args.max_ckpt_save_num
    )
    logger.info("**************End training********************")



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    # print(torch.cuda.device_count())
    main()