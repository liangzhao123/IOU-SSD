from pvdet.tools.config import cfg,log_cfg_to_file
import os
import datetime
from pvdet.dataset.utils import common_utils
from tensorboardX import SummaryWriter
from pvdet.dataset.Dataset import build_data_loader
from pvdet.model.detectors import build_netword
import torch.distributed as dist
from pvdet.tools.optimization import build_optimizer, build_scheduler
import glob
from pvdet.tools.train_utils.train_utils import model_fn_decorator
from pvdet.tools.train_utils.train_utils import train_model
from easydict import EasyDict as edict
import sys
# sys.path.append("/home/liang/PVRCNN-V1.1")
sys.path.append("/media/ubuntu-502/pan1/liang/PVRCNN-V1.1")
def parse_config():

    args = edict()
    args.local_rank = 0
    args.batch_size = 2
    args.epochs = 80
    args.workers = 15
    args.extra_tag = "0.0.0"
    args.ckpt = os.path.join(cfg.CODE_DIR,"ckpt",cfg.TAG,args.extra_tag)
    args.pretrained_model = None
    args.version=args.extra_tag
    args.ckpt_save_interval = 2
    args.max_ckpt_save_num = 40


    args.start_epoch = 0
    args.save_to_file = True
    args.split = "train"

    return args


def main():
    args = parse_config()
    output_dir = os.path.join(cfg.CODE_DIR,"output",cfg.TAG,args.split,args.extra_tag)
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
    tb_log = SummaryWriter(log_dir=tb_log_path if cfg.train_args.LOCAL_RANK ==0 else None)

    dist_train=False #非分布式运行
    dataset,data_loader,sampler= build_data_loader(batch_size=args.batch_size,
                                                   num_workers=cfg.DATA_CONFIG.NUM_WORKERS,
                                                   training=True,
                                                   dist=dist_train,
                                                   logger=logger)
    model = build_netword(num_class=len(cfg.CLASS_NAMES), dataset=dataset,logger=logger)
    model.cuda()

    optimizer = build_optimizer(model, cfg.MODEL.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1
    if cfg.train_args.pretrained_model is not None:
        model.load_params_from_file(filename=cfg.train_args.pretrained_model,to_cpu=dist_train)

    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*checkpoint_epoch_*.pth"))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getatime)
        it, start_epoch = model.load_params_with_optimizer(ckpt_list[24], to_cpu=dist,
                                                           optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1

    model.train()
    #logger.info(model)

    lr_scheduler,lr_warmup_scheduler = build_scheduler(optimizer,
                                                       total_iters_each_epoch=len(data_loader),
                                                       total_epochs=args.epochs,
                                                       last_epoch=last_epoch,
                                                       optim_cfg=cfg.MODEL.OPTIMIZATION)
    logger.info("************start training*************")
    train_model(
        model,
        optimizer,
        data_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.MODEL.OPTIMIZATION,
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
    main()