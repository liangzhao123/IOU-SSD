import os
import torch
from tensorboardX import SummaryWriter
import time
from new_train.config import cfg
import re
import datetime
from pvdet.dataset.utils import common_utils
from new_train.config import log_cfg_to_file
import torch.distributed  as dist
from new_train.dataset import build_data_loader
from pvdet.model.detectors import build_netword
import glob
from pvdet.tools.eval_utils import eval_utils
from easydict import EasyDict as edict
from pvdet.tools.train_utils.train_utils import  example_convert_to_torch
from pvdet.model.detectors.F_PVRCNN import FPVdet
from pvdet.model.detectors.part2net import Part2net


def version(check):
    assert check=="0.0.6"
def parse_config():
    args = edict()
    args.TAG = "LZnet"
    args.detector_name =  args.TAG
    args.local_rank = 0
    args.batch_size = 4
    args.epochs = 80
    args.workers = 15
    args.extra_tag = "0.1.2"
    args.ckpt_id = 80
    args.pretrained_model = None

    args.ckpt_save_interval = 2
    args.max_ckpt_save_num = 40

    args.eval_all = False
    args.start_epoch = 0
    args.save_to_file = True
    args.split = "val" #train,val,test,val_20%
    #"/home/ubuntu-502/liang/PVRCNN-V1.1/ckpt/LZnet/0.0.6/checkpoint_epoch_80.pth"
    args.ckpt_dir = os.path.join(cfg.CODE_DIR,"ckpt",args.TAG,args.extra_tag)
    #"checkpoint_epoch_%d.pth" % args.ckpt_id
    args.ckpt = os.path.join(args.ckpt_dir,"checkpoint_epoch_%d.pth" % args.ckpt_id)

    return args



def repeat_eval_ckpt(model,test_loader,test_cfg,eval_output_dir,logger,ckpt_dir):
    ckpt_record_file = os.path.join(eval_output_dir,("eval_list_%s.txt"% cfg.MODEL.TEST.SPLIT))
    with open(ckpt_record_file,"a"):
        pass
    tb_log = SummaryWriter(log_dir=os.path.join(eval_output_dir,"tensorboard_%s"% cfg.MODEL.TEST.SPLIT))
    total_time = 0
    first_eval = True

    while True:
        cur_epoch_id,cur_ckpt = get_no_evaluated_ckpt(ckpt_dir,ckpt_record_file,test_cfg)
        if cur_epoch_id ==-1 or int(float(cur_epoch_id))<test_cfg.start_epoch:
            wait_second = 30
            print("Wait %s second for next check(progress:%.1f/%d minutes): %s \r"
                  %(wait_second,total_time*1.0/60,test_cfg.max_wait_mins,ckpt_dir),end="",flush=True
                  )
            time.sleep(wait_second)
            total_time+=30

            if total_time >test_cfg.max_wait_mins*60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt,logger=logger)
        model.cuda()

        #开始测试
        cur_result_dir = os.path.join(eval_output_dir,"epoch_%s"%cur_epoch_id,cfg.MODEL.TEST.SPLIT)
        tb_dict = eval_utils.eval_one_epoch(model,test_loader,cur_epoch_id,
                                            logger,save_to_file=test_cfg.save_to_file,
                                            result_dir=eval_output_dir)
        for key,val, in tb_dict.items():
            tb_log.add_scalar(key,val,cur_epoch_id)

        with open(ckpt_record_file,"a") as f:
            print("%s" % cur_epoch_id,file=f)
        logger.info("Epoch % has been evaluated" % cur_epoch_id)



def get_no_evaluated_ckpt(ckpt_dir,ckpt_record_file,test_cfg):
    ckpt_list = glob.glob(os.path.join(ckpt_dir,"*ckpt_epoch_*.pth"))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file,"r").readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall("ckpt_epoch_(.*).pth",cur_ckpt)
        if num_list.__len__() ==0:
            continue

        epoch_id = num_list[-1]
        if "optim" in epoch_id:
            continue

        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= test_cfg.start_epoch:
            return epoch_id,cur_ckpt


def main():
    args = parse_config()
    os.makedirs(args.ckpt_dir,exist_ok=True)
    output_dir = os.path.join(cfg.CODE_DIR,"output",args.TAG,args.extra_tag)
    os.makedirs(output_dir,exist_ok=True)
    eval_output_dir = os.path.join(output_dir,"eval")

    if not args.eval_all:
        # num_list  = re.findall(r"\d+",args.ckpt_id) if args.ckpt_id is not None else []
        # epoch_id = num_list[-1] if num_list.__len__()>0 else "no number"
        epoch_id = args.ckpt_id
        eval_output_dir = os.path.join(eval_output_dir,"epoch_%s"% epoch_id)
    else:
        eval_output_dir = os.path.join(eval_output_dir,"eval_all_default")

    os.makedirs(eval_output_dir,exist_ok=True)

    log_file = os.path.join(eval_output_dir,"log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file)

    #log to file
    logger.info("*****************start logging****************")
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)


    #非分布式训练
    dist_test = False
    if dist_test:
        total_gpus = dist.get_world_size()
        logger.info("total_batch_size:%d"% (total_gpus*args.batch_size))

    for key,val in vars(args).items():
        logger.info("{:16} {}".format(key,val))

    log_cfg_to_file(cfg,logger=logger)

    ckpt_dir = args.ckpt

    test_set,test_dataloader,sampler = build_data_loader(
        dist=dist_test,
        data_path=cfg.DATA_DIR,
        batch_size=args.batch_size,
        num_workers=cfg.DATA_CONFIG.NUM_WORKERS,
        logger=logger,
        split=args.split,
        training=False,
        args=args
    )
    if args.detector_name=="LZnet":
        model = FPVdet(dataset=test_set,logger=logger)
    elif args.detector_name=="PVRCNN":
        model = Part2net(num_class=len(cfg.CLASS_NAMES),dataset=test_set)
    else:
        raise NotImplementedError


    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model,test_dataloader,cfg,eval_output_dir,logger,ckpt_dir)
        else:
            eval_single_ckpt(model,test_dataloader,eval_output_dir,logger,ckpt_dir,epoch_id,args)

def eval_single_ckpt(model,test_loader,
                     eval_output_dir,logger,ckpt_dir,epoch_id,args):
    if isinstance(epoch_id,(int,float)):
        epoch_id = str(epoch_id)
    ckpt_filename = os.path.join(ckpt_dir, "checkpoint_epoch_%s.pth" % epoch_id)

    if args.ckpt is not None:
        model.load_params_from_file(filename=args.ckpt, logger=logger)
    else:
        model.load_params_from_file(filename=ckpt_filename,logger=logger)
    model.cuda()
    #logger.info(model)
    eval_utils.eval_one_epoch(
        model,test_loader,epoch_id,logger,
        result_dir=eval_output_dir,
        save_to_file=args.save_to_file
    )

def detect_specific_frame():
    args = parse_config()

    output_dir = os.path.join(cfg.CODE_DIR, "output", cfg.TAG, args.extra_tag)
    os.makedirs(output_dir, exist_ok=True)
    eval_output_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_output_dir,exist_ok=True)
    log_file = os.path.join(eval_output_dir, "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file)

    test_set, test_dataloader, sampler = build_data_loader(
        dist=False,
        data_path=cfg.DATA_DIR,
        batch_size=args.batch_size,
        num_workers=cfg.DATA_CONFIG.NUM_WORKERS,
        logger=logger,
        split=args.split,
        training=False,
        args=args
    )
    model = build_netword(num_class=len(cfg.CLASS_NAMES), dataset=test_set, logger=logger)
    model.load_params_from_file(args.ckpt, logger=logger)
    # for fix,id in enumerate(test_set.sample_idx_list):
    #     if id == "000137":
    #         break
    for i,data in enumerate(test_dataloader):
        if i== 58:
            break
    input = example_convert_to_torch(data)
    model.cuda()
    model.eval()
    output,_ = model(input)
    print(output)

if __name__ == '__main__':
    # detect_specific_frame()
    main()