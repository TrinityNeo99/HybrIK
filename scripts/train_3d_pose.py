"""Script for multi-gpu training."""
import os
import pickle as pk
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from torch.nn.utils import clip_grad
sys.path.append("./")
from hybrik.datasets import MixDataset, MixDatasetCam, PW3D, MixDataset2Cam, Binocuular_coco
from hybrik.models import builder
from hybrik.opt import cfg, logger, opt
from hybrik.utils.env import init_dist
from hybrik.utils.metrics import DataLogger, NullWriter, calc_coord_accuracy
from hybrik.utils.transforms import get_func_heatmap_to_coord
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# torch.set_num_threads(64)
num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu


def _init_fn(worker_id):
    np.random.seed(opt.seed + worker_id)
    random.seed(opt.seed + worker_id)


def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def train(opt, train_loader, m, criterion, optimizer, writer, epoch_num):
    print("train start...")
    loss_logger = DataLogger()
    acc_xyz_17_logger = DataLogger()
    m.train()

    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)
    for j, (inps, labels, _) in enumerate(train_loader):
        # if j > 1:
        #     break  # break for early stop
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu).requires_grad_() for inp in inps]
        else:
            inps = inps.cuda(opt.gpu).requires_grad_()
        for k, _ in labels.items():
            labels[k] = labels[k].cuda(opt.gpu)

        output = m(inps)

        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)

        loss = loss_mpjpe(pred_xyz_jts_17, labels['joint_cam_17'])

        # print("MPJPE (mm): ", loss.item()*1000)
        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        for group in optimizer.param_groups:
            for param in group["params"]:
                clip_grad.clip_grad_norm_(param, 5)

        optimizer.step()

        opt.trainIters += 1
        if opt.log:
            # TQDM
            train_loader.set_description(
                'loss: {loss:.8f}'.format(
                    loss=loss_logger.avg)
            )

    if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_xyz_17_logger.avg


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=24, pred_root=False):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
    kpt_pred = {}
    m.eval()


    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    sample_mpjpe = []
    for inps, labels, img_ids in gt_val_loader:
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        # print(inps.shape)
        output = m(inps, flip_test=opt.flip_test)

        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        sample_mpjpe.append(mpjpe(pred_xyz_jts_17, labels['joint_cam_17'].cpu().data.numpy()))

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        mpjpe_m = np.mean(sample_mpjpe)*1000
        return mpjpe_m


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    if opt.seed is not None:
        setup_seed(opt.seed)

    if opt.launcher == 'slurm':
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):
    if opt.seed is not None:
        setup_seed(opt.seed)

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    if not opt.log:
        logger.setLevel(50)
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    opt.nThreads = int(opt.nThreads / num_gpu)

    # Model Initialize
    m = preset_model(cfg)
    if opt.params:
        from thop import clever_format, profile
        input = torch.randn(1, 3, 256, 256).cuda(opt.gpu)
        flops, params = profile(m.cuda(opt.gpu), inputs=(input, ))
        macs, params = clever_format([flops, params], "%.3f")
        logger.info(macs, params)

    m.cuda(opt.gpu)
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu], find_unused_parameters=True)
    criterion = builder.build_loss(cfg.LOSS).cuda(opt.gpu)
    optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    if opt.log:
        writer = SummaryWriter('.tensorboard/{}/{}-{}'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
    else:
        writer = None

    print(cfg.DATASET.DATASET)
    if cfg.DATASET.DATASET == 'mix_smpl':
        train_dataset = MixDataset(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix_smpl_cam':
        train_dataset = MixDatasetCam(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix2_smpl_cam':
        train_dataset = MixDataset2Cam(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == "pw3d":
        train_dataset = PW3D(
            cfg=cfg,
            ann_file="3DPW_train_new.json",
            train=True)
    elif cfg.DATASET.DATASET == "binocular_coco":
        train_dataset = Binocuular_coco(
            cfg=cfg,
            ann_file="pingpong_front_left_image_3d_joint_train.json",
            train=True)
    else:
        raise NotImplementedError

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=opt.nThreads, sampler=train_sampler, worker_init_fn=_init_fn, pin_memory=True)

    # gt val dataset
    if cfg.DATASET.DATASET == 'mix_smpl':
        gt_val_dataset_h36m = MixDataset(
            cfg=cfg,
            train=False)
    elif cfg.DATASET.DATASET == 'mix_smpl_cam' or cfg.DATASET.DATASET == 'mix2_smpl_cam':
        gt_val_dataset_h36m = MixDatasetCam(
            cfg=cfg,
            train=False)
    elif cfg.DATASET.DATASET == "pw3d":
        gt_val_dataset_3dpw = PW3D(
            cfg=cfg,
            ann_file='3DPW_test_new.json',
            train=False)
    elif cfg.DATASET.DATASET == "binocular_coco":
        gt_val_dataset_binocular_coco = Binocuular_coco(
            cfg=cfg,
            ann_file="pingpong_front_left_image_3d_joint_test.json",
            train=False)
    else:
        raise NotImplementedError

    opt.trainIters = 0
    best_err_h36m = 999
    best_err_3dpw = 999

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        train_sampler.set_epoch(i)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, acc17 = train(opt, train_loader, m, criterion, optimizer, writer, i)
        logger.epochInfo('Train', opt.epoch, loss, acc17)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            if opt.log:
                # Save checkpoint
                torch.save(m.module.state_dict(), './exp/{}/{}-{}/model_{}.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id, opt.epoch))

            # Prediction Test
            with torch.no_grad():
                # gt_tot_err_h36m = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord)
                gt_tot_err_3dpw = validate_gt(m, opt, cfg, gt_val_dataset_binocular_coco, heatmap_to_coord)
                if opt.log:
                    # if gt_tot_err_h36m <= best_err_h36m:
                    #     best_err_h36m = gt_tot_err_h36m
                    #     torch.save(m.module.state_dict(), './exp/{}/{}-{}/best_h36m_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
                    if gt_tot_err_3dpw <= best_err_3dpw:
                        best_err_3dpw = gt_tot_err_3dpw
                        torch.save(m.module.state_dict(), './exp/{}/{}-{}/best_3dpw_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))

                    # logger.info(f'##### Epoch {opt.epoch} | h36m err: {gt_tot_err_h36m} / {best_err_h36m} | 3dpw err: {gt_tot_err_3dpw} / {best_err_3dpw} #####')
                    logger.info(f'##### Epoch {opt.epoch} | 3dpw err: {gt_tot_err_3dpw} / {best_err_3dpw} #####')

        torch.distributed.barrier()  # Sync

    torch.save(m.module.state_dict(), './exp/{}/{}-{}/final_DPG.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED, map_location='cpu'))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD, map_location='cpu')
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()
    print(model)
    return model


if __name__ == "__main__":
    main()
