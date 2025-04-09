import datetime
import os
from argparse import ArgumentParser
import sys

# CUDA_VISIBLE_DEVICES=0 python mgca_detector.py --devices 1 --model_name lovt.pth --dataset rsna --data_pct 1 --learning_rate 1e-3 --batchsize 24
sys.path.append('/home/***/project/cgns')
os.environ["WANDB_MODE"] = "offline"

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from cgns.datasets.data_module import DataModule
from cgns.datasets.detection_dataset import (OBJCXRDetectionDataset,
                                             RSNADetectionDataset)
from cgns.datasets.transforms import DetectionDataTransforms
from cgns.models.backbones.detector_backbone import ResNetDetector
from cgns.models.ssl_detector import SSLDetector

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home1/***/cgns/'

def cli_main():
    parser = ArgumentParser("Finetuning of object detection task for MGCA")
    parser.add_argument("--model_name", type=str,
                        default="CENS-JL22.pth")  # 4_stage2_best_resnet50.pth mgca_resnet50.pth
    parser.add_argument("--dataset", type=str,
                        default="rsna", help="rsna or object_cxr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)  # 可调整
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--backbone_name", type=str, default="resnet_50")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50
    args.accelerator = "gpu"
    seed_everything(args.seed)
    ckpt_path = os.path.join("/home1/***/checkpoints/", args.model_name)

    if args.dataset == "rsna":
        datamodule = DataModule(RSNADetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    elif args.dataset == "object_cxr":
        datamodule = DataModule(OBJCXRDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    else:
        raise RuntimeError(f"{args.dataset} does not exist!")

    args.img_encoder = ResNetDetector(model_name=args.backbone_name, pretrained=False)  # 只有cnn的
    ckpt_dict = torch.load(ckpt_path)
    args.img_encoder.model.load_state_dict(ckpt_dict, strict=False)

    # Freeze encoder
    for param in args.img_encoder.parameters():
        param.requires_grad = False
    # 如果你想解冻整个`layer4`，你可以这样做：
    # for param in args.img_encoder.model.layer4.parameters():
    #     param.requires_grad = True

    model = SSLDetector(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ckpt_dir = os.path.join(
    #     BASE_DIR, f"../../../data/ckpts/detection/{extension}")
    ckpt_dir = os.path.join(
        BASE_DIR, f"data/ckpts/detection/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_mAP", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_mAP", min_delta=0.,
                      patience=10, verbose=False, mode="max")
    ]
    # logger_dir = os.path.join(
    #     BASE_DIR, f"../../../data")
    logger_dir = os.path.join(
        BASE_DIR, f"data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="detection", save_dir=logger_dir,
        name=f"CGNS_{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger
    )
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
