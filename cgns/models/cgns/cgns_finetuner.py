import sys
sys.path.append('/home/***/project/cgns')

import datetime
import os
from argparse import ArgumentParser

import torch
from cgns.prior_resnet import ResNet
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

os.environ["WANDB_MODE"] = "offline"
from pytorch_lightning.loggers import WandbLogger

from cgns.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from cgns.datasets.data_module import DataModule_fintune,DataModule
from cgns.datasets.transforms import DataTransforms, Moco2Transform
from cgns.models.backbones.vits import create_vit
from cgns.models.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home1/***/cgns/'

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--model_name", type=str,
                        default="***.pth")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--data_pct", type=float, default=1)
    parser.add_argument("--backbone_name", type=str, default="resnet_50")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 100
    ckpt_path = os.path.join("/home1/***/checkpoints/", args.model_name)
    seed_everything(args.seed)

    if args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule_fintune(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
        # datamodule.prepare_data_h5()
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.backbone_name == 'resnet_50':
        model = ResNet(name='resnet_50',in_channels=3,pretrained=True)
        weights = torch.load(ckpt_path)
        state_dict_with_prefix = ResNet.add_prefix_to_state_dict(weights, 'encoder.')
        model.load_state_dict(state_dict_with_prefix, strict=True)
        args.backbone = model

        args.in_features = 2048
        args.num_classes = num_classes
        args.multilabel = multilabel

    elif args.backbone_name == 'vit_base':
        vit_name = 'base'
        image_size = 224
        vit_grad_ckpt = False
        vit_ckpt_layer = 0
        model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        weights = torch.load(ckpt_path)
        model.load_state_dict(weights, strict=True)
        args.backbone = model

        args.in_features = 768
        args.num_classes = num_classes
        args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__) 
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ckpt_dir = os.path.join(
    #     BASE_DIR, f"../../../data/ckpts/mgca_finetune/{extension}")
    ckpt_dir = os.path.join(
        BASE_DIR, f"data/ckpts/cgns_finetune/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"), 
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),  
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # logger_dir = os.path.join(
    #     BASE_DIR, f"../../../data/wandb")
    logger_dir = os.path.join(
        BASE_DIR, f"data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="mgca_finetune",
        save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True, 
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)

    # test
    trainer.test(tuner, datamodule, ckpt_path="best")
if __name__ == "__main__":
    cli_main()
