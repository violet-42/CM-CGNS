from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import AUROC, Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


class SSLFineTuner(LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 model_name: str = "resnet_50",
                 in_features: int = 2048,
                 num_classes: int = 5,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 multilabel: bool = True,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if multilabel:
            # self.train_auc = AUROC(task='multilabel',num_labels=num_classes)
            # self.val_auc = AUROC(task='multilabel',num_labels=num_classes,
            #                      compute_on_step=False)
            # self.test_auc = AUROC(task='multilabel',num_labels=num_classes,
            #                       compute_on_step=False)

            self.train_auc = AUROC(num_classes=num_classes)
            self.val_auc = AUROC(num_classes=num_classes,
                                 compute_on_step=False)
            self.test_auc = AUROC(num_classes=num_classes,
                                  compute_on_step=False)
        else:
            self.train_acc = Accuracy(task='multiclass',num_classes=num_classes, topk=1)
            self.val_acc = Accuracy(
                task='multiclass',num_classes=num_classes, topk=1, compute_on_step=False)
            self.test_acc = Accuracy(
                task='multiclass',num_classes=num_classes, topk=1, compute_on_step=False)

        self.save_hyperparameters(ignore=['backbone'])

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.encoder.layer4.parameters():
            param.requires_grad = True
        self.model_name = model_name
        self.multilabel = multilabel

        self.linear_layer = SSLEvaluator(
            n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            auc = self.train_auc(torch.sigmoid(logits).float(), y.long())
            self.log("train_auc_step", auc, prog_bar=True, sync_dist=True)
            self.log("train_auc_epoch", self.train_auc,
                     prog_bar=True, sync_dist=True)
        else:
            acc = self.train_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("train_acc_step", acc, prog_bar=True, sync_dist=True)
            self.log("train_acc_epoch", self.train_acc,
                     prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if self.multilabel:
            self.val_auc(torch.sigmoid(logits).float(), y.long())
            self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.val_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        if self.multilabel:
            self.test_auc(torch.sigmoid(logits).float(), y.long())
            # TODO: save probs and labels
            self.log("test_auc", self.test_auc, on_epoch=True)
        else:
            self.test_acc(F.softmax(logits, dim=-1).float(), y.long())
            self.log("test_acc", self.test_acc, on_epoch=True)

        return loss

    def shared_step(self, batch):
        x, y = batch
        # x = x.mean(1, keepdim=True) 
        # For multi-class
        with torch.no_grad():
            if "vit" in self.model_name:
                feats = self.backbone(x)
                feats = feats[:, 0]
            else:
                _, feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), y.float())
        else:
            y = y.squeeze()
            loss = F.cross_entropy(logits.float(), y.long())

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )
        T_max = self.hparams.max_epochs 
        eta_min = 0  

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', 
                'frequency': 1
            }
        }

        # return optimizer

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
