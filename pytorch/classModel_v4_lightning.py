from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision import transforms
#lightning
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler 

'''
用pytorch-lightning包装classModel_v2_api.py
数据集配置参数模块
训练配置参数模块
基于LightningDataModule的数据加载和数据集定义模块
基于LightningModule的模型定义模块
基于TensorBoardLogger的日志记录模块
基于ModelCheckpoint的模型保存模块
基于PyTorchProfiler的性能分析模块
基于pytorch_lightning.Trainer的训练器来集成上述模块
'''
class DataConfiguration:
    batch_size:int=32
    num_classes:int=10
    train_valid_ratio:float=0.8
    data_root:str="../data"
    num_workers:int=4

class TrainConfiguration:
    lr:float=0.1
    optimizer:str="SGD"
    epochs:int=5
    train_logs:str="classModel-logs"

#alias TrainConfiguration to train_config
train_config=TrainConfiguration()

#custom transform
def img_prerpocess_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

#数据模块
class LitDataModule(pl.LightningDataModule):
    def __init__(self,data_config:DataConfiguration):
        super().__init__()
        self.data_config=data_config
        self.train_transform=img_prerpocess_transform()
        self.test_transform=img_prerpocess_transform()
    #下载数据集
    def prepare_data(self) -> None:
        torchvision.datasets.FashionMNIST(
            root=self.data_config.data_root,
            train=True,#train set
            transform=self.train_transform,
            download=True
        )
        torchvision.datasets.FashionMNIST(
            root=self.data_config.data_root,
            train=False,#test set
            transform=self.test_transform,
            download=True
        )
    #数据加载
    def setup(self,stage=None):
        if stage=="fit" or stage is None:
            data=torchvision.datasets.FashionMNIST(
                root=self.data_config.data_root,
                train=True,
                transform=self.train_transform,
            )
            num_train=int(len(data)*self.data_config.train_valid_ratio)
            num_valid=len(data)-num_train
            self.train_data,self.valid_data=torch.utils.data.random_split(data,[num_train,num_valid])
        if stage=="test" or stage is None:
            self.test_data=torchvision.datasets.FashionMNIST(
                root=self.data_config.data_root,
                train=False,
                transform=self.test_transform)

    #生成数据加载器
    def train_dataloader(self):
        return data.DataLoader(self.train_data,batch_size=self.data_config.batch_size,shuffle=True,num_workers=self.data_config.num_workers,persistent_workers=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.valid_data,batch_size=self.data_config.batch_size,shuffle=False,num_workers=self.data_config.num_workers,persistent_workers=True)
    
    def test_dataloader(self):
        return data.DataLoader(self.test_data,batch_size=self.data_config.batch_size,shuffle=False,num_workers=self.data_config.num_workers)  

#模型模块
class LitClassModel(pl.LightningModule):
    def __init__(self,train_config:TrainConfiguration):
        super().__init__()
        self.train_config=train_config
        self.model=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
        self.model.apply(self.init_weights)#初始化参数
        self.loss=nn.CrossEntropyLoss(reduction='none')    
        #指标
        self.mean_train_loss=MeanMetric()#平均损失
        self.mean_train_acc=MulticlassAccuracy(num_classes=10)#多分类准确率
        self.mean_valid_loss=MeanMetric()#验证集平均损失
        self.mean_valid_acc=MulticlassAccuracy(num_classes=10)#验证集多分类准确率

    def init_weights(self,m):
        if type(m)==nn.Linear:
            nn.init.normal_(m.weight,std=0.01)

    def forward(self,data):
        return self.model(data)
    
    def training_step(self, batch, *args, **kwargs):
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y).mean()
        # 记录batch日志
        self.mean_train_loss.update(loss.item(), X.shape[0])
        self.mean_train_acc.update(y_hat.detach().argmax(dim=1), y)
        # self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=True)
        # self.log("train/batch_acc", self.mean_train_acc, prog_bar=True, logger=True)
        return loss.mean()
    
   
    #每个epoch结束后调用
    def on_train_epoch_end(self) -> None:
        #记录epoch日志
        # Computing and logging the training mean loss & mean f1.
        train_loss=self.mean_train_loss.compute()
        train_acc=self.mean_train_acc.compute()
        self.log("train/loss", train_loss, prog_bar=True, logger=True)
        self.log("train/acc", train_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)

    #验证集评估
    def validation_step(self,batch, *args: Any, **kwargs: Any):
        X,y=batch
        y_hat=self(X)
        loss=self.loss(y_hat,y).mean()
        #记录batch日志
        self.mean_valid_loss.update(loss.item(),X.shape[0])
        self.mean_valid_acc.update(y_hat.detach().argmax(dim=1),y)
        self.log("valid/batch_loss",self.mean_valid_loss,prog_bar=True,logger=True)
        self.log("valid/batch_acc",self.mean_valid_acc,prog_bar=True,logger=True)
        return loss

    #每个epoch结束后调用
    def on_validation_epoch_end(self) -> None:
        # Computing and logging the validation mean loss & mean f1.
        val_loss=self.mean_valid_loss.compute()
        val_acc=self.mean_valid_acc.compute()
        self.log("valid/loss", val_loss, prog_bar=True, logger=True)
        self.log("valid/acc", val_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)



    def configure_optimizers(self):
        optimizer=getattr(torch.optim,self.train_config.optimizer)(
            self.model.parameters(),
            lr=self.train_config.lr
        )
        return optimizer
    
if __name__=="__main__":
        
    #固定随机种子,保证实验可复现性
    pl.seed_everything(42,workers=True)

    #创建数据模块
    data_module=LitDataModule(DataConfiguration())
    data_module.prepare_data()#下载数据
    data_module.setup()#加载数据
    train_loader=data_module.train_dataloader()#训练集加载器

    #创建模型
    model=LitClassModel(train_config)

    #监控某个指标,每次达到最好时保存当前模型
    model_checkpoint = ModelCheckpoint(
        monitor="valid/acc",  #监控验证集的acc
        mode="max",  #{min,max},监控acc用max,监控loss用min
        dirpath="./",
        filename="classModel_v4_{epoch:03d}_{valid/acc:.2f}",
        auto_insert_metric_name=False, #false用于自定义filename
        save_weights_only=True
    )

    #创建tensorboard日志
    tb_logger = TensorBoardLogger(save_dir=train_config.train_logs, name="classModel_v4")

    #创建性能分析器
    profiler = PyTorchProfiler(dirpath=train_config.train_logs,filename="classModel_v4_profiler")

    #创建训练器
    trainer=pl.Trainer(
        accelerator="auto",#自动选择加速器
        devices="auto",#自动选择设备
        strategy="auto",#自动选择策略
        max_epochs=train_config.epochs,#最大训练轮数
        profiler=profiler,#性能分析器
        logger=tb_logger,#tensorboard日志
        callbacks=[model_checkpoint],#回调函数
        enable_model_summary=True#启用模型摘要
    )

    #开始训练
    trainer.fit(model,data_module)