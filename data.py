import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision


class DmgDataset(data.Dataset):
    """
    损伤数据集类，继承自 torch.utils.data.Dataset。
    用于加载 CFRP（碳纤维增强聚合物）低速冲击（LVI）损伤的 .npy 数据文件。

    参数说明（文件命名约定）：
      - R：冲击半径（impactor radius）
      - E：冲击能量（impact energy）
      - T：冲击角度（impact angle θ）
    损伤类型选项（opt）：
      - 'SDEG'  ：单一退化量（15 层）
      - 'MTDMG' ：多层拉伸损伤（16 层）
      - 'MCDMG' ：多层压缩损伤（16 层）
      - 'MDMG'  ：混合损伤（16 层）
      - 'FDMG'  ：全场损伤（16 层）
    """

    def __init__(self, data_folder, opt='SDEG', train=True, device='cuda:0'):
        # 记录是否为训练集
        self.train = train
        # 记录损伤类型选项
        self.opt = opt

        # 根据损伤类型确定序列长度（层数）：
        # 'SDEG' 使用 15 层，其他类型使用 16 层
        if opt == "SDEG":
            self.sequence_length = 15
        else:
            self.sequence_length = 16

        # 根据 train 标志选择 'train' 或 'valid' 子目录
        folder = os.path.join(data_folder, 'train' if train else 'valid')

        # 列出该子目录下所有文件名
        all_npys = os.listdir(folder)

        # 筛选出以对应损伤类型结尾的 .npy 文件，并构建完整路径列表
        self.all_opt_npy = [os.path.join(folder, item) for item in all_npys if item.endswith(self.opt+'.npy')]

        # 记录目标设备（CPU 或 GPU）
        self.device = device

        # 定义高斯模糊变换，用于对拉伸/混合损伤类型做平滑处理
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(2.0, 2.0))

    def __len__(self):
        # 返回数据集中样本总数
        return len(self.all_opt_npy)

    def __getitem__(self, idx):
        # 步骤 1：加载 .npy 文件，取最后 sequence_length 层，
        #          并裁剪空间维度（去除边缘噪声：行去掉前2行和后3行，列同理）
        #          转换为 float32 后加载到指定设备
        seq_npy = torch.from_numpy(
            np.load(self.all_opt_npy[idx])[16 - self.sequence_length:, 2:-3, 2:-3].astype('float32')
        ).to(self.device)  # shape: (sequence_length, H, W)

        # 步骤 2：若为拉伸损伤（MTDMG）或混合损伤（MDMG），应用高斯模糊平滑数据
        if self.opt == "MTDMG" or self.opt == "MDMG":
            seq_npy = self.blur(seq_npy)

        # 步骤 3：从文件名中解析冲击载荷参数
        #   文件名格式：R{半径}_{能量}E_{角度}T_...{opt}.npy
        #   split('\\') 取最后一段文件名，split('_')[:3] 取前三段关键字
        tmp_kws = self.all_opt_npy[idx].split('\\')[-1].split('_')[:3]

        # 步骤 4：将字符串参数转换为归一化浮点数：
        #   R / 100，E / 1000，T / 100
        load = torch.from_numpy(
            np.array([
                float(tmp_kws[0][1:]) / 100,   # 冲击半径 R，归一化
                float(tmp_kws[1][1:]) / 1000,  # 冲击能量 E，归一化
                float(tmp_kws[2][1:]) / 100    # 冲击角度 T，归一化
            ]).astype('float32')
        ).to(self.device)  # shape: (3,)

        # 返回：载荷向量（3,）和损伤序列张量（sequence_length, H, W）
        return load, seq_npy


class DmgData(data.Dataset):
    """
    数据加载器封装类，提供训练集、验证集和测试集的 DataLoader。
    """

    def __init__(self, data_path, opt, batch_size, num_workers, device):
        # 数据集根目录路径
        self.data_path = data_path
        # 每个批次的样本数
        self.batch_size = batch_size
        # 数据加载的工作进程数
        self.num_workers = num_workers
        # 损伤类型选项
        self.opt = opt
        # 目标设备
        self.device = device

    def _dataloader(self, train):
        # 步骤 1：实例化 DmgDataset，选择训练集或验证集
        dataset = DmgDataset(self.data_path, opt=self.opt, train=train, device=self.device)

        # 步骤 2：构建 DataLoader
        #   - shuffle=train：训练时随机打乱，验证时不打乱
        #   - drop_last=True：丢弃最后一个不完整批次，保持批次大小一致
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=train,
            drop_last=True
        )
        return dataloader

    def train_dataloader(self):
        # 返回打乱顺序的训练集 DataLoader
        return self._dataloader(True)

    def val_dataloader(self):
        # 返回不打乱顺序的验证集 DataLoader
        return self._dataloader(False)

    def test_dataloader(self):
        # 返回不打乱顺序的测试集 DataLoader（与验证集使用相同数据）
        return self._dataloader(False)
