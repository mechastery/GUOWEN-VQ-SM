import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.autograd import Function


class VectorQuantizer(nn.Module):
    """
    标准向量量化模块（VQ），通过最近邻匹配将连续潜向量离散化到固定码本。

    参数：
      num_embeddings  (int)  ：码本大小（码字数量）
      embedding_dim   (int)  ：每个码字的维度
      commitment_cost (float)：承诺损失系数 β，用于约束编码器输出接近码本
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        # 保存码本维度和码字数量
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # 初始化可学习的码本嵌入矩阵，shape: (num_embeddings, embedding_dim)
        # 权重在 [-1/num_embeddings, 1/num_embeddings] 范围内均匀初始化
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        # 承诺损失系数
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # 步骤 1：将输入从 BCHW 格式转换为 BHWC 格式，方便按像素量化
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape  # (B, H, W, C)

        # 步骤 2：将空间维度展平，得到 (B*H*W, embedding_dim) 的二维张量
        flat_input = inputs.view(-1, self._embedding_dim)

        # 步骤 3：计算每个潜向量到所有码字的 L2 距离（展开的欧式距离公式）
        #   ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z·e^T
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # 步骤 4：找到每个潜向量最近的码字索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)

        # 步骤 5：构造 one-hot 编码矩阵，shape: (B*H*W, num_embeddings)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 步骤 6：通过 one-hot 编码从码本中查找对应码字，还原空间结构
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # 步骤 7：计算向量量化损失
        #   e_latent_loss：码本更新损失（码本向编码器输出靠近）
        #   q_latent_loss：承诺损失（编码器输出向码本靠近）
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # 步骤 8：直通估计器（Straight-Through Estimator）
        #   前向传播使用量化值，反向传播时梯度直接流向编码器输出
        quantized = inputs + (quantized - inputs).detach()

        # 步骤 9：计算困惑度（perplexity），衡量码本利用率
        #   困惑度越高表示码本利用越充分
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # 步骤 10：将量化结果从 BHWC 格式转回 BCHW 格式后返回
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """
    基于指数移动平均（EMA）的向量量化模块。
    码本通过 EMA 更新，而非梯度反传，训练更稳定。

    参数：
      num_embeddings  (int)   ：码本大小
      embedding_dim   (int)   ：码字维度
      commitment_cost (float) ：承诺损失系数
      decay           (float) ：EMA 衰减率（通常取 0.99 或 0.999）
      epsilon         (float) ：Laplace 平滑项，防止除以零
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        # 保存基本参数
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # 初始化码本嵌入矩阵（正态分布初始化）
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        # EMA 统计量：每个码字被分配的样本数量的指数移动平均
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))

        # EMA 权重：用于更新码本的指数移动平均值
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        # EMA 衰减率和平滑系数
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # 步骤 1：将输入从 BCHW 转换为 BHWC 格式
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape  # (B, H, W, C)

        # 步骤 2：展平空间维度，得到 (B*H*W, embedding_dim)
        flat_input = inputs.view(-1, self._embedding_dim)

        # 步骤 3：计算各潜向量到所有码字的 L2 距离
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # 步骤 4：找到最近码字的索引并构造 one-hot 编码
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 步骤 5：通过 one-hot 编码从码本中查找对应码字并还原空间结构
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # 步骤 6：训练阶段通过 EMA 更新码本（不通过梯度反传）
        if self.training:
            # 更新每个码字被分配的样本数量的 EMA 统计
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace 平滑：避免某些码字长期未被分配时 EMA 值过小
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            # 计算当前批次中分配到每个码字的潜向量之和
            dw = torch.matmul(encodings.t(), flat_input)

            # 更新码字的 EMA 权重
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            # 用 EMA 权重除以 EMA 分配计数，得到更新后的码本
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # 步骤 7：计算承诺损失（只约束编码器输出，不更新码本）
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # 步骤 8：直通估计器，保证梯度可以流向编码器
        quantized = inputs + (quantized - inputs).detach()

        # 步骤 9：计算困惑度衡量码本利用率
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # 步骤 10：将量化结果转回 BCHW 格式后返回
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ResBlock(nn.Module):
    """
    卷积残差块，包含两个卷积层和批归一化，配合跳连接缓解梯度消失。

    结构：ReLU → Conv2d(3×3) → BN → ReLU → Conv2d(1×1) → BN
    输出：x + block(x)（残差连接）
    """

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),   # 3×3 卷积保持空间尺寸
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),          # 1×1 卷积（通道混合）
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # 残差连接：将输入直接加到块的输出上
        return x + self.block(x)


class ResBlockL(nn.Module):
    """
    线性残差块，用于全连接网络中的残差连接。

    结构：ReLU → Linear → BN → ReLU → Linear → BN
    输出：x + block(x)
    """

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # 残差连接
        return x + self.block(x)


class VQVAE_Simp(nn.Module):
    """
    简化版 VQ-VAE（向量量化变分自编码器），用于 CFRP 损伤场的编码与重建。

    结构：
      Encoder：3 层步长为 2 的卷积（下采样 8×）+ 4 个残差块 + 1×1 卷积映射到嵌入维度
      VQ 层：标准 VQ 或 EMA-VQ（由 decay 参数决定）
      Decoder：1×1 卷积恢复通道 + 4 个残差块 + 3 层转置卷积（上采样 8×）+ Tanh 激活

    参数：
      opt           (str)   ：损伤类型，决定输入/输出通道数
      dim           (int)   ：中间卷积层的通道数
      embedding_dim (int)   ：VQ 码字的维度
      num_embeddings(int)   ：码本大小
      beta          (float) ：VQ 承诺损失系数
      decay         (float) ：EMA 衰减率，>0 时使用 EMA-VQ，否则使用标准 VQ
    """

    def __init__(self, opt, dim=64, embedding_dim=64, num_embeddings=2048, beta=0.25, decay=0):
        super().__init__()

        # 根据损伤类型确定输入/输出通道数
        if opt == 'SDEG':
            self.input_dim = 15   # SDEG 数据有 15 层
        else:
            self.input_dim = 16   # 其他损伤类型有 16 层

        # 根据 decay 参数选择向量量化模块
        if decay > 0:
            # 使用 EMA 更新码本（更稳定）
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, beta, decay)
        else:
            # 使用标准梯度更新码本
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # 数据方差（用于归一化重建损失，默认为 1.0）
        self.data_variance = 1.0

        # 编码器：将输入（B, input_dim, H, W）压缩到潜空间（B, embedding_dim, H/8, W/8）
        self.Encoder = nn.Sequential(
            # 第 1 层下采样：步长 2，空间尺寸减半
            nn.Conv2d(in_channels=self.input_dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # 第 2 层下采样
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # 第 3 层下采样
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            # 4 个残差块提升特征表达能力
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            # 1×1 卷积将通道数映射到 embedding_dim
            nn.Conv2d(dim, embedding_dim, kernel_size=1, stride=1, padding=0),
        )

        # 解码器：将量化潜向量（B, embedding_dim, H/8, W/8）重建为原始尺寸（B, input_dim, H, W）
        self.Decoder = nn.Sequential(
            # 1×1 卷积将 embedding_dim 恢复为 dim
            nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),
            # 4 个残差块
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            # 第 1 层上采样：转置卷积，空间尺寸翻倍
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            # 第 2 层上采样
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            # 第 3 层上采样，输出通道数恢复为 input_dim
            nn.ConvTranspose2d(dim, self.input_dim, 4, 2, 1),
            # Tanh 激活将输出限制到 [-1, 1]
            nn.Tanh()
        )

    def encode(self, x):
        """
        编码过程：
          步骤 1：将输入 x 通过编码器得到连续潜向量 z_e_x
          步骤 2：通过向量量化层将 z_e_x 离散化，返回量化损失和量化后的潜向量
        返回：(loss_vq, quantized, perplexity, encodings)
          - loss_vq   ：向量量化损失
          - quantized ：量化后的潜向量（one-hot 编码对应的码字张量）
          - perplexity：码本利用率
          - encodings ：one-hot 编码矩阵
        """
        z_e_x = self.Encoder(x)
        loss, quantized, perplexity, encodings = self._vq_vae(z_e_x)
        return loss, quantized, perplexity, encodings

    def decode(self, latents):
        """
        解码过程：将量化潜向量通过解码器重建为损伤场
        参数：latents - 量化后的潜向量
        返回：x_recon - 重建的损伤场张量
        """
        x_recon = self.Decoder(latents)
        return x_recon

    def training_step(self, x):
        """
        训练步骤（包含梯度计算）：
          步骤 1：编码输入 x，得到量化损失和量化潜向量
          步骤 2：解码量化潜向量，得到重建结果
          步骤 3：计算归一化重建损失（MSE / data_variance）
          步骤 4：合并重建损失和 VQ 损失
        返回：(x_recon, loss, loss_recons, loss_vq)
        """
        loss_vq, quantized, perplexity, encodings = self.encode(x)
        x_recon = self.decode(quantized)
        loss_recons = F.mse_loss(x_recon, x) / self.data_variance
        loss = loss_recons + loss_vq
        return x_recon, loss, loss_recons.item(), loss_vq.item()

    def validation_step(self, x):
        """
        验证步骤（不需要梯度）：
          与训练步骤逻辑相同，但返回标量损失值（.item()），不保留计算图
        返回：(x_recon, loss, loss_recons, loss_vq)，损失均为 Python 标量
        """
        loss_vq, quantized, perplexity, encodings = self.encode(x)
        x_recon = self.decode(quantized)
        loss_recons = F.mse_loss(x_recon, x) / self.data_variance
        loss = loss_recons + loss_vq
        return x_recon, loss.item(), loss_recons.item(), loss_vq.item()


class Predictor(nn.Module):
    """
    载荷预测器：从载荷参数（R, E, T）预测 VQ-VAE 的潜空间表示。

    结构：全连接网络（MLP），将 3 维载荷向量映射为
    (embedding_dim, size, size) 的空间特征图。

    参数：
      embedding_dim (int)：潜向量的通道维度（与 VQ-VAE 保持一致）
      size          (int)：空间特征图的高度和宽度（H/8 = 96/8 = 12）
    """

    def __init__(self, embedding_dim=64, size=12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.size = size

        # MLP：3 → embedding_dim → embedding_dim → embedding_dim * size * size
        self.predictor = nn.Sequential(
            nn.Linear(3, embedding_dim),              # 输入：3 维载荷向量
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),  # 隐藏层
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim * size * size),  # 输出：展平的空间特征
        )

    def forward(self, x):
        # 步骤 1：获取批次大小 b 和特征维度 v（此处 v=3）
        b, v = x.size()

        # 步骤 2：通过 MLP 将载荷向量映射为展平的潜向量
        recon_vect = self.predictor(x)  # shape: (b, embedding_dim * size * size)

        # 步骤 3：将展平向量 reshape 为空间特征图
        latents = torch.reshape(recon_vect, (b, self.embedding_dim, self.size, self.size))
        return latents  # shape: (b, embedding_dim, size, size)


class simple_CNN(nn.Module):
    """
    简单 CNN 端到端预测器（基线对比模型）：直接从载荷参数预测损伤场。

    结构：
      MLP：3 → embedding_dim → embedding_dim → embedding_dim * size * size
      Decoder（同 VQ-VAE 的解码器结构）：
        1 个 3×3 卷积 + 4 个残差块 + 3 层转置卷积 + Tanh

    参数：
      opt           (str)：损伤类型（决定输出通道数）
      embedding_dim (int)：中间特征维度
      size          (int)：中间空间特征图尺寸
      dim           (int)：解码器卷积通道数
    """

    def __init__(self, opt, embedding_dim=64, size=12, dim=64):
        super().__init__()

        # 根据损伤类型确定输出通道数
        if opt == 'SDEG':
            self.input_dim = 15
        else:
            self.input_dim = 16
        self.embedding_dim = embedding_dim
        self.size = size

        # MLP 预测器：将 3 维载荷向量映射为展平的潜向量
        self.predictor = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim * size * size),
        )

        # 单独的残差块（备用，当前 forward 中未直接调用）
        self.res_block = ResBlock(dim=embedding_dim)

        # 解码器（与 VQ-VAE 解码器结构相同）
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, self.input_dim, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 步骤 1：获取批次大小
        b, v = x.size()

        # 步骤 2：通过 MLP 将载荷向量映射为展平的潜向量
        recon_vect = self.predictor(x)  # shape: (b, embedding_dim * size * size)

        # 步骤 3：将展平向量 reshape 为空间特征图
        latents = torch.reshape(recon_vect, (b, self.embedding_dim, self.size, self.size))

        # 步骤 4：通过解码器重建损伤场
        pre = self.decoder(latents)  # shape: (b, input_dim, H, W)
        return pre


class simple_NN(nn.Module):
    """
    简单全连接神经网络预测器（基线对比模型）：纯 MLP 版本，不使用卷积。

    结构：
      MLP 编码器：3 → embedding_dim → embedding_dim → embedding_dim * size * size
      MLP 解码器：多层全连接 + 残差块 + 最终输出 → input_dim * size * size

    参数与 simple_CNN 相同。
    """

    def __init__(self, opt, embedding_dim=64, size=12, dim=64):
        super().__init__()

        # 根据损伤类型确定输出通道数
        if opt == 'SDEG':
            self.input_dim = 15
        else:
            self.input_dim = 16
        self.embedding_dim = embedding_dim
        self.size = size

        # MLP 编码器：载荷向量 → 潜向量
        self.predictor = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim * size * size),
        )

        # 线性残差块（用于全连接解码器）
        self.res_block = ResBlockL(dim=dim * size * size)

        # MLP 解码器：展平潜向量 → 损伤场（展平形式）
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * size * size, dim * size * size),
            ResBlock(dim * size * size),
            ResBlock(dim * size * size),
            ResBlock(dim * size * size),
            ResBlock(dim * size * size),
            nn.ReLU(True),
            nn.Linear(dim * size * size, dim * size * size * 4),
            nn.ReLU(True),
            nn.Linear(dim * size * size, dim * size * size * 4),
            nn.ReLU(True),
            nn.Linear(dim * size * size, self.input_dim * size * size),
            nn.Tanh()
        )

    def forward(self, x):
        # 步骤 1：获取批次大小
        b, v = x.size()

        # 步骤 2：通过 MLP 将载荷向量映射为展平的潜向量
        recon_vect = self.predictor(x)  # shape: (b, embedding_dim * size * size)

        # 步骤 3：通过全连接解码器得到展平的损伤场输出
        pre = self.decoder(recon_vect)  # shape: (b, input_dim * size * size)

        # 步骤 4：将展平输出 reshape 为空间图像格式
        pre_imgs = torch.reshape(pre, (b, self.input_dim, self.size, self.size))
        return pre_imgs
