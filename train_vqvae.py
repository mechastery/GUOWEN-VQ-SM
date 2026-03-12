from networks import VQVAE_Simp, Predictor, simple_CNN
from data import DmgData
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt


def train_vqvae(opt='SDEG'):
    """
    训练 VQ-VAE（向量量化变分自编码器）。

    该函数执行以下步骤：
      1. 加载数据集（训练集和验证集）
      2. 初始化或恢复 VQ-VAE 模型及优化器
      3. 按 epoch 迭代训练，计算并反传损失
      4. 每 10 个 epoch 在验证集上评估，保存最优和最新模型权重
      5. 每 200 个 epoch 绘制并保存损失曲线图

    参数：
      opt (str)：损伤类型，如 'SDEG'、'MTDMG'、'MCDMG'、'FDMG'、'MDMG'
    """
    # --- 超参数与路径配置 ---
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'  # 数据集根目录
    param_dir = r'params\\'                         # 模型权重和记录文件保存目录
    batch_size = 4                                   # 每批样本数
    num_workers = 0                                  # 数据加载线程数（Windows 下通常为 0）
    device = torch.device('cuda:0')                  # 使用第一块 GPU

    # --- 步骤 1：构建数据加载器 ---
    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    train_dl = data_loader.train_dataloader()   # 训练集 DataLoader（随机打乱）
    valid_dl = data_loader.val_dataloader()     # 验证集 DataLoader（不打乱）

    # --- 模型超参数 ---
    embedding_dim = 16    # VQ 码字维度
    num_embeddings = 256  # 码本大小
    num_epoch = 800       # 最大训练轮数

    # --- 步骤 2：初始化模型 ---
    model = VQVAE_Simp(opt, embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)

    # --- 步骤 3：尝试从上次中断处恢复训练 ---
    if os.path.exists(param_dir + opt + "_%d_%d_coder_last.pth"%(embedding_dim, num_embeddings)):
        # 加载上次保存的模型权重
        model.load_state_dict(torch.load(param_dir + opt + "_%d_%d_coder_last.pth"%(embedding_dim, num_embeddings)))
        # 加载损失记录（用于确定当前已训练的 epoch 数）
        loss_rec = np.loadtxt(param_dir + opt + "_%d_%d_recorder.txt"%(embedding_dim, num_embeddings), delimiter=',').tolist()
    else:
        # 首次训练，初始化损失记录（格式：[epoch, lr, train_loss, train_mse, valid_loss, valid_mse]）
        loss_rec = [[0, 0.0003, 100, 100, 100, 100]]

    # --- 配置优化器 ---
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 根据已有记录推算起始 epoch（每 10 个 epoch 记录一次）
    epoch = (len(loss_rec) - 1) * 10

    # --- 步骤 4：训练主循环 ---
    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []      # 当前 epoch 的总损失缓冲
        tmp_loss_mse_rec = []  # 当前 epoch 的 MSE 重建损失缓冲

        # 遍历训练集中的每个批次
        for idx, [load, sql_npy] in enumerate(train_dl):
            optimizer.zero_grad()  # 清空梯度

            # 前向传播：计算重建结果及各项损失
            x_recon, loss, loss_recons, loss_vq = model.training_step(sql_npy)

            loss.backward()       # 反向传播，计算梯度
            optimizer.step()      # 更新模型参数

            tmp_loss_rec.append(loss.item())     # 记录总损失
            tmp_loss_mse_rec.append(loss_recons) # 记录重建损失

        # --- 步骤 5：每 10 个 epoch 进行一次验证评估 ---
        if epoch % 10 == 0 or epoch == num_epoch - 1:
            # 计算训练集平均损失
            train_loss_mean = np.mean(np.array(tmp_loss_rec))
            train_loss_mse_mean = np.mean(np.array(tmp_loss_mse_rec))

            # 清空缓冲，准备累积验证集损失
            tmp_loss_rec = []
            tmp_loss_mse_rec = []

            # 遍历验证集
            for idx, [load, sql_npy] in enumerate(valid_dl):
                x_recon, loss, loss_recons, loss_vq = model.validation_step(sql_npy)
                tmp_loss_rec.append(loss)
                tmp_loss_mse_rec.append(loss_recons)

            # 计算验证集平均损失
            valid_loss_mean = np.mean(np.array(tmp_loss_rec))
            valid_loss_rec_mean = np.mean(np.array(tmp_loss_mse_rec))

            # 追加本次评估记录到损失列表
            loss_rec.append([epoch, lr, train_loss_mean, train_loss_mse_mean, valid_loss_mean, valid_loss_rec_mean])

            # --- 步骤 6：保存最优模型（验证重建损失最低时） ---
            if valid_loss_rec_mean < np.min(np.array(loss_rec[:-1])[:, -1]):
                torch.save(model.state_dict(), param_dir + opt + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings))

            # 打印当前进度
            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f, Rec loss: %.5f" %
                  (epoch, train_loss_mean, valid_loss_mean, valid_loss_rec_mean))

            # 保存最新模型权重和损失记录
            torch.save(model.state_dict(), param_dir + opt + "_%d_%d_coder_last.pth"%(embedding_dim, num_embeddings))
            np.savetxt(param_dir + opt + "_%d_%d_recorder.txt"%(embedding_dim, num_embeddings), np.array(loss_rec), delimiter=',')

            # --- 步骤 7：每 200 个 epoch 绘制损失曲线并可视化重建结果 ---
            if epoch%200 == 0:
                plt.plot(np.log(np.array(loss_rec)[1:, 2]), label='Train Loss')
                plt.plot(np.log(np.array(loss_rec)[1:, 3]), label='Train Rec Loss')
                plt.plot(np.log(np.array(loss_rec)[1:, 4]), label='Valid Loss')
                plt.plot(np.log(np.array(loss_rec)[1:, 5]), label='Valid Rec Loss')
                plt.title(opt + ': Dim %d_ Num %d'%(embedding_dim, num_embeddings))
                plt.legend()
                plt.savefig(param_dir + opt + '_Dim%d_Num%d'%(embedding_dim, num_embeddings) + '.png')
                plt.show()
                show(x_recon, sql_npy, load, idx=0)  # 可视化最后一批次的重建结果


def train_predictor(opt='SDEG'):
    """
    训练载荷预测器（Predictor）。

    该函数的训练流程：
      1. 加载已训练好的 VQ-VAE 编码器（冻结，仅用于提取潜向量标签）
      2. 训练 Predictor MLP，使其输出的潜向量尽量接近真实损伤的量化潜向量
      3. 损失函数 = MSE(重建损伤场, 真实损伤场) + 0.1 * MSE(预测潜向量, 真实潜向量)
      4. 每 eval_step 个 epoch 在验证集上评估，保存最优模型

    参数：
      opt (str)：损伤类型
    """
    # --- 超参数与路径配置 ---
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'params\\'
    batch_size = 4
    num_workers = 0
    device = torch.device('cuda:0')

    # --- 步骤 1：构建数据加载器 ---
    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    train_dl = data_loader.train_dataloader()
    valid_dl = data_loader.val_dataloader()

    # --- 模型超参数 ---
    embedding_dim = 16
    num_embeddings = 256

    # --- 步骤 2：加载已训练好的 VQ-VAE 编码器（冻结参数，仅做推理） ---
    coder_model = VQVAE_Simp(opt, embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model.load_state_dict(torch.load(param_dir + opt + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model.eval()  # 关闭 Dropout/BN 的训练模式，固定码本

    # --- 步骤 3：初始化 Predictor ---
    predictor = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    if os.path.exists(param_dir + opt + "_predictor_best.pth"):
        # 从上次训练中断处恢复
        predictor.load_state_dict(torch.load(param_dir + opt + "_predictor_best.pth"))
        loss_rec = np.loadtxt(param_dir + opt + "_predictor_recorder.txt", delimiter=',').tolist()
        if len(loss_rec) <= 1:
            loss_rec = [[0, 0.0003, 100, 100]]
    else:
        loss_rec = [[0, 0.0003, 100, 100]]

    # --- 训练配置 ---
    num_epoch = 2000
    eval_step = 10   # 每隔多少 epoch 做一次验证
    lr = 1e-4
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    epoch = (len(loss_rec) - 1) * 10

    # --- 步骤 4：训练主循环 ---
    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []  # 当前 epoch 损失缓冲

        for idx, [load, sql_npy] in enumerate(train_dl):
            optimizer.zero_grad()

            # 步骤 4a：Predictor 从载荷向量预测潜向量
            latent = predictor(load)

            # 步骤 4b：用冻结的 VQ-VAE 编码器获取真实损伤的量化潜向量（作为标签）
            _, latent_label, _, _ = coder_model.encode(sql_npy)

            # 步骤 4c：计算潜向量对齐损失（引导 Predictor 输出靠近真实潜空间）
            latent_loss = torch.nn.functional.mse_loss(latent_label, latent)

            # 步骤 4d：用 VQ-VAE 解码器将预测潜向量解码为损伤场
            pre_sql = coder_model.decode(latent)

            # 步骤 4e：综合损失 = 重建损失 + 0.1 * 潜向量对齐损失
            loss = torch.nn.functional.mse_loss(pre_sql, sql_npy) + latent_loss * 0.1

            loss.backward()       # 反向传播
            optimizer.step()      # 更新 Predictor 参数
            tmp_loss_rec.append(loss.item())

        # --- 步骤 5：每 eval_step 个 epoch 进行验证 ---
        if epoch % eval_step == 0 or epoch == num_epoch - 1:
            train_loss_mean = np.mean(np.array(tmp_loss_rec))
            tmp_loss_rec = []
            tmp_loss_mse_rec = []

            for idx, [load, sql_npy] in enumerate(valid_dl):
                # 验证时只计算重建损失（MSE），不需要潜向量标签
                latent = predictor(load)
                pre_sql = coder_model.decode(latent)
                loss = torch.nn.functional.mse_loss(pre_sql, sql_npy)
                tmp_loss_rec.append(loss.item())
                # 每 100 个 epoch 可视化一次验证结果
                if epoch%100 == 0:
                    show(pre_sql, sql_npy, load, idx=0, save=True)

            valid_loss_mean = np.mean(np.array(tmp_loss_rec))

            # 追加本次评估记录
            loss_rec.append([epoch, lr, train_loss_mean, valid_loss_mean])

            # 保存当前验证集损失最低的模型（最优模型）
            if valid_loss_mean < np.min(np.array(loss_rec[:-1])[:, 3]):
                torch.save(predictor.state_dict(), param_dir + opt + "_predictor_best.pth")

            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f" %
                  (epoch, train_loss_mean, valid_loss_mean))

            # 保存最新模型权重和损失记录
            torch.save(predictor.state_dict(), param_dir + opt + "_predictor_last.pth")
            np.savetxt(param_dir + opt + '_predictor_recorder.txt', np.array(loss_rec), delimiter=',')


def train_simCNN(opt='SDEG'):
    """
    训练简单 CNN 端到端预测器（baseline 对比模型）。

    该函数不依赖 VQ-VAE，而是直接训练一个 CNN 从载荷向量预测损伤场。
    损失函数 = MSE(预测损伤场, 真实损伤场)

    参数：
      opt (str)：损伤类型
    """
    # --- 超参数与路径配置 ---
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'params\\'
    batch_size = 4
    num_workers = 0
    device = torch.device('cuda:0')

    # --- 步骤 1：构建数据加载器 ---
    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    train_dl = data_loader.train_dataloader()
    valid_dl = data_loader.val_dataloader()

    # --- 步骤 2：初始化简单 CNN 预测器 ---
    predictor = simple_CNN(opt, embedding_dim=64).to(device)

    if os.path.exists(param_dir + opt + "_Direc_predictor_last.pth"):
        # 从上次中断处恢复
        predictor.load_state_dict(torch.load(param_dir + opt + "_Direc_predictor_last.pth"))
        loss_rec = np.loadtxt(param_dir + opt + '_Direc_predictor_recorder.txt', delimiter=',').tolist()
    else:
        loss_rec = [[0, 0.0003, 100, 100]]

    # --- 训练配置 ---
    num_epoch = 2000
    eval_step = 10
    lr = 1e-4
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    epoch = (len(loss_rec) - 1) * 10

    # --- 步骤 3：训练主循环 ---
    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []

        for idx, [load, sql_npy] in enumerate(train_dl):
            optimizer.zero_grad()

            # 前向传播：直接从载荷预测损伤场
            pre_sql = predictor(load)

            # 计算 MSE 重建损失
            loss = torch.nn.functional.mse_loss(pre_sql, sql_npy)

            loss.backward()       # 反向传播
            optimizer.step()      # 更新参数
            tmp_loss_rec.append(loss.item())

        # --- 步骤 4：每 eval_step 个 epoch 进行验证 ---
        if epoch % eval_step == 0 or epoch == num_epoch - 1:
            train_loss_mean = np.mean(np.array(tmp_loss_rec))
            tmp_loss_rec = []

            for idx, [load, sql_npy] in enumerate(valid_dl):
                pre_sql = predictor(load)
                loss = torch.nn.functional.mse_loss(pre_sql, sql_npy)
                tmp_loss_rec.append(loss.item())

            valid_loss_mean = np.mean(np.array(tmp_loss_rec))

            # 追加本次评估记录
            loss_rec.append([epoch, lr, train_loss_mean, valid_loss_mean])

            # 保存当前验证集损失最低的模型
            if valid_loss_mean < np.min(np.array(loss_rec[:-1])[:, 3]):
                torch.save(predictor.state_dict(), param_dir + opt + "_Direc_predictor_best.pth")

            print("Epoch %d, Train Loss: %.5f, Valid Loss: %.5f" %
                  (epoch, train_loss_mean, valid_loss_mean))

            # 保存最新模型权重和损失记录
            torch.save(predictor.state_dict(), param_dir + opt + "_Direc_predictor_last.pth")
            np.savetxt(param_dir + opt + '_Direc_predictor_recorder.txt', np.array(loss_rec), delimiter=',')


def show(rec, lab, load, idx=1024, save=False):
    """
    可视化重建结果与真实标签的对比图。

    参数：
      rec  (Tensor)：重建的损伤场，shape (B, C, H, W)
      lab  (Tensor)：真实损伤场标签，shape (B, C, H, W)
      load (Tensor)：载荷向量，shape (B, 3)，包含 R、E、T
      idx  (int)   ：要可视化的通道索引；1024 表示可视化所有通道
      save (bool)  ：是否将图像保存到文件
    """
    # 步骤 1：取第 2 个样本（索引 1），并转为 numpy 数组（脱离计算图，移至 CPU）
    rec_np = rec[1, :, :, :].detach().cpu().numpy()
    lab_np = lab[1, :, :, :].detach().cpu().numpy()
    load_np = load[1, ].detach().cpu().numpy()

    if idx == 1024:
        # 步骤 2a：遍历所有通道（损伤层），逐帧显示
        for i in range(rec_np.shape[0]):
            rec_frame_np = rec_np[i, :, :]
            lab_frame_np = lab_np[i, :, :]
            plt.figure(figsize=(16, 8))

            # 左图：重建结果
            plt.subplot(1, 2, 1)
            plt.imshow(rec_frame_np, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("Reconstructed of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))

            # 右图：真实标签
            plt.subplot(1, 2, 2)
            plt.imshow(lab_frame_np, vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("Label of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))

            if save:
                plt.savefig(r'results//'+str(i)+".png")
            plt.show()
    else:
        # 步骤 2b：只显示指定通道
        rec_frame_np = rec_np[idx, :, :]
        lab_frame_np = lab_np[idx, :, :]

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(rec_frame_np, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Reconstructed of R%.2f_E%.2f_T%.2f"%(load_np[0], load_np[1], load_np[2]))
        plt.subplot(1, 2, 2)
        plt.imshow(lab_frame_np, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Label of R%.2f_E%.2f_T%.2f"%(load_np[0], load_np[1], load_np[2]))
        plt.show()


if __name__ == '__main__':
    # 顺序训练所有损伤类型的 VQ-VAE + Predictor + simCNN
    train_vqvae(opt='SDEG')
    train_predictor(opt='SDEG')
    train_simCNN(opt='SDEG')
    train_vqvae(opt='MTDMG')
    train_predictor(opt='MTDMG')
    # train_simCNN(opt='MTDMG')
    train_vqvae(opt='MCDMG')
    train_predictor(opt='MCDMG')
    # train_simCNN(opt='MCDMG')
    train_vqvae(opt='FDMG')
    train_predictor(opt='FDMG')
    # train_simCNN(opt='FDMG')
    train_vqvae(opt='MDMG')
    train_predictor(opt='MDMG')
    train_simCNN(opt='MDMG')

