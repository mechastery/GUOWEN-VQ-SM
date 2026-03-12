from networks import VQVAE_Simp, Predictor, simple_CNN
from data import DmgData
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def eval_vae_predictor(opt='SDEG'):
    """
    评估 VQ-SM（VQ-VAE + Predictor）模型在验证集上的预测效果。

    流程：
      1. 加载验证集数据
      2. 加载已训练好的 VQ-VAE 编码器（用于解码）
      3. 加载已训练好的 Predictor（用于从载荷预测潜向量）
      4. 遍历验证集，对每个样本做推理并可视化结果

    参数：
      opt (str)：损伤类型，如 'SDEG'、'MTDMG'、'MCDMG'、'MDMG'
    """
    # --- 超参数与路径配置 ---
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'D:\Codes\LiminateDMG_SM\20230318\params\\'
    batch_size = 1    # 评估时每次处理一个样本
    num_workers = 0
    device = torch.device('cuda:0')

    # --- 步骤 1：构建验证集数据加载器 ---
    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    valid_dl = data_loader.val_dataloader()

    # --- 模型超参数（需与训练时一致） ---
    embedding_dim = 16
    num_embeddings = 256

    # --- 步骤 2：加载 VQ-VAE 编码器（用于解码预测潜向量） ---
    coder_model = VQVAE_Simp(opt, embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model.load_state_dict(torch.load(param_dir + opt + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model.eval()  # 关闭 BatchNorm/Dropout 训练模式

    # --- 步骤 3：加载 Predictor ---
    predictor = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    predictor.load_state_dict(torch.load(param_dir + opt + "_predictor_best.pth"))
    predictor.eval()

    # --- 步骤 4：遍历验证集，推理并可视化 ---
    for idx, [load, sql_npy] in enumerate(valid_dl):
        # 步骤 4a：Predictor 从载荷向量预测潜空间表示
        latent = predictor(load)
        # 步骤 4b：VQ-VAE 解码器将潜向量重建为损伤场
        pre_sql = coder_model.decode(latent)
        # 步骤 4c：可视化重建结果与真实标签的对比
        show(opt, pre_sql, sql_npy, load, method='VAE')


def eval_vae_predictor_TC(opt='MDEG'):
    """
    同时评估拉伸损伤（T）和压缩损伤（C）模型，并将两者融合后与真实 MDMG 标签对比。

    融合规则：在每个空间位置取拉伸损伤和压缩损伤中绝对值更大的一项作为输出。

    流程：
      1. 加载 MDMG（综合损伤）验证集数据
      2. 分别加载 MTDMG（拉伸）和 MCDMG（压缩）的 VQ-VAE + Predictor
      3. 对每个样本分别推理并融合，与真实标签对比可视化

    参数：
      opt (str)：方法标识（此参数在本函数中未直接使用，保留为接口一致性）
    """
    # --- 超参数与路径配置 ---
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'D:\Codes\LiminateDMG_SM\20230318\params\\'
    batch_size = 1
    num_workers = 0
    device = torch.device('cuda:0')

    # --- 步骤 1：加载 MDMG 综合损伤验证集 ---
    data_loader = DmgData(data_path, 'MDMG', batch_size, num_workers, device)
    valid_dl = data_loader.val_dataloader()

    # --- 模型超参数 ---
    embedding_dim = 16
    num_embeddings = 256

    # --- 步骤 2：加载拉伸损伤（MTDMG）VQ-VAE 和 Predictor ---
    coder_model = VQVAE_Simp(opt='MTDMG', embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model.load_state_dict(torch.load(param_dir + 'MTDMG' + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model.eval()

    predictor = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    predictor.load_state_dict(torch.load(param_dir + 'MTDMG' + "_predictor_best.pth"))
    predictor.eval()

    # --- 步骤 3：加载压缩损伤（MCDMG）VQ-VAE 和 Predictor ---
    coder_model_C = VQVAE_Simp(opt='MCDMG', embedding_dim=embedding_dim, num_embeddings=num_embeddings).to(device)
    coder_model_C.load_state_dict(torch.load(param_dir + 'MCDMG' + "_%d_%d_coder_best.pth"%(embedding_dim, num_embeddings)))
    coder_model_C.eval()

    predictor_C = Predictor(embedding_dim=embedding_dim, size=12).to(device)
    predictor_C.load_state_dict(torch.load(param_dir + 'MCDMG' + "_predictor_best.pth"))
    predictor_C.eval()

    # --- 步骤 4：遍历验证集，分别推理并融合 ---
    for idx, [load, sql_npy] in enumerate(valid_dl):
        # 步骤 4a：拉伸损伤模型推理
        latent_T = predictor(load)
        pre_sql_T = coder_model.decode(latent_T)

        # 步骤 4b：压缩损伤模型推理
        latent_C = predictor_C(load)
        pre_sql_C = coder_model_C.decode(latent_C)

        # 步骤 4c：可视化融合结果（show_TC 内部完成两者的取最大融合）
        show_TC(pre_sql_T, pre_sql_C, sql_npy, load, method='VAE')


def eval_cnn_predictor(opt='SDEG'):
    """
    评估简单 CNN 端到端预测器（baseline）在验证集上的表现。

    流程：
      1. 加载验证集
      2. 加载已训练的 simple_CNN 模型
      3. 推理并可视化结果

    参数：
      opt (str)：损伤类型
    """
    # --- 超参数与路径配置 ---
    data_path = r'F:\DATASET\LVI_SM\npy\dataset2'
    param_dir = r'D:\Codes\LiminateDMG_SM\20230318\params\\'
    batch_size = 1
    num_workers = 0
    device = torch.device('cuda:0')

    # --- 步骤 1：构建验证集数据加载器 ---
    data_loader = DmgData(data_path, opt, batch_size, num_workers, device)
    valid_dl = data_loader.val_dataloader()

    # --- 步骤 2：加载 simple_CNN 模型 ---
    predictor = simple_CNN(opt, embedding_dim=64).to(device)
    predictor.load_state_dict(torch.load(param_dir + opt + "_Direc_predictor_best.pth"))
    predictor.eval()

    # --- 步骤 3：推理并可视化 ---
    for idx, [load, sql_npy] in enumerate(valid_dl):
        pre_sql = predictor(load)
        show(opt, pre_sql, sql_npy, load, method='CNN')


def show(opt, rec, lab, load, save=True, method='VAE'):
    """
    可视化单一损伤类型的重建结果与真实标签对比，并保存为 PDF。

    参数：
      opt    (str)   ：损伤类型（用于文件命名）
      rec    (Tensor)：预测损伤场，shape (1, C, H, W)
      lab    (Tensor)：真实损伤场，shape (1, C, H, W)
      load   (Tensor)：载荷向量，shape (1, 3)，包含 R、E、T
      save   (bool)  ：是否保存图像到 results/ 目录
      method (str)   ：方法标识（'VAE' 或 'CNN'），用于文件命名
    """
    # 步骤 1：取批次中第一个样本（batch_size=1 时即唯一样本），转为 numpy
    rec_np = rec[0, :, :, :].detach().cpu().numpy()
    lab_np = lab[0, :, :, :].detach().cpu().numpy()
    load_np = load[0, ].detach().cpu().numpy()

    # 步骤 2：遍历所有损伤层（通道），逐帧绘制对比图
    for i in range(rec_np.shape[0]):
        rec_frame_np = rec_np[i, :, :]
        lab_frame_np = lab_np[i, :, :]

        # 左图：预测结果（jet 色彩映射，范围 [-1, 1]）
        plt.subplot(1, 2, 1)
        plt.imshow(rec_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb = plt.colorbar()
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Reconstructed of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))

        # 右图：真实标签
        plt.subplot(1, 2, 2)
        plt.imshow(lab_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb2 = plt.colorbar()
        cb2.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Label of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))

        # 步骤 3：保存图像（文件名包含方法名、损伤类型、载荷参数和层索引）
        if save:
            plt.savefig(r'results/'+method+'_'+opt+'_R%.2f_E%.2f_T%.2f_L%d.pdf'% (load_np[0], load_np[1], load_np[2], i))
        plt.show()


def show_TC(rec_T, rec_C, lab, load, save=True, method='VAE'):
    """
    可视化拉伸损伤（T）和压缩损伤（C）融合后的预测结果与真实 MDMG 标签对比。

    融合规则：
      - 在每个空间位置，比较拉伸损伤和压缩损伤的值
      - 若 T >= C，取拉伸值；否则取压缩值（取绝对值更大的一项）

    参数：
      rec_T  (Tensor)：拉伸损伤预测场，shape (1, C, H, W)
      rec_C  (Tensor)：压缩损伤预测场，shape (1, C, H, W)
      lab    (Tensor)：真实综合损伤场，shape (1, C, H, W)
      load   (Tensor)：载荷向量，shape (1, 3)
      save   (bool)  ：是否保存图像
      method (str)   ：方法标识，用于文件命名
    """
    # 步骤 1：取批次中第一个样本，转为 numpy
    recT_np = rec_T[0, :, :, :].detach().cpu().numpy()
    recC_np = rec_C[0, :, :, :].detach().cpu().numpy()

    # 步骤 2：计算融合掩码（T >= C 的位置为 True）
    minus = (recT_np - recC_np) >= 0   # T > C 的位置取 True（布尔数组）

    # 步骤 3：按掩码融合两个预测结果
    #   minus=True 的位置取 recT_np，minus=False 的位置取 recC_np
    rec_np = recT_np * minus - recC_np * (-(minus - 1))

    lab_np = lab[0, :, :, :].detach().cpu().numpy()
    load_np = load[0, ].detach().cpu().numpy()

    # 步骤 4：遍历所有损伤层，逐帧绘制融合结果与真实标签对比
    for i in range(rec_np.shape[0]):
        rec_frame_np = rec_np[i, :, :]
        lab_frame_np = lab_np[i, :, :]

        # 左图：融合预测结果
        plt.subplot(1, 2, 1)
        plt.imshow(rec_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb = plt.colorbar()
        cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Reconstructed of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))

        # 右图：真实标签
        plt.subplot(1, 2, 2)
        plt.imshow(lab_frame_np, vmin=-1, vmax=1, cmap='jet')
        cb2 = plt.colorbar()
        cb2.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.title("Label of R%.2f_E%.2f_T%.2f" % (load_np[0], load_np[1], load_np[2]))

        # 步骤 5：保存图像（文件名标注为综合损伤类型 'MDMG_d'）
        if save:
            plt.savefig(r'results/'+method+'_'+"MDMG_d"+'_R%.2f_E%.2f_T%.2f_L%d.pdf'% (load_np[0], load_np[1], load_np[2], i))
        plt.show()


if __name__ == '__main__':
    # 以下评估函数可按需取消注释运行
    # eval_vae_predictor(opt='SDEG')
    # eval_cnn_predictor(opt='SDEG')
    # eval_vae_predictor(opt='MTDMG')
    # eval_cnn_predictor(opt='MTDMG')
    # eval_vae_predictor(opt='MCDMG')
    # eval_cnn_predictor(opt='MCDMG')
    # eval_vae_predictor(opt='MDMG')
    # eval_cnn_predictor(opt='MDMG')
    eval_vae_predictor_TC(opt='MDMG')  # 默认评估拉伸+压缩融合的综合损伤预测
