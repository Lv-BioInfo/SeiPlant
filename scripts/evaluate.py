from models.model_architectures import sei_model,Agro_Sei_Transformer,Agro_Sei
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from datetime import datetime
from utils.graph_utils import plot_auroc_auprc
import os

def evaluate_model(model, test_loader, device, prediction_dir, type="classification"):
    """
        评估 Sei-X 模型（或其他 CNN 模型）

        Parameters
        ----------
        model : nn.Module
            传入的 PyTorch 模型（如 SeiX）
        test_loader : DataLoader
            测试数据加载器
        device : str
            设备（"cuda" or "cpu"）
        prediction_dir : str
            预测结果保存路径
        fold_num : int, optional
            交叉验证 fold 数, 默认值 1
        """

    model.to(device)  # 确保模型在正确设备上
    model.eval()  # 设置为评估模式

    eval_model(test_loader=test_loader, device=device, prediction_dir=prediction_dir, model=model, type=type)

def eval_model(test_loader, device, prediction_dir, model, type=type):
    model.eval()  # 进入评估模式
    if type == "classification":

        criterion = nn.BCELoss()  # **多标签二分类损失**
    else:
         criterion = nn.MSELoss()  # **多标签回归损失**

    total_labels = torch.tensor([], dtype=torch.float)
    total_outputs = torch.tensor([], dtype=torch.float)

    total_loss = 0
    num_samples = 0

    # 测试模型
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float).permute(0, 2, 1)
            labels = labels.to(device, dtype=torch.float)  # BCE loss 需要 float 类型

            outputs = model(inputs).squeeze()  # 模型预测值 (logits)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
            if outputs.dim() == 1:  # 说明丢失了 batch 维度
                outputs = outputs.unsqueeze(0)  # 变成 [1, 7]
            # 存储真实标签 & 预测值
            total_outputs = torch.cat((total_outputs, outputs.cpu()))  # logits (未经过 Sigmoid)
            total_labels = torch.cat((total_labels, labels.cpu()))


    # 计算平均损失
    avg_loss = total_loss / num_samples
    print(f"Test Loss: {avg_loss:.4f}")

    # 转换数据格式
    total_labels = total_labels.numpy().ravel()  # **展开标签**
    total_outputs = torch.sigmoid(total_outputs).numpy().ravel()  # **展开预测概率**

    # # **二值化 (threshold=0.5)**
    # total_outputs = (total_outputs > 0.5).astype(int)  # **0.5 以上为 1，否则为 0**

    # **保存预测结果**
    test_prediction = {
        'predictions': total_outputs,  # **存储二值化后的预测结果**
        'targets': total_labels
    }
    np.save(prediction_dir, test_prediction)
    
    if type == "classification":
        curve_filename = os.path.join(prediction_dir.rsplit('\\', 1)[0], f"auroc_auprc_test.png")
        auroc, auprc = plot_auroc_auprc(total_labels, total_outputs, save_path=curve_filename, picture=True)

        return avg_loss, auroc, auprc  # **返回测试损失, AUROC & AUPRC**
    else:
        return avg_loss, None, None

def eval_model_classification(test_loader, device, prediction_dir, model):

    model.eval()  # 进入评估模式
    criterion = nn.BCELoss()  # **多标签二分类损失**

    total_labels = torch.tensor([], dtype=torch.float)
    total_outputs = torch.tensor([], dtype=torch.float)

    total_loss = 0
    num_samples = 0

    # 测试模型
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float).permute(0, 2, 1)
            labels = labels.to(device, dtype=torch.float)  # BCE loss 需要 float 类型

            outputs = model(inputs).squeeze()  # 模型预测值 (logits)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

            # 存储真实标签 & 预测值
            total_outputs = torch.cat((total_outputs, outputs.cpu()))  # logits (未经过 Sigmoid)
            total_labels = torch.cat((total_labels, labels.cpu()))

    # 计算平均损失
    avg_loss = total_loss / num_samples
    print(f"Test BCE Loss: {avg_loss:.4f}")

    # 转换数据格式
    total_labels = total_labels.numpy().ravel()  # **展开标签**
    total_outputs = torch.sigmoid(total_outputs).numpy().ravel()  # **展开预测概率**

    # # **二值化 (threshold=0.5)**
    # total_outputs = (total_outputs > 0.5).astype(int)  # **0.5 以上为 1，否则为 0**

    # # **保存预测结果**
    # test_prediction = {
    #     'predictions': total_outputs,  # **存储二值化后的预测结果**
    #     'targets': total_labels
    # }
    # np.save(prediction_dir, test_prediction)

    curve_filename = os.path.join(prediction_dir.rsplit('\\', 1)[0], f"auroc_auprc_test.png")
    auroc, auprc = plot_auroc_auprc(total_labels, total_outputs, save_path=curve_filename, picture=True)

    return avg_loss, auroc, auprc  # **返回测试损失, AUROC & AUPRC**

def eval_model_regression(test_loader, device, prediction_dir, model, fold_num):

    model.eval()  # 进入评估模式
    criterion = nn.MSELoss()  # **多标签二分类损失**

    total_labels = torch.tensor([], dtype=torch.float)
    total_outputs = torch.tensor([], dtype=torch.float)

    total_loss = 0
    num_samples = 0

    # 测试模型
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float).permute(0, 2, 1)
            labels = labels.to(device, dtype=torch.float)  # BCE loss 需要 float 类型

            outputs = model(inputs).squeeze()  # 模型预测值 (logits)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

            # 存储真实标签 & 预测值
            total_outputs = torch.cat((total_outputs, outputs.cpu()))  # logits (未经过 Sigmoid)
            total_labels = torch.cat((total_labels, labels.cpu()))

    # 计算平均损失
    avg_loss = total_loss / num_samples
    print(f"Test MSE Loss: {avg_loss:.4f}")

    # # 转换数据格式
    # total_labels = total_labels.numpy().ravel()  # **展开标签**
    # total_outputs = torch.sigmoid(total_outputs).numpy().ravel()  # **展开预测概率**

    # # **二值化 (threshold=0.5)**
    # total_outputs = (total_outputs > 0.5).astype(int)  # **0.5 以上为 1，否则为 0**

    # # **保存预测结果**
    # test_prediction = {
    #     'predictions': total_outputs,  # **存储二值化后的预测结果**
    #     'targets': total_labels
    # }
    # np.save(prediction_dir, test_prediction)

    # curve_filename = os.path.join(prediction_dir.rsplit('/', 1)[0], f"fold_num_{fold_num}_auroc_auprc_test.png")
    # auroc, auprc = plot_auroc_auprc(total_labels, total_outputs, save_path=curve_filename, picture=True)

    return avg_loss, None, None  # **返回测试损失, AUROC & AUPRC**

def plot_predictions_vs_targets(predictions, targets):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets.numpy(), predictions.numpy(), alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 理想情况下的对角线
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predictions vs True Values")
    plt.show()

def plot_error_distribution(predictions, targets):
    errors = (predictions - targets).numpy()
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.show()


def evaluate_predictions(prediction_file):
    """
    从 .npy 文件中加载 predictions 和 targets，并计算差异。
    :param prediction_file: 保存 predictions 和 targets 的 .npy 文件路径
    """
    # 加载 .npy 文件
    data = np.load(prediction_file, allow_pickle=True).item()
    predictions = data['predictions']
    targets = data['targets']

    # 将 numpy 数组转换为 PyTorch 张量
    predictions = torch.tensor(predictions, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # 计算 MSE
    mse = torch.mean((predictions - targets) ** 2).item()
    print(f"MSE: {mse:.4f}")

    # 计算 MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()
    print(f"MAE: {mae:.4f}")

    # 计算自定义准确率（阈值设为 0.1）
    threshold = 0.1
    errors = torch.abs(predictions - targets)
    correct_predictions = (errors <= threshold).float()
    accuracy = torch.mean(correct_predictions).item()
    print(f"Accuracy (threshold={threshold}): {accuracy:.4f}")

    # 计算 R²
    mean_targets = torch.mean(targets)
    ss_total = torch.sum((targets - mean_targets) ** 2)
    ss_residual = torch.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    print(f"R²: {r2:.4f}")

    # 计算 RMSE
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    print(f"RMSE: {rmse:.4f}")

    plot_top_20_percent_errors(predictions, targets)

def plot_correct_predictions(predictions, targets, threshold=0.1):
    """
    仅显示预测正确的部分的值。
    :param predictions: 预测值（PyTorch 张量或 NumPy 数组）
    :param targets: 真实值（PyTorch 张量或 NumPy 数组）
    :param threshold: 判定正确的阈值
    """
    # 将输入转换为 NumPy 数组（如果输入是 PyTorch 张量）
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    # 计算预测值与真实值之间的绝对差异
    errors = np.abs(predictions - targets)

    # 筛选出预测正确的样本
    correct_indices = errors <= threshold
    correct_predictions = predictions[correct_indices]
    correct_targets = targets[correct_indices]

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(correct_targets, correct_predictions, alpha=0.5, label='Correct Predictions')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal Line')  # 理想情况下的对角线
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"Correct Predictions (Threshold={threshold})")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_top_20_percent_errors(predictions, targets):
    """
    绘制差异最大的前 20% 的样本的散点图。
    :param predictions: 预测值（PyTorch 张量或 NumPy 数组）
    :param targets: 真实值（PyTorch 张量或 NumPy 数组）
    """
    # 将输入转换为 NumPy 数组（如果输入是 PyTorch 张量）
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    # 计算预测值与真实值之间的绝对差异
    errors = np.abs(predictions - targets)

    # 对差异进行排序，并筛选出差异最大的前 20% 的样本
    num_samples = len(errors)
    num_top_errors = int(0.2 * num_samples)  # 前 20% 的样本数量
    sorted_indices = np.argsort(errors)[-num_top_errors:]  # 差异最大的前 20% 的索引
    top_errors = errors[sorted_indices]
    top_predictions = predictions[sorted_indices]
    top_targets = targets[sorted_indices]

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(top_targets, top_predictions, alpha=0.5, color='red', label='Top 20% Errors')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Ideal Line')  # 理想情况下的对角线
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Top 20% Errors: Predictions vs True Values")
    plt.legend()
    plt.grid(True)
    plt.show()