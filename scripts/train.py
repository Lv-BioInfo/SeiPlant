import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
import numpy as np
from sklearn.metrics import roc_curve, auc
from datetime import datetime  # 导入 datetime 库，用于生成唯一的文件名
from models.model_architectures import sei_model,MHA_Sei
from models.model_architectures import Agro_Sei, Agro_Sei_Transformer
from utils.graph_utils import plot_auroc_auprc
import os

def train_model(model, train_loader, val_loader, device, model_dir, result_path, cross_train=False, epoch=1, lr=1e-5, type="classification"):
    """
    训练 Sei-X 模型（或其他 CNN 模型）

    Parameters
    ----------
    model : nn.Module
        传入的 PyTorch 模型（如 SeiX）
    train_loader : DataLoader
        训练数据加载器
    val_loader : DataLoader
        验证数据加载器
    tag_dict : dict
        基因组特征字典
    seq_len : int
        序列长度
    device : str
        设备（"cuda" or "cpu"）
    model_dir : str
        存储模型的路径
    result_path : str
        结果保存路径
    fold_num : int, optional
        交叉验证 fold 数, 默认值 1
    cross_train : bool, optional
        是否进行交叉训练（加载已有模型参数），默认 False
    epoch : int, optional
        训练轮数，默认值 1
    lr : float, optional
        学习率，默认值 1e-5
    """

    model = model.to(device)  # 确保模型在正确设备上

    if cross_train:
        print(f"Load model parameters from {model_dir}")
        model.load_state_dict(torch.load(model_dir))  # 仅加载参数，不重新初始化模型

    training_classifier(
        n_epoch=epoch, lr=lr, model_dir=model_dir,
        train=train_loader, valid=val_loader,
        model=model, device=device,
        result_path=result_path, type=type)

def training_classifier(n_epoch, lr, model_dir, train, valid, model, device, result_path, type="classification", early_stopping_patience=10):
    # 返回参数的个数，总的和训练的
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters total: {}, trainable: {}\n'.format(total, trainable))
    model.train()  # 将 model 的模式改为 train

    if type == 'classification':
        criterion = nn.BCELoss()  # 适用于多标签分类
    else:
        criterion = nn.MSELoss()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss, best_loss = 0, 1e5  # 初始化损失值
    no_improve_counter = 0  # 用于计数验证损失没有改善的epoch数

    # 用于存储每个 epoch 的损失
    train_losses = []
    valid_losses = []

    # 训练过程
    for epoch in range(n_epoch):
        total_loss = 0
        # Training phase
        model.train()  # 设置为训练模式
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1)  # For sei model
            outputs = model(inputs)
            outputs = outputs.squeeze()  # 去掉外层的 dimension
            loss = criterion(outputs, labels)  # 计算训练损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(epoch + 1, i + 1, t_batch, loss.item()), end='\r')

        train_loss = total_loss / t_batch
        train_losses.append(train_loss)  # 记录训练损失
        print('\nTrain | Loss:{:.5f}'.format(train_loss))

        # Validation phase
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            total_loss = 0
            all_labels = []
            all_probs = []

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)  # For sei model
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # 存储真实标签和预测概率
                all_labels.append(labels.cpu().numpy().ravel() )
                all_probs.append(outputs.cpu().numpy().ravel() )  # **转换为概率**


            valid_loss = total_loss / v_batch
            valid_losses.append(valid_loss)  # 记录验证损失
            print("Valid | Loss:{:.5f} ".format(valid_loss))
            if type == "classification":
                # **计算 AUC**
                all_labels = np.concatenate(all_labels)
                all_probs = np.concatenate(all_probs)

                auroc, auprc = plot_auroc_auprc(all_labels, all_probs)
                print(f"Valid AUROC: {auroc:.4f}")
                print(f"Valid AUPRC: {auprc:.4f}")

            # Early stopping check
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), model_dir)  # 保存模型参数
                print('Saving model with loss {:.3f}'.format(valid_loss))
                no_improve_counter = 0  # Reset counter if validation loss improves
            else:
                no_improve_counter += 1  # Increment counter if no improvement

            # Check if early stopping is triggered
            if no_improve_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation loss")
                break  # Stop training

        print('-----------------------------------------------')

    epochs_trained = len(train_losses)

    # **绘制损失曲线**
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs_trained + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs_trained + 1), valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存损失曲线
    loss_filename = os.path.join(result_path, f'Training_validation_loss_plot.png')
    plt.savefig(loss_filename, dpi=300)  # 高质量保存
    print(f"Loss plot saved as {loss_filename}")

    plt.close('all')

def training_classifier_classification(n_epoch, lr, model_dir, train, valid, model, device, result_path, fold_num=1, early_stopping_patience=20):
    # 返回参数的个数，总的和训练的
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters total: {}, trainable: {}\n'.format(total, trainable))
    model.train()  # 将 model 的模式改为 train

    criterion = nn.BCELoss()  # 适用于多标签分类
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss, best_loss = 0, 1e5  # 初始化损失值
    no_improve_counter = 0  # 用于计数验证损失没有改善的epoch数

    # 用于存储每个 epoch 的损失
    train_losses = []
    valid_losses = []

    # 训练过程
    for epoch in range(n_epoch):
        total_loss = 0
        # Training phase
        model.train()  # 设置为训练模式
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1)  # For sei model
            outputs = model(inputs)
            outputs = outputs.squeeze()  # 去掉外层的 dimension
            loss = criterion(outputs, labels)  # 计算训练损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(epoch + 1, i + 1, t_batch, loss.item()), end='\r')

        train_loss = total_loss / t_batch
        train_losses.append(train_loss)  # 记录训练损失
        print('\nTrain | Loss:{:.5f}'.format(train_loss))

        # Validation phase
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            total_loss = 0
            all_labels = []
            all_probs = []

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)  # For sei model
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # 存储真实标签和预测概率
                all_labels.append(labels.cpu().numpy().ravel() )
                all_probs.append(outputs.cpu().numpy().ravel() )  # **转换为概率**


            valid_loss = total_loss / v_batch
            valid_losses.append(valid_loss)  # 记录验证损失
            print("Valid | Loss:{:.5f} ".format(valid_loss))

            # **计算 AUC**
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)

            auroc, auprc = plot_auroc_auprc(all_labels, all_probs)
            print(f"Valid AUROC: {auroc:.4f}")
            print(f"Valid AUPRC: {auprc:.4f}")

            # Early stopping check
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), model_dir)  # 保存模型参数
                print('Saving model with loss {:.3f}'.format(valid_loss))
                no_improve_counter = 0  # Reset counter if validation loss improves
            else:
                no_improve_counter += 1  # Increment counter if no improvement

            # Check if early stopping is triggered
            if no_improve_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation loss")
                break  # Stop training

        print('-----------------------------------------------')

    epochs_trained = len(train_losses)

    # **绘制损失曲线**
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs_trained + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs_trained + 1), valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存损失曲线
    loss_filename = os.path.join(result_path, f'fold_num_{fold_num}_training_validation_loss_plot.png')
    plt.savefig(loss_filename, dpi=300)  # 高质量保存
    print(f"Loss plot saved as {loss_filename}")

    plt.close('all')

def training_classifier_regression(n_epoch, lr, model_dir, train, valid, model, device, result_path, fold_num, early_stopping_patience=20):
    # 返回参数的个数，总的和训练的
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters total: {}, trainable: {}\n'.format(total, trainable))
    model.train()  # 将 model 的模式改为 train

    criterion = nn.MSELoss()  # 适用于多标签分类
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss, best_loss = 0, 1e5  # 初始化损失值
    no_improve_counter = 0  # 用于计数验证损失没有改善的epoch数

    # 用于存储每个 epoch 的损失
    train_losses = []
    valid_losses = []

    # 训练过程
    for epoch in range(n_epoch):
        total_loss = 0
        # Training phase
        model.train()  # 设置为训练模式
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1)  # For sei model
            outputs = model(inputs)
            outputs = outputs.squeeze()  # 去掉外层的 dimension
            loss = criterion(outputs, labels)  # 计算训练损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(epoch + 1, i + 1, t_batch, loss.item()), end='\r')

        train_loss = total_loss / t_batch
        train_losses.append(train_loss)  # 记录训练损失
        print('\nTrain | Loss:{:.5f}'.format(train_loss))

        # Validation phase
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            total_loss = 0
            all_labels = []
            all_probs = []

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)  # For sei model
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # 存储真实标签和预测概率
                all_labels.append(labels.cpu().numpy().ravel() )
                all_probs.append(outputs.cpu().numpy().ravel() )  # **转换为概率**


            valid_loss = total_loss / v_batch
            valid_losses.append(valid_loss)  # 记录验证损失
            print("Valid | Loss:{:.5f} ".format(valid_loss))

            # # **计算 AUC**
            # all_labels = np.concatenate(all_labels)
            # all_probs = np.concatenate(all_probs)

            # auroc, auprc = plot_auroc_auprc(all_labels, all_probs)
            # print(f"Valid AUROC: {auroc:.4f}")
            # print(f"Valid AUPRC: {auprc:.4f}")

            # Early stopping check
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), model_dir)  # 保存模型参数
                print('Saving model with loss {:.3f}'.format(valid_loss))
                no_improve_counter = 0  # Reset counter if validation loss improves
            else:
                no_improve_counter += 1  # Increment counter if no improvement

            # Check if early stopping is triggered
            if no_improve_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation loss")
                break  # Stop training

        print('-----------------------------------------------')

    # epochs_trained = len(train_losses)

    # # **绘制损失曲线**
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, epochs_trained + 1), train_losses, label='Training Loss', color='blue')
    # plt.plot(range(1, epochs_trained + 1), valid_losses, label='Validation Loss', color='orange')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.grid(True)

    # # 保存损失曲线
    # loss_filename = os.path.join(result_path, f'fold_num_{fold_num}_training_validation_loss_plot.png')
    # plt.savefig(loss_filename, dpi=300)  # 高质量保存
    # print(f"Loss plot saved as {loss_filename}")

    # plt.close('all')

def training_classifier_old(batch_size, n_epoch, lr, model_dir, train, valid, model, device, early_stopping_patience=5):
    """
    记录老的训练函数
    """
    # 返回参数的个数，总的和训练的
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters total: {}, trainable: {}\n'.format(total, trainable))
    model.train()  # 将 model 的模式改为 train
    criterion = nn.MSELoss() # 因为是回归任务，改为MSE loss损失函数
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    total_loss, best_loss = 0, 1e5  # 初始化损失值
    no_improve_counter = 0  # 用于计数验证损失没有改善的epoch数

    # 用于存储每个 epoch 的损失
    train_losses = []
    valid_losses = []

    # 训练过程
    for epoch in range(n_epoch):
        total_loss = 0
        # Training phase
        model.train()  # 设置为训练模式
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1)  # For sei model
            outputs = model(inputs)
            outputs = outputs.squeeze()  # 去掉外层的 dimension
            loss = criterion(outputs, labels)  # 计算训练损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(epoch + 1, i + 1, t_batch, loss.item()), end='\r')

        train_loss = total_loss / t_batch
        train_losses.append(train_loss)  # 记录训练损失
        print('\nTrain | Loss:{:.5f}'.format(train_loss))

        # Validation phase
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            total_loss = 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)  # For sei model
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            valid_loss = total_loss / v_batch
            valid_losses.append(valid_loss)  # 记录验证损失
            print("Valid | Loss:{:.5f} ".format(valid_loss))

            # Early stopping check
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), model_dir)  # 保存模型参数
                print('Saving model with loss {:.3f}'.format(valid_loss))
                no_improve_counter = 0  # Reset counter if validation loss improves
            else:
                no_improve_counter += 1  # Increment counter if no improvement

            # Check if early stopping is triggered
            if no_improve_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation loss")
                break  # Stop training

        print('-----------------------------------------------')

    # 绘制训练损失和验证损失的图
    plt.figure(figsize=(10, 10))
    epochs_trained = epoch + 1  # 记录实际训练的 epoch 数量

    # 绘制损失曲线
    plt.plot(range(1, epochs_trained + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs_trained + 1), valid_losses, label='Validation Loss')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 获取当前时间并格式化为字符串
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 根据当前时间生成唯一文件名
    filename = f'training_validation_loss_plot_{timestamp}.png'
    # 保存图表为图片文件
    plt.savefig(filename)  # 可以指定路径和文件名
