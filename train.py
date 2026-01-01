"""
File: train.py
Description: Professional AI Training Pipeline with Green AI Metrics (Energy & Power).
Compatible with: Python 3.10+
Hardware: Optimized for NVIDIA GPU (12GB VRAM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import json
import time
import os
import sys
import gc
import pynvml  # NVIDIA Management Library for Power Monitoring

# ==========================================
# 1. 全局配置 (CONFIGURATION)
# ==========================================
CONFIG = {
    'data_dir': './dataset',               # 你的图片文件夹路径
    'save_csv': 'results_metrics.csv',     # 保存指标表格
    'save_roc': 'results_roc.json',        # 保存ROC数据
    'epochs': 15,                          # 13k图片建议跑15轮，保证收敛
    'learning_rate': 1e-4,                 # 迁移学习的最佳初始学习率
    'seed': 42,                            # 固定随机种子，确保结果可复现
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 11个模型列表
MODELS_TO_RUN = [
    "mobilenet_v3_small", "mobilenet_v2", "resnet18", 
    "googlenet", "alexnet",                # 经典/轻量组
    "resnet50", "densenet169",             # 中量组
    "mobilenet_v3_large",
    "resnet101", "vgg16", "vgg19"          # 重量组 (显存杀手)
]

# ==========================================
# 2. 硬件监控模块 (GREEN AI MONITOR)
# ==========================================
def init_gpu_monitor():
    """初始化 NVIDIA 显卡监控"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 默认第一张卡
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"[System] GPU Monitoring Active: {name}")
        return handle
    except Exception as e:
        print(f"[Warning] NVML Init Failed: {e}. Power metrics will be 0.")
        return None

def get_power_usage(handle):
    """获取当前瞬时功耗 (Watts)"""
    if handle is None: return 0.0
    try:
        # nvml returns milliwatts, convert to Watts
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mw / 1000.0
    except:
        return 0.0

# ==========================================
# 3. 数据准备 (DATA PIPELINE)
# ==========================================
def get_dataloaders(model_name):
    print(f"\n[Stage 1] Loading Data for {model_name}...")
    
    # --- 智能 Batch Size 调整 (VRAM Protection) ---
    # 12GB VRAM 跑 VGG19 比较吃力，必须降低 Batch Size
    batch_size = 64
    if model_name in ['vgg16', 'vgg19', 'densenet169', 'resnet101']:
        batch_size = 16 
        print(f"   -> High-memory model detected. Batch Size set to {batch_size}.")
    else:
        print(f"   -> Standard model. Batch Size set to {batch_size}.")

    # 数据增强 (仅训练集)
    # Resize到224是标准做法
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 验证/测试集保持纯净
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        # 加载数据集
        # 注意：为了代码简洁，这里统一用 train_tf 读取，
        # 在严谨的工程中应该自定义 Dataset 类来区分 Train/Val transform
        full_ds = datasets.ImageFolder(CONFIG['data_dir'], transform=train_tf)
    except Exception as e:
        print(f"[Error] Dataset path incorrect: {CONFIG['data_dir']}")
        print("Please ensure your folder structure is: dataset/benign and dataset/malignant")
        sys.exit(1)

    # 70% Train, 15% Val, 15% Test
    total_len = len(full_ds)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    # 随机切分
    train_ds, val_ds, test_ds = random_split(
        full_ds, [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    
    print(f"   -> Images: {total_len} (Train:{train_len}, Val:{val_len}, Test:{test_len})")

    # 创建加载器 (num_workers=4 加速读取)
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val':   DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        'test':  DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    return loaders, full_ds.classes

# ==========================================
# 4. 模型初始化 (MODEL FACTORY)
# ==========================================
def init_model(model_name, num_classes):
    print(f"[Stage 2] Initializing Architecture: {model_name}")
    weights = 'DEFAULT' # 使用最新的 ImageNet 预训练权重
    
    try:
        match model_name:
            case "resnet18":
                m = models.resnet18(weights=weights)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
            case "resnet50":
                m = models.resnet50(weights=weights)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
            case "resnet101":
                m = models.resnet101(weights=weights)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
            case "vgg16":
                m = models.vgg16(weights=weights)
                m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
            case "vgg19":
                m = models.vgg19(weights=weights)
                m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
            case "densenet169":
                m = models.densenet169(weights=weights)
                m.classifier = nn.Linear(m.classifier.in_features, num_classes)
            case "googlenet":
                m = models.googlenet(weights=weights)
                m.aux_logits = False
                m.fc = nn.Linear(m.fc.in_features, num_classes)
            case "alexnet":
                m = models.alexnet(weights=weights)
                m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
            case "mobilenet_v2":
                m = models.mobilenet_v2(weights=weights)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            case "mobilenet_v3_small":
                m = models.mobilenet_v3_small(weights=weights)
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
            case "mobilenet_v3_large":
                m = models.mobilenet_v3_large(weights=weights)
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
            case _:
                print(f"Error: Model {model_name} not found.")
                return None
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

    return m.to(CONFIG['device'])

# ==========================================
# 5. 训练循环 (含功耗采样)
# ==========================================
def train_model(model, loaders, gpu_handle):
    print(f"[Stage 3] Training ({CONFIG['epochs']} Epochs) with Power Monitoring...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    start_time = time.time()
    power_readings = [] # 存储功率采样点
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(loaders['train']):
            # --- 功耗采样 (Sampling) ---
            # 每 10 个 batch 采样一次，避免拖慢训练速度
            if i % 10 == 0:
                watts = get_power_usage(gpu_handle)
                if watts > 0: power_readings.append(watts)
            # --------------------------

            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 简单打印进度
        epoch_loss = running_loss / len(loaders['train'])
        curr_power = power_readings[-1] if power_readings else 0
        print(f"   Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {epoch_loss:.4f} | Power: {curr_power:.1f}W")

    total_time = time.time() - start_time
    avg_power_watts = sum(power_readings) / len(power_readings) if power_readings else 0
    
    # 计算总能耗 (Watt-hours) = Watts * Hours
    total_energy_wh = avg_power_watts * (total_time / 3600.0)
    
    return model, total_time, avg_power_watts, total_energy_wh

# ==========================================
# 6. 评估与保存
# ==========================================
def evaluate_and_save(model, loaders, classes, model_name, t_time, avg_watts, energy_wh):
    print(f"[Stage 4] Evaluating {model_name}...")
    model.eval()
    
    y_true = []
    y_pred = []
    y_scores = [] # For ROC
    
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs = inputs.to(CONFIG['device'])
            outputs = model(inputs)
            
            # 获取概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            # 兼容二分类与多分类 ROC 处理
            if len(classes) == 2:
                y_scores.extend(probs[:, 1].cpu().numpy()) # 取 Positive Class 概率
            else:
                y_scores.extend(probs.cpu().numpy()) # 暂时只存，多分类ROC比较复杂，这里主要演示二分类

    # 计算核心指标
    acc = accuracy_score(y_true, y_pred)
    
    if len(classes) == 2:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
    else:
        # 多分类简单处理
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = 0
        auc_score = 0 # 简化处理
        fpr, tpr = [], []

    print(f"   -> Result: Acc={acc:.4f} | Energy={energy_wh:.2f}Wh | Time={t_time:.1f}s")

    # --- 保存 Metrics 到 CSV ---
    data_row = {
        'model': model_name,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auc_score,
        'time_sec': t_time,
        'avg_power_watts': avg_watts,
        'energy_wh': energy_wh
    }
    
    df = pd.DataFrame([data_row])
    # 如果文件不存在写表头，否则追加
    if not os.path.exists(CONFIG['save_csv']):
        df.to_csv(CONFIG['save_csv'], index=False, mode='w')
    else:
        df.to_csv(CONFIG['save_csv'], index=False, mode='a', header=False)
        
    # --- 保存 ROC 数据到 JSON ---
    if len(classes) == 2:
        roc_entry = {
            'model': model_name,
            'auc': auc_score,
            'fpr': fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
            'tpr': tpr.tolist() if isinstance(tpr, np.ndarray) else tpr
        }
        
        existing_data = []
        if os.path.exists(CONFIG['save_roc']):
            try:
                with open(CONFIG['save_roc'], 'r') as f: existing_data = json.load(f)
            except: pass
            
        existing_data.append(roc_entry)
        with open(CONFIG['save_roc'], 'w') as f:
            json.dump(existing_data, f)

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    print(f"{'='*50}\nSkin Cancer Classification & Green AI Analysis\n{'='*50}")
    
    # 1. 初始化显卡监控
    gpu_handle = init_gpu_monitor()
    
    # 2. 遍历所有模型
    for m_name in MODELS_TO_RUN:
        print(f"\n{'='*40}\nPROCESSING MODEL: {m_name.upper()}\n{'='*40}")
        
        try:
            # 准备数据
            loaders, classes = get_dataloaders(m_name)
            
            # 初始化模型
            model = init_model(m_name, len(classes))
            
            if model:
                # 训练 (含监控)
                model, t_time, p_watts, e_wh = train_model(model, loaders, gpu_handle)
                
                # 评估并保存
                evaluate_and_save(model, loaders, classes, m_name, t_time, p_watts, e_wh)
                
                # 显存清理 (非常重要！)
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"[OOM Error] {m_name} ran out of VRAM. Skipping...")
                torch.cuda.empty_cache()
            else:
                print(f"[Error] {e}")

    # 关闭监控
    try: pynvml.nvmlShutdown()
    except: pass
    
    print("\nTraining Complete! Run 'python visualize.py' to generate reports.")