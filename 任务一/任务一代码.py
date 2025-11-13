# -*- coding: utf-8 -*-
import os
import sys
import glob
import time
import copy
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- Matplotlib 样式 -----------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
research_colors = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8',
                   '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']

# ----------------------- 全局配置（本地权重优先） -----------------------
CFG = {
    "data_root": r"F:\本科组数据",
    "image_dir": "附件1",
    "weather_dir": "附件2",
    "predict_dir": "附件3",
    "outputs": "./石榴病害分析结果",

    "seed": 42,
    "num_classes": 3,
    "img_size": 224,
    "batch_size_ssl": 16,
    "batch_size": 8,
    "epochs_ssl": 50,
    "epochs_ft": 80,
    "lr_ssl": 5e-4,
    "lr_ft": 1e-4,
    "weight_decay": 5e-2,

    "weather_pref_cols": ["temp", "dewpt", "rh", "precip_rate", "solar_rad",
                          "ghi", "dhi", "dni", "pres", "wind_spd", "vis"],
    "fusion": "early_concat",     # early_concat | early_film

    # 预训练/缓存控制（强制离线）
    "pretrained_dir": "./pretrained_cache",
    "use_mirror": False,
    "hf_endpoint": "https://hf-mirror.com",
    "force_offline": True,        # 离线环境建议 True

    # 本地权重（请根据你机器上的路径确认存在）
    "local_weights": {
        "resnet50": r"E:\PythonProject\resnet50.pth",
        "vit_base_patch16_224": r"E:\PythonProject\vit_base_patch16_224.pth",
        "convnextv2_base": r"E:\PythonProject\convnextv2_base.pth",
        "swin_base_patch4_window7_224": r"E:\PythonProject\swin_base_patch4_window7_224.pth",
    }
}

PHASE_MAP = {"健康期": 0, "初发期": 1, "发病期": 2}
IDX2LABEL = {v: k for k, v in PHASE_MAP.items()}

# ----------------------- 实用函数 -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def device_of():
    return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

# ----------------------- timm + 缓存/离线设置 -----------------------
def setup_hf_mirror_and_cache():
    cache_dir = os.path.abspath(CFG.get("pretrained_dir", "./pretrained_cache"))
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TORCH_HOME"] = cache_dir
    os.environ["TIMM_DOWNLOAD_CACHE"] = cache_dir
    # 强制离线：不会尝试联网
    os.environ["HF_HUB_OFFLINE"] = "1"
    # 若你有可用镜像可改为 True，并关闭 force_offline
    if CFG.get("use_mirror", False) and not CFG.get("force_offline", True):
        os.environ["HF_ENDPOINT"] = CFG.get("hf_endpoint", "https://hf-mirror.com")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    import timm
    setup_hf_mirror_and_cache()
except Exception as e:
    timm = None
    print("⚠️ 需要安装 timm：pip install timm", file=sys.stderr)

# 宽松加载兜底（strict=False）
def load_local_weights_relaxed(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    # 兼容 Lightning/自定义包装
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"宽松加载：missing={len(missing)}, unexpected={len(unexpected)}")

# 安全创建 timm 模型：固定不联网，若有本地权重则宽松加载
def create_timm_backbone(model_name: str, num_classes: int = 0,
                         prefer_pretrained: bool = False, local_ckpt: Optional[str] = None):
    assert timm is not None, "timm 未安装"
    # 始终本地构建，pretrained=False（避免任何联网尝试）
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    # 若存在本地权重，宽松加载
    if local_ckpt and os.path.exists(local_ckpt):
        try:
            print(f"使用本地权重: {local_ckpt}")
            load_local_weights_relaxed(model, local_ckpt)
        except Exception as ee:
            print(f"本地权重加载失败（将保持随机初始化）: {ee}")
    else:
        print(f"未找到本地权重：{local_ckpt}，将随机初始化 {model_name}")
    return model

# ----------------------- 可视化 -----------------------
def generate_chinese_visualization(true_labels, predicted_labels, feature_vectors, output_path):
    ensure_dir(output_path)
    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[IDX2LABEL[i] for i in range(3)],
                yticklabels=[IDX2LABEL[i] for i in range(3)])
    plt.title('石榴病害阶段分类混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签'); plt.ylabel('真实标签')
    plt.tight_layout(); plt.savefig(f'{output_path}/混淆矩阵.png', dpi=300, bbox_inches='tight'); plt.close()
    # 指标
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    perf = pd.DataFrame({'评估指标':['准确率','加权F1分数','Kappa系数'],
                         '数值':[f'{acc:.4f}', f'{f1:.4f}', f'{kappa:.4f}'],
                         '说明':['正确分类比例','考虑类别不平衡','一致性度量']})
    # t-SNE
    if isinstance(feature_vectors, np.ndarray) and feature_vectors.shape[0] > 2:
        try:
            n = feature_vectors.shape[0]
            perplex = min(30, max(5, n//3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplex, init="random", learning_rate="auto")
            f2d = tsne.fit_transform(feature_vectors)
            plt.figure(figsize=(12, 10))
            y = np.array(true_labels)
            for i in range(3):
                m = (y == i)
                if m.sum() == 0: continue
                plt.scatter(f2d[m,0], f2d[m,1], color=research_colors[i], alpha=0.7, s=40,
                            label=IDX2LABEL[i], edgecolors='white', linewidth=0.5)
            plt.title('特征空间分布 (t-SNE)'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f'{output_path}/特征空间分布.png', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"t-SNE 失败: {e}")
    return perf

# ----------------------- 自监督模块（稳健 SimCLR / BYOL） -----------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)

class NTXentLoss(nn.Module):
    """稳健版 SimCLR 对比损失（不移除对角，直接mask；正样本列索引不越界）"""
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    @staticmethod
    def _l2_normalize(x, dim=1, eps=1e-8):
        return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

    def forward(self, z1, z2):
        assert z1.dim() == 2 and z2.dim() == 2, "z1/z2 应为 [B, D]"
        B = z1.size(0)
        assert B >= 2, f"SimCLR 批次过小：{B}，请将 batch_size_ssl >= 2 且 drop_last=True"

        z1 = self._l2_normalize(z1); z2 = self._l2_normalize(z2)
        z = torch.cat([z1, z2], dim=0)                 # [2B, D]
        sim = torch.matmul(z, z.t()) / self.tau        # [2B, 2B]

        # mask 自身相似度
        diag = torch.eye(2*B, device=z.device, dtype=sim.dtype)
        sim = sim - 1e9 * diag

        # 正样本列索引：i <-> i+B
        indices = torch.arange(B, device=z.device)
        pos = torch.cat([indices + B, indices], dim=0)  # [2B]

        # 行内 log-softmax，再取正样本列
        log_prob = torch.log_softmax(sim, dim=1)        # [2B, 2B]
        loss = -log_prob[torch.arange(2*B, device=z.device), pos]
        return loss.mean()

class BYOLLoss(nn.Module):
    def forward(self, p, z):
        p = F.normalize(p, dim=1); z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

class SelfSupervisedModel(nn.Module):
    def __init__(self, backbone="resnet50", projection_dim=128):
        super().__init__()
        if backbone == "convnext":
            self.online_encoder = create_timm_backbone(
                "convnextv2_base", num_classes=0, prefer_pretrained=False,
                local_ckpt=CFG["local_weights"].get("convnextv2_base",""))
            feat_dim = self.online_encoder.num_features
        elif backbone == "swin":
            self.online_encoder = create_timm_backbone(
                "swin_base_patch4_window7_224", num_classes=0, prefer_pretrained=False,
                local_ckpt=CFG["local_weights"].get("swin_base_patch4_window7_224",""))
            feat_dim = self.online_encoder.num_features
        else:
            self.online_encoder = create_timm_backbone(
                "resnet50", num_classes=0, prefer_pretrained=False,
                local_ckpt=CFG["local_weights"].get("resnet50",""))
            feat_dim = self.online_encoder.num_features

        self.online_proj = ProjectionHead(feat_dim, projection_dim)
        self.predict_head = ProjectionHead(projection_dim, projection_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_proj   = copy.deepcopy(self.online_proj)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        for p in self.target_proj.parameters(): p.requires_grad = False

    def update_target(self, tau=0.99):
        with torch.no_grad():
            for op, tp in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                tp.data = tau * tp.data + (1 - tau) * op.data
            for op, tp in zip(self.online_proj.parameters(), self.target_proj.parameters()):
                tp.data = tau * tp.data + (1 - tau) * op.data

    def simclr(self, x1, x2):
        h1 = self.online_encoder(x1); h2 = self.online_encoder(x2)
        z1 = self.online_proj(h1);    z2 = self.online_proj(h2)
        return z1, z2

    def byol(self, x1, x2):
        h1 = self.online_encoder(x1); h2 = self.online_encoder(x2)
        z1 = self.online_proj(h1);    z2 = self.online_proj(h2)
        p1 = self.predict_head(z1);   p2 = self.predict_head(z2)
        with torch.no_grad():
            t1 = self.target_proj(self.target_encoder(x2))
            t2 = self.target_proj(self.target_encoder(x1))
        return (p1, t1), (p2, t2)

# ----------------------- 混合模型/基线 -----------------------
class WeatherFeatureFusionModule(nn.Module):
    def __init__(self, img_dim, w_dim, fusion_type="early_concat"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "early_concat":
            self.fc = nn.Linear(img_dim + w_dim, img_dim)
        else:
            self.gamma = nn.Linear(w_dim, img_dim)
            self.beta  = nn.Linear(w_dim, img_dim)
    def forward(self, img, weather):
        if self.fusion_type == "early_concat":
            wf = weather.unsqueeze(1).expand(-1, img.size(1), -1)
            return self.fc(torch.cat([img, wf], dim=-1))
        else:
            g = self.gamma(weather).unsqueeze(1); b = self.beta(weather).unsqueeze(1)
            return img * g + b

class PromptTuningModule(nn.Module):
    def __init__(self, prompt_len=5, dim=512):
        super().__init__(); self.p = nn.Parameter(torch.randn(1, prompt_len, dim)*0.02)
    def forward(self, x):
        return torch.cat([self.p.expand(x.size(0), -1, -1), x], dim=1)

class HybridTransformerModel(nn.Module):
    def __init__(self, num_classes=3, weather_feature_dim=11, backbone="both", use_ssl=True):
        super().__init__()
        self.backbone = backbone
        if backbone in ["convnext","both"]:
            self.conv = create_timm_backbone("convnextv2_base", 0, prefer_pretrained=False,
                                             local_ckpt=CFG["local_weights"].get("convnextv2_base",""))
            self.conv_dim = self.conv.num_features
        if backbone in ["swin","both"]:
            self.swin = create_timm_backbone("swin_base_patch4_window7_224", 0, prefer_pretrained=False,
                                             local_ckpt=CFG["local_weights"].get("swin_base_patch4_window7_224",""))
            self.swin_dim = self.swin.num_features
        if backbone=="both":
            total = self.conv_dim + self.swin_dim
        elif backbone=="convnext":
            total = self.conv_dim
        else:
            total = self.swin_dim

        self.weather = WeatherFeatureFusionModule(total, weather_feature_dim, CFG["fusion"])
        self.prompt = PromptTuningModule(5, total)
        self.head = nn.Sequential(
            nn.LayerNorm(total), nn.Dropout(0.3),
            nn.Linear(total, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def load_ssl_weights(self, ssl_ckpt: str, ssl_method: str):
        if not os.path.exists(ssl_ckpt):
            print(f"自监督权重不存在: {ssl_ckpt}"); return
        ckpt = torch.load(ssl_ckpt, map_location='cpu')
        if ssl_method == "simclr":
            sd = ckpt.get("model_state_dict", {})
            enc = {k.replace("online_encoder.",""): v for k,v in sd.items() if k.startswith("online_encoder.")}
        else:
            enc = ckpt.get("online_encoder_state_dict", {})
        if self.backbone in ["convnext","both"]:
            self.conv.load_state_dict(enc, strict=False)
        if self.backbone in ["swin","both"]:
            self.swin.load_state_dict(enc, strict=False)
        print(f"已加载 SSL 预训练: {ssl_ckpt}")

    def forward(self, x, weather=None):
        if self.backbone in ["convnext","both"]:
            f1 = self.conv(x);
            if self.backbone=="convnext": feat=f1
        if self.backbone in ["swin","both"]:
            f2 = self.swin(x);
            if self.backbone=="swin": feat=f2
        if self.backbone=="both":
            feat = torch.cat([f1, f2], dim=1)
        if weather is not None:
            feat = self.weather(feat.unsqueeze(1), weather).squeeze(1)
        feat = self.prompt(feat.unsqueeze(1)).mean(dim=1)
        logits = self.head(feat)
        return logits, feat

class BaselineResNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = create_timm_backbone("resnet50", 0, False, CFG["local_weights"].get("resnet50",""))
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
    def forward(self, x):
        f = self.backbone(x); return self.fc(f), f

class BaselineViT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = create_timm_backbone("vit_base_patch16_224", 0, False, CFG["local_weights"].get("vit_base_patch16_224",""))
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
    def forward(self, x):
        f = self.backbone(x); return self.fc(f), f

# ----------------------- 数据集 -----------------------
class PomegranateDiseaseDataset(Dataset):
    """训练/验证/测试（带天气）"""
    def __init__(self, root: str, img_dir: str, w_dir: str, transform=None, self_ssl=False):
        super().__init__()
        self.img_root = os.path.join(root, img_dir)
        self.w_root = os.path.join(root, w_dir)
        self.transform = transform
        self.self_ssl = self_ssl
        self.weather_map = self._load_weather()
        self.samples = []  # (img_path, label, week, fruit_id, weather)
        self._build()
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")

    def _load_weather(self):
        mp = {}
        files = glob.glob(os.path.join(self.w_root, "*.xlsx"))
        if not files: print(f"未找到气象 Excel：{self.w_root}")
        for fp in files:
            try:
                df = pd.read_excel(fp); wk = os.path.basename(fp).split('.')[0]
                feats = [float(pd.to_numeric(df[c], errors='coerce').mean()) if c in df.columns else 0.0
                         for c in CFG["weather_pref_cols"]]
                mp[wk]=feats; print(f"加载气象数据: {wk}, 特征维度:{len(feats)}")
            except Exception as e:
                print(f"加载气象失败 {fp}: {e}")
        return mp

    def _build(self):
        files = glob.glob(os.path.join(self.img_root, "week_*", "*", "*.jpg"))
        if not files: print(f"未在 {self.img_root} 找到图像")
        for p in files:
            parts = p.split(os.sep)
            week = parts[-3]; tree = parts[-2]; name = os.path.splitext(os.path.basename(p))[0]
            fruit_id = f"{tree}-{name}"
            try: wn = int(week.split('_')[1])
            except: wn = 1
            # 规则生成“伪标签”，仅用于训练示例（若你有真标签请替换）
            if wn <= 5: phase='健康期'
            elif wn <= 10: phase='初发期'
            else: phase='发病期'
            label = PHASE_MAP[phase]
            weather = self.weather_map.get(week, [0.0]*len(CFG["weather_pref_cols"]))
            self.samples.append((p,label,week,fruit_id,weather))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,y,wk,fid,wf = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.self_ssl:
            if self.transform:
                return self.transform(img), self.transform(img), y
        else:
            if self.transform: img = self.transform(img)
            return img, y, wk, fid, np.array(wf, dtype=np.float32)

class PredictAttachmentDataset(Dataset):
    """用于任务一（附件1整包）或任务二（附件3整包）的无标签推理"""
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []  # (img_path, week, fruit_id)
        files = glob.glob(os.path.join(root_dir, "week_*", "*", "*.jpg"))
        for p in files:
            parts = p.split(os.sep)
            week = parts[-3]; tree=parts[-2]; name=os.path.splitext(os.path.basename(p))[0]
            self.samples.append((p, week, f"{tree}-{name}"))
        if not self.samples:
            print(f"未在 {root_dir} 找到图像")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,wk,fid = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, wk, fid

# ----------------------- 数据增强 -----------------------
def get_ssl_transforms(s=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(s, scale=(0.2,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
def get_train_transforms(s=224):
    return transforms.Compose([
        transforms.Resize((s,s)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
def get_test_transforms(s=224):
    return transforms.Compose([
        transforms.Resize((s,s)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# ----------------------- 自监督预训练 -----------------------
def self_supervised_pretraining(backbone="convnext", method="simclr"):
    print(f"=== SSL 预训练: {method} - {backbone} ===")
    device = device_of()
    out_dir = os.path.join(CFG["outputs"], "自监督预训练", f"{method}_{backbone}")
    ensure_dir(out_dir)

    ds = PomegranateDiseaseDataset(CFG["data_root"], CFG["image_dir"], CFG["weather_dir"],
                                   transform=get_ssl_transforms(CFG["img_size"]), self_ssl=True)
    # 关键：drop_last=True，保证 batch 配对完整
    dl = DataLoader(ds, batch_size=CFG["batch_size_ssl"], shuffle=True, num_workers=0, drop_last=True)

    model = SelfSupervisedModel(backbone=backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr_ssl"], weight_decay=CFG["weight_decay"])
    loss_fn = NTXentLoss(0.1) if method=="simclr" else BYOLLoss()

    best = 1e9; history=[]
    for ep in range(1, CFG["epochs_ssl"]+1):
        model.train(); tot=0.0
        for v1, v2, _ in dl:
            v1, v2 = v1.to(device), v2.to(device)
            # 健壮性断言
            B = v1.size(0)
            assert B >= 2, "SSL 批次过小，请提高 CFG['batch_size_ssl'] 并保持 drop_last=True"
            opt.zero_grad()
            if method=="simclr":
                z1, z2 = model.simclr(v1, v2)
                # 数值稳定检查
                assert torch.isfinite(z1).all() and torch.isfinite(z2).all(), "检测到 NaN/Inf 投影"
                loss = loss_fn(z1, z2)
            else:
                (p1,t1),(p2,t2) = model.byol(v1, v2)
                loss = (loss_fn(p1,t1)+loss_fn(p2,t2))/2; model.update_target()
            loss.backward(); opt.step(); tot += loss.item()
        avg = tot/max(1,len(dl)); history.append({"epoch":ep,"loss":avg})
        print(f"[SSL {ep}/{CFG['epochs_ssl']}] loss={avg:.4f}")
        if avg < best:
            best = avg
            ckpt = os.path.join(out_dir, f"最佳_{method}_{backbone}.pth")
            torch.save({'model_state_dict': model.state_dict(),
                        'online_encoder_state_dict': model.online_encoder.state_dict(),
                        'epoch': ep, 'loss': avg}, ckpt)
            print(f"保存最佳 SSL 模型: {ckpt}")
    pd.DataFrame(history).to_csv(os.path.join(out_dir,"训练记录.csv"), index=False)
    return out_dir

# ----------------------- 监督微调/评估 -----------------------
def train_supervised_model(model_name="HybridModel", backbone="convnext",
                           use_ssl=False, ssl_method="simclr"):
    print(f"=== 监督训练: {model_name} / backbone={backbone} / SSL={use_ssl and ssl_method} ===")
    set_seed(CFG["seed"]); device = device_of()
    ensure_dir(CFG["outputs"]); ensure_dir(os.path.join(CFG["outputs"],"可视化结果")); ensure_dir(os.path.join(CFG["outputs"],"模型权重"))

    ds = PomegranateDiseaseDataset(CFG["data_root"], CFG["image_dir"], CFG["weather_dir"],
                                   transform=get_train_transforms(CFG["img_size"]))
    n=len(ds); tr=int(0.7*n); va=int(0.15*n); te=n-tr-va
    gen = torch.Generator().manual_seed(CFG["seed"])
    tr_set, va_set, te_set = random_split(ds, [tr,va,te], generator=gen)
    tr_dl = DataLoader(tr_set, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_set, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    te_dl = DataLoader(te_set, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    if model_name=="HybridModel":
        model = HybridTransformerModel(CFG["num_classes"], len(CFG["weather_pref_cols"]), backbone=backbone, use_ssl=use_ssl).to(device)
        if use_ssl:
            # 若 backbone="both"，约定加载 convnext 预训练（可自行扩展为两主干分别加载）
            load_key = backbone if backbone!="both" else "convnext"
            ssl_ckpt = os.path.join(CFG["outputs"], "自监督预训练", f"{ssl_method}_{load_key}", f"最佳_{ssl_method}_{load_key}.pth")
            model.load_ssl_weights(ssl_ckpt, ssl_method)
    elif model_name=="ResNet":
        model = BaselineResNet(CFG["num_classes"]).to(device)
    elif model_name=="ViT":
        model = BaselineViT(CFG["num_classes"]).to(device)
    else:
        raise ValueError("未知模型名")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr_ft"], weight_decay=CFG["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs_ft"])

    best=0.0; best_path=os.path.join(CFG["outputs"],"模型权重",f"最佳_{model_name}_{backbone}{'_ssl' if use_ssl else '_scratch'}.pth")
    history=[]
    for ep in range(1, CFG["epochs_ft"]+1):
        model.train(); tr_loss=0.0
        for img,y,_,_,wf in tr_dl:
            img=img.to(device); y=torch.as_tensor(y,device=device, dtype=torch.long); wf=torch.as_tensor(wf,device=device)
            opt.zero_grad()
            logits,_ = model(img, wf) if model_name=="HybridModel" else model(img)
            # 标签安全检查
            assert y.min().item() >= 0 and y.max().item() < CFG["num_classes"], "标签越界"
            loss = loss_fn(logits,y); loss.backward(); opt.step(); tr_loss += loss.item()
        # val
        model.eval(); corr=0; tot=0
        with torch.no_grad():
            for img,y,_,_,wf in va_dl:
                img=img.to(device); y=torch.as_tensor(y,device=device, dtype=torch.long); wf=torch.as_tensor(wf,device=device)
                logits,_ = model(img, wf) if model_name=="HybridModel" else model(img)
                pred = logits.argmax(1); corr += (pred==y).sum().item(); tot += y.size(0)
        val_acc = corr/max(1,tot)
        sch.step()
        rec={'epoch':ep,'train_loss':tr_loss/max(1,len(tr_dl)),'val_accuracy':val_acc,'lr':sch.get_last_lr()[0]}
        history.append(rec)
        print(f"[{ep}/{CFG['epochs_ft']}] loss={rec['train_loss']:.4f} val={val_acc:.4f} lr={rec['lr']:.6f}")
        if val_acc>best:
            best=val_acc; torch.save(model.state_dict(), best_path); print(f"保存最佳: {best_path} (val={best:.4f})")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(CFG["outputs"], f"{model_name}_{backbone}{'_ssl' if use_ssl else '_scratch'}_训练记录.csv"),
                   index=False, encoding='utf-8-sig')
    return model, hist_df, te_dl, device

@torch.no_grad()
def evaluate_model(model, dl, device, model_name="HybridModel"):
    model.eval()
    preds, labels, feats, weeks, fids = [], [], [], [], []
    for img,y,wk,fid,wf in dl:
        img=img.to(device); wf=torch.as_tensor(wf,device=device)
        logits, f = model(img, wf) if model_name=="HybridModel" else model(img)
        p = logits.argmax(1)
        preds.extend(p.cpu().numpy()); labels.extend(np.array(y))
        feats.extend(f.cpu().numpy()); weeks.extend(wk); fids.extend(fid)
    vis = os.path.join(CFG["outputs"], "可视化结果", model_name)
    perf = generate_chinese_visualization(labels, preds, np.array(feats), vis)
    return preds, labels, weeks, fids, perf

# ----------------------- 任务一（附件1整包） & 任务二（附件3整包） -----------------------
@torch.no_grad()
def predict_folder_and_save(folder_dir: str, model, device, out_name: str, model_name="HybridModel"):
    ds = PredictAttachmentDataset(folder_dir, transform=get_test_transforms(CFG["img_size"]))
    dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    preds, weeks, fids = [], [], []
    for img, wk, fid in dl:
        img=img.to(device)
        wf_zeros = torch.zeros((img.size(0), len(CFG["weather_pref_cols"])), device=device, dtype=img.dtype)
        logits,_ = model(img, wf_zeros) if model_name=="HybridModel" else model(img)
        p = logits.argmax(1).cpu().numpy().tolist()
        preds.extend(p); weeks.extend(list(wk)); fids.extend(list(fid))
    rows = [{'周数': w.split('_(')[0], '果实编号': fid, '果实阶段': IDX2LABEL[int(pp)]}
            for w,fid,pp in zip(weeks, fids, preds)]
    df = pd.DataFrame(rows)
    out_path = os.path.join(CFG["outputs"], out_name)
    ensure_dir(os.path.dirname(out_path))
    df.to_excel(out_path, index=False)
    print(f"已保存：{out_path}")
    return df

# ----------------------- 消融实验（只比较两个核心因素） -----------------------
def run_ablation_minimal():
    """
    仅比较两个因素：
      1) SSL 贡献：scratch vs SimCLR-SSL vs BYOL-SSL（同一主干）
      2) 主干选择：ConvNeXt vs Swin（在各自最优 SSL 设置下对比）
    """
    results = []

    for backbone in ["convnext", "swin"]:
        print(f"\n### Backbone = {backbone}")

        # (A) 无预训练
        model_s, _, te_dl, dev = train_supervised_model("HybridModel", backbone, use_ssl=False)
        preds, labels, _, _, perf_s = evaluate_model(model_s, te_dl, dev, "HybridModel")
        acc_s = float(perf_s.iloc[0,1]); f1_s = float(perf_s.iloc[1,1]); kap_s = float(perf_s.iloc[2,1])
        results.append({"主干":backbone,"设置":"scratch","Acc":acc_s,"F1":f1_s,"Kappa":kap_s})

        # (B) SimCLR 预训练
        _ = self_supervised_pretraining(backbone, "simclr")
        model_sim, _, te_dl, dev = train_supervised_model("HybridModel", backbone, use_ssl=True, ssl_method="simclr")
        preds, labels, _, _, perf_sim = evaluate_model(model_sim, te_dl, dev, "HybridModel")
        acc_sim = float(perf_sim.iloc[0,1]); f1_sim = float(perf_sim.iloc[1,1]); kap_sim = float(perf_sim.iloc[2,1])
        results.append({"主干":backbone,"设置":"simclr","Acc":acc_sim,"F1":f1_sim,"Kappa":kap_sim})

        # (C) BYOL 预训练
        _ = self_supervised_pretraining(backbone, "byol")
        model_byol, _, te_dl, dev = train_supervised_model("HybridModel", backbone, use_ssl=True, ssl_method="byol")
        preds, labels, _, _, perf_byol = evaluate_model(model_byol, te_dl, dev, "HybridModel")
        acc_byol = float(perf_byol.iloc[0,1]); f1_byol = float(perf_byol.iloc[1,1]); kap_byol = float(perf_byol.iloc[2,1])
        results.append({"主干":backbone,"设置":"byol","Acc":acc_byol,"F1":f1_byol,"Kappa":kap_byol})

    table = pd.DataFrame(results)
    ensure_dir(CFG["outputs"])
    table.to_csv(os.path.join(CFG["outputs"], "消融实验_两因素.csv"), index=False, encoding="utf-8-sig")

    # 简图
    plt.figure(figsize=(12,6))
    x = np.arange(len(table)); w=0.25
    plt.bar(x-w, table["Acc"], width=w, label="Acc", alpha=0.85)
    plt.bar(x,   table["F1"],  width=w, label="F1",  alpha=0.85)
    plt.bar(x+w, table["Kappa"], width=w, label="Kappa", alpha=0.85)
    plt.xticks(x, [f"{r['主干']}-{r['设置']}" for _,r in table.iterrows()], rotation=30)
    plt.title("消融实验：SSL 贡献 & 主干选择"); plt.legend(); plt.grid(True, axis='y', alpha=0.3)
    ensure_dir(os.path.join(CFG["outputs"], "可视化结果"))
    plt.tight_layout(); plt.savefig(os.path.join(CFG["outputs"], "可视化结果", "消融_两因素.png"), dpi=300); plt.close()

    return table

# ----------------------- 主程序 -----------------------
def main():
    print("开始：自监督 + 混合Transformer + 本地权重优先 + 任务一/任务二")
    if timm is None:
        print("错误: 需要安装 timm (pip install timm)"); sys.exit(1)
    if not os.path.exists(CFG["data_root"]):
        print(f"错误: 数据路径不存在: {CFG['data_root']}"); sys.exit(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"使用GPU: {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    set_seed(CFG["seed"])

    # 1) 消融（两因素）
    ablation_tbl = run_ablation_minimal()
    # 选出各主干中最优设置
    best_rows = ablation_tbl.sort_values(["主干","Acc"], ascending=[True,False]).groupby("主干").head(1)
    print("\n各主干最优设置：")
    print(best_rows)

    # 2) 用整体最优（按 Acc）重新训练一个“最终模型”
    best_row = ablation_tbl.loc[ablation_tbl["Acc"].idxmax()]
    best_backbone = best_row["主干"]; best_setting = best_row["设置"]  # scratch / simclr / byol
    use_ssl = (best_setting != "scratch"); ssl_method = best_setting if use_ssl else "simclr"
    final_model, _, test_dl, dev = train_supervised_model("HybridModel", best_backbone, use_ssl, ssl_method)
    # 测试并画图/表
    preds, gts, weeks, fids, perf = evaluate_model(final_model, test_dl, dev, "HybridModel")

    # 3) 任务一：对附件1整包推理并输出 result1
    folder1 = os.path.join(CFG["data_root"], CFG["image_dir"])
    _ = predict_folder_and_save(folder1, final_model, dev, out_name="result1_HybridModel.xlsx", model_name="HybridModel")

    # 4) 任务二：对附件3整包推理并输出 result2
    folder3 = os.path.join(CFG["data_root"], CFG["predict_dir"])
    _ = predict_folder_and_save(folder3, final_model, dev, out_name="result2_HybridModel.xlsx", model_name="HybridModel")

    print(f"\n全流程完成；输出目录：{CFG['outputs']}")

if __name__ == "__main__":

    main()
