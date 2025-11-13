# -*- coding: utf-8 -*-
import os, re, glob, math, random, warnings, json, tempfile
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from PIL import Image
import torchvision.transforms as T

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# =========================================================
# 设备与后端：务必在任何用到 device 之前
# =========================================================
if not torch.cuda.is_available():
    raise SystemError("未检测到可用的 CUDA GPU。请在支持 CUDA 的环境下运行，或将本检查改为 CPU 回退。")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# 防重复打印（只打印一次设备信息）
try:
    _PRINTED_ONCE
except NameError:
    _PRINTED_ONCE = False

if not _PRINTED_ONCE:
    print("===== 设备信息 =====")
    print("当前设备：", device)
    print("GPU 数量：", torch.cuda.device_count())
    print("GPU 名称：", torch.cuda.get_device_name(0))
    print("CUDA 能力：", torch.cuda.get_device_capability(0))
    print("====================================================\n")
    _PRINTED_ONCE = True

# =========================================================
# 目录与全局配置（集中管理 + ASCII 安全回退）
# =========================================================
# 数据根目录
BASE_PATH = r"D:\桌面\本科组数据"
IMG_DIR = os.path.join(BASE_PATH, "附件1")
WEA_DIR = os.path.join(BASE_PATH, "附件2")
PRED_DIR = os.path.join(BASE_PATH, "附件3")

# 结果主目录（可能含中文）
OUT_DIR = r"D:\q3_results"
FIG_DIR = os.path.join(OUT_DIR, "figs")
CKPT_DIR = os.path.join(OUT_DIR, "ckpts")
TASK3_DIR = os.path.join(OUT_DIR, "任务三_推理输出")

# ASCII 安全回退目录（确保任何环境都能写）
OUT_DIR_FALLBACK = os.path.join(tempfile.gettempdir(), "q3_results")
FIG_DIR_FB = os.path.join(OUT_DIR_FALLBACK, "figs")
CKPT_DIR_FB = os.path.join(OUT_DIR_FALLBACK, "ckpts")
TASK3_DIR_FB = os.path.join(OUT_DIR_FALLBACK, "task3_outputs")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def init_dirs():
    for d in [OUT_DIR, FIG_DIR, CKPT_DIR, TASK3_DIR, OUT_DIR_FALLBACK, FIG_DIR_FB, CKPT_DIR_FB, TASK3_DIR_FB]:
        ensure_dir(d)


# 初始化目录
init_dirs()

# 其他全局参数
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LABELS = ["健康期", "初发期", "发病期"]
PHASE2IDX = {k: i for i, k in enumerate(LABELS)}
IDX2PHASE = {i: k for k, i in PHASE2IDX.items()}

# 训练超参 - 降低学习率，增加权重衰减以防止过拟合和梯度爆炸
SEQ_LEN = 4
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4  # 降低学习率
WEIGHT_DECAY = 1e-3  # 增加权重衰减
DROPOUT = 0.2
NUM_HEADS = 4
HIDDEN = 256
VISUAL_DIM = 512
WEATHER_COLS_PREF = ["temp", "dewpt", "rh", "precip_rate", "solar_rad", "ghi", "dhi", "dni", "pres", "wind_spd", "vis"]

# 课程学习
CURRICULUM_WARM_EPOCHS = 8

# 蒸馏 - 调整参数
KD_ALPHA = 0.3  # 降低知识蒸馏权重
KD_TEMPERATURE = 2.0  # 降低温度参数
EMA_DECAY = 0.995  # 调整EMA衰减

# MC Dropout
MC_PASSES = 10

# 梯度裁剪
GRAD_CLIP = 1.0

# 研究配色
research_colors = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8',
                   '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette(research_colors)


# ===================== 实用函数 =====================
def safe_save_state_dict(state_dict: dict, primary_path: str) -> str:
    primary_dir = os.path.dirname(primary_path)
    try:
        ensure_dir(primary_dir)
        torch.save(state_dict, primary_path)
        print(f"  -> 保存最佳 (主路径): {primary_path}")
        return primary_path
    except Exception as e:
        print(f"  !! 主路径保存失败: {e}")

    fb_path = os.path.join(CKPT_DIR_FB, os.path.basename(primary_path))
    try:
        ensure_dir(CKPT_DIR_FB)
        torch.save(state_dict, fb_path)
        print(f"  -> 保存最佳 (回退路径): {fb_path}")
        return fb_path
    except Exception as e2:
        raise RuntimeError(f"主路径与回退路径均保存失败。最后错误: {e2}")


def parse_week_num(week_str: str) -> int:
    m = re.search(r"week_(\d+)", week_str)
    return int(m.group(1)) if m else 1


def scan_attachment1() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(IMG_DIR):
        raise RuntimeError(f"附件1目录不存在：{IMG_DIR}")
    week_dirs = sorted([d for d in os.listdir(IMG_DIR) if d.startswith("week_")])
    for wk in week_dirs:
        wkdir = os.path.join(IMG_DIR, wk)
        if not os.path.isdir(wkdir):
            continue
        tree_dirs = sorted([d for d in os.listdir(wkdir) if os.path.isdir(os.path.join(wkdir, d))])
        for tree in tree_dirs:
            tdir = os.path.join(wkdir, tree)
            for img in os.listdir(tdir):
                if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    name = os.path.splitext(img)[0]
                    rows.append({
                        "week": wk,
                        "week_num": parse_week_num(wk),
                        "tree": tree,
                        "fruit_id": f"{tree}-{name}",
                        "image_path": os.path.join(tdir, img)
                    })
    return pd.DataFrame(rows)


def read_weather_means() -> Dict[str, Dict[str, float]]:
    means = {}
    if not os.path.isdir(WEA_DIR):
        raise RuntimeError(f"附件2目录不存在：{WEA_DIR}")
    xls = sorted(glob.glob(os.path.join(WEA_DIR, "week_*.xlsx")))
    for fp in xls:
        wk = os.path.basename(fp).split(".")[0]
        try:
            df = pd.read_excel(fp)
            d = {}
            for c in WEATHER_COLS_PREF:
                if c in df.columns:
                    d[c] = float(pd.to_numeric(df[c], errors="coerce").mean())
                else:
                    d[c] = 0.0
            means[wk] = d
        except Exception as e:
            print(f"[weather] 读取失败 {fp}: {e}")
    return means


def make_pseudo_label(week_num: int) -> int:
    if week_num <= 5:  return PHASE2IDX["健康期"]
    if week_num <= 10: return PHASE2IDX["初发期"]
    return PHASE2IDX["发病期"]


def group_split_by_fruit(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    fruits = df["fruit_id"].unique().tolist()
    random.shuffle(fruits)
    n = len(fruits);
    n_tr = int(n * train_ratio);
    n_val = int(n * val_ratio)
    tr_ids = set(fruits[:n_tr]);
    val_ids = set(fruits[n_tr:n_tr + n_val]);
    te_ids = set(fruits[n_tr + n_val:])
    tr = df[df["fruit_id"].isin(tr_ids)].copy()
    va = df[df["fruit_id"].isin(val_ids)].copy()
    te = df[df["fruit_id"].isin(te_ids)].copy()
    return tr, va, te


# ===================== 数据集定义 =====================
class MultiModalSeqDataset(Dataset):
    def __init__(self, df_all: pd.DataFrame, weather_means: Dict[str, Dict[str, float]],
                 seq_len=4, transform=None, use_curriculum=False):
        super().__init__()
        self.df_all = df_all.copy()
        self.weather_means = weather_means
        self.seq_len = seq_len
        self.transform = transform
        self.use_curriculum = use_curriculum
        self.samples = self._build_samples()

        if self.use_curriculum:
            self.difficulty = np.array([s["target_week_num"] for s in self.samples], dtype=np.float32)
            self.difficulty = (self.difficulty - self.difficulty.min()) / max(1e-6, (
                    self.difficulty.max() - self.difficulty.min()))

    def _build_samples(self):
        samples = []
        for fid, g in self.df_all.groupby("fruit_id"):
            g = g.sort_values("week_num")
            if len(g) < self.seq_len:
                continue
            for i in range(self.seq_len - 1, len(g)):
                window = g.iloc[i - self.seq_len + 1: i + 1]
                target_row = g.iloc[i]
                samples.append({
                    "fruit_id": fid,
                    "weeks": window["week"].tolist(),
                    "week_nums": window["week_num"].tolist(),
                    "img_paths": window["image_path"].tolist(),
                    "target_week": target_row["week"],
                    "target_week_num": int(target_row["week_num"]),
                    "label": int(target_row["label"])
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs = []
        for p in s["img_paths"]:
            try:
                im = Image.open(p).convert("RGB")
            except:
                im = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform: im = self.transform(im)
            imgs.append(im)
        imgs = torch.stack(imgs, dim=0)  # (seq, 3, H, W)

        ws = []
        for wk in s["weeks"]:
            v = [self.weather_means.get(wk, {}).get(c, 0.0) for c in WEATHER_COLS_PREF]
            ws.append(v)
        ws = torch.tensor(ws, dtype=torch.float32)

        y = torch.tensor(s["label"], dtype=torch.long)
        return imgs, ws, y, s["target_week"]


class CurriculumSampler(Sampler):
    def __init__(self, dataset: MultiModalSeqDataset, epoch_ref: List[int], warm_epochs=8):
        self.dataset = dataset
        self.epoch_ref = epoch_ref
        self.warm_epochs = warm_epochs
        self.N = len(dataset)

    def __iter__(self):
        epoch = self.epoch_ref[0]
        indices = np.arange(self.N)
        if epoch < self.warm_epochs and hasattr(self.dataset, "difficulty"):
            scores = self.dataset.difficulty
            order = np.argsort(scores)
            frac = np.clip(0.4 + 0.06 * epoch, 0.4, 1.0)
            k = max(BATCH_SIZE, int(len(order) * frac))  # 确保至少选择BATCH_SIZE个样本
            chosen = order[:k]
            np.random.shuffle(chosen)
            return iter(chosen.tolist())
        else:
            np.random.shuffle(indices)
            return iter(indices.tolist())

    def __len__(self):
        return self.N


# ===================== 模型定义 =====================
class PatchCNN(nn.Module):
    def __init__(self, out_dim=VISUAL_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class CrossModalAttention(nn.Module):
    def __init__(self, v_dim, w_dim, d_model=HIDDEN, nhead=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, d_model)
        self.w_proj = nn.Linear(w_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # 初始化
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.w_proj.weight)

    def forward(self, v_seq, w_seq):
        q = self.v_proj(v_seq)
        k = self.w_proj(w_seq)
        v = k
        out, attn = self.attn(q, k, v, need_weights=True, average_attn_weights=False)
        out = self.ln(q + self.drop(out))
        return out, attn


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim, channels=[256, 128], kernel=3, dropout=DROPOUT):
        super().__init__()
        layers = []
        last = in_dim
        for i, ch in enumerate(channels):
            dil = 2 ** i
            pad = (kernel - 1) * dil // 2
            layers += [
                nn.Conv1d(last, ch, kernel, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            last = ch
        self.net = nn.Sequential(*layers)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=HIDDEN, nhead=NUM_HEADS, num_layers=2, dropout=DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout,
            dim_feedforward=d_model * 4
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model, max_len=128)

    def forward(self, x):
        x = self.pos(x)
        return self.encoder(x)


class MultiModalDiseasePredictor(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.visual = PatchCNN(out_dim=VISUAL_DIM, dropout=DROPOUT)
        self.cross = CrossModalAttention(v_dim=VISUAL_DIM, w_dim=len(WEATHER_COLS_PREF), d_model=HIDDEN,
                                         nhead=NUM_HEADS, dropout=DROPOUT)
        self.tcn = TemporalConvNet(in_dim=HIDDEN, channels=[HIDDEN, HIDDEN // 2], kernel=3, dropout=DROPOUT)
        self.tfm = TemporalTransformer(d_model=HIDDEN // 2, nhead=NUM_HEADS, num_layers=2, dropout=DROPOUT)
        self.lstm = nn.LSTM(input_size=HIDDEN // 2, hidden_size=HIDDEN, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=0.0)

        # 分类器
        self.cls = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, num_classes)
        )

        # 初始化分类器
        for m in self.cls.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_seq, wea_seq, mc_dropout=False):
        # img_seq: (B, L, 3, H, W)
        B, L = img_seq.size(0), img_seq.size(1)
        # 展平时间维 -> (B*L, 3, H, W)
        x4d = img_seq.reshape(B * L, *img_seq.shape[2:])
        v_flat = self.visual(x4d)  # (B*L, VISUAL_DIM)
        v = v_flat.view(B, L, -1)  # (B, L, VISUAL_DIM)

        x, attn = self.cross(v, wea_seq)  # (B, L, HIDDEN)
        x = self.tcn(x)  # (B, L, HIDDEN)
        x = self.tfm(x)  # (B, L, HIDDEN//2)

        if mc_dropout:
            x = F.dropout(x, p=DROPOUT, training=True)

        x, _ = self.lstm(x)  # (B, L, 2*HIDDEN)
        last = x[:, -1, :]  # (B, 2*HIDDEN)
        logits = self.cls(last)  # (B, num_classes)
        return logits, attn


# ============ 知识蒸馏：Teacher 为 Student 的 EMA 影子 ============
class EMATeacher:
    def __init__(self, model: nn.Module, decay=EMA_DECAY):
        self.decay = decay
        self.teacher = MultiModalDiseasePredictor(num_classes=3).to(device)
        self.teacher.load_state_dict(model.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.data = self.decay * tp.data + (1 - self.decay) * sp.data


def kd_loss(student_logits, teacher_logits, T=KD_TEMPERATURE):
    ps = F.log_softmax(student_logits / T, dim=1)
    pt = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(ps, pt, reduction="batchmean") * (T * T)


# ===================== 训练与评估 =====================
def build_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def prepare_dataframe() -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    print("== 解析附件1 + 附件2 ==")
    df = scan_attachment1()
    if df.empty:
        raise RuntimeError("附件1未解析到任何图像。")
    weather_means = read_weather_means()
    df["label"] = df["week_num"].apply(make_pseudo_label)
    return df, weather_means


def train_val_test_loaders(df_all, weather_means):
    tr_df, va_df, te_df = group_split_by_fruit(df_all, 0.7, 0.15)
    print(
        f"[Split] train fruits={tr_df['fruit_id'].nunique()}, val={va_df['fruit_id'].nunique()}, test={te_df['fruit_id'].nunique()}")

    tr_tf, te_tf = build_transforms()
    epoch_ref = [0]

    tr_set = MultiModalSeqDataset(tr_df, weather_means, seq_len=SEQ_LEN, transform=tr_tf, use_curriculum=True)
    va_set = MultiModalSeqDataset(va_df, weather_means, seq_len=SEQ_LEN, transform=te_tf, use_curriculum=False)
    te_set = MultiModalSeqDataset(te_df, weather_means, seq_len=SEQ_LEN, transform=te_tf, use_curriculum=False)

    tr_samp = CurriculumSampler(tr_set, epoch_ref=epoch_ref, warm_epochs=CURRICULUM_WARM_EPOCHS)

    # 高效 DataLoader
    num_workers = max(2, os.cpu_count() // 2)
    tr_loader = DataLoader(
        tr_set, batch_size=BATCH_SIZE, sampler=tr_samp,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2  # 减少预取因子
    )
    va_loader = DataLoader(
        va_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    te_loader = DataLoader(
        te_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    return tr_loader, va_loader, te_loader, epoch_ref


def plot_training_curves(tr_losses, va_losses, tr_accs, va_accs, save_path):
    try:
        epochs = range(1, len(tr_losses) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_losses, label="训练损失", linewidth=2)
        plt.plot(epochs, va_losses, label="验证损失", linewidth=2)
        plt.xlabel("轮次");
        plt.ylabel("损失");
        plt.title("训练/验证损失曲线");
        plt.grid(True, alpha=0.3);
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, tr_accs, label="训练准确率", linewidth=2)
        plt.plot(epochs, va_accs, label="验证准确率", linewidth=2)
        plt.xlabel("轮次");
        plt.ylabel("准确率");
        plt.title("训练/验证准确率曲线");
        plt.grid(True, alpha=0.3);
        plt.legend()

        plt.tight_layout();
        plt.savefig(save_path, dpi=300);
        plt.close()
        print(f"✅ 已保存图：{save_path}")
    except Exception as e:
        fb_path = os.path.join(FIG_DIR_FB, os.path.basename(save_path))
        ensure_dir(os.path.dirname(fb_path))
        plt.tight_layout();
        plt.savefig(fb_path, dpi=300);
        plt.close()
        print(f"✅ 已保存图（回退路径）：{fb_path}；主路径失败：{e}")


def plot_confusion(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测标签");
    plt.ylabel("真实标签");
    plt.title("混淆矩阵")
    try:
        plt.tight_layout();
        plt.savefig(save_path, dpi=300);
        plt.close()
        print(f"✅ 已保存图：{save_path}")
    except Exception as e:
        fb_path = os.path.join(FIG_DIR_FB, os.path.basename(save_path))
        ensure_dir(os.path.dirname(fb_path))
        plt.tight_layout();
        plt.savefig(fb_path, dpi=300);
        plt.close()
        print(f"✅ 已保存图（回退路径）：{fb_path}；主路径失败：{e}")


def check_nan_in_model(model):
    """检查模型中是否有NaN参数"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"警告: 参数 {name} 包含NaN值")
            return True
    return False


def run_train():
    df_all, weather_means = prepare_dataframe()
    tr_loader, va_loader, te_loader, epoch_ref = train_val_test_loaders(df_all, weather_means)

    model = MultiModalDiseasePredictor(num_classes=len(LABELS)).to(device)
    # 尝试编译模型（Windows 不支持会给出 warning）
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"[warn] torch.compile 不可用或失败：{e}")

    teacher = EMATeacher(model, decay=EMA_DECAY)

    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=EPOCHS)
    # 添加学习率调度器
    reduce_lr_scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5, verbose=True)
    ce = nn.CrossEntropyLoss()

    tr_losses = [];
    va_losses = [];
    tr_accs = [];
    va_accs = []
    best_val = 0.0

    best_path_ref = [os.path.join(CKPT_DIR, "best_model.pth")]

    scaler = torch.cuda.amp.GradScaler()

    # 早停参数
    patience = 10
    early_stop_counter = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_ref[0] = epoch
        # ---- Train
        model.train()
        tl = 0.0;
        tc = 0;
        tn = 0
        nan_detected = False

        for imgs, wea, y, _ in tr_loader:
            # 检查输入数据
            if torch.isnan(imgs).any() or torch.isnan(wea).any() or torch.isnan(y).any():
                print("警告: 训练数据中包含NaN值，跳过该批次")
                continue

            imgs = imgs.to(device, non_blocking=True)
            wea = wea.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, _ = model(imgs, wea, mc_dropout=False)

                # 检查logits是否包含NaN
                if torch.isnan(logits).any():
                    print("警告: 模型输出包含NaN值，跳过该批次")
                    nan_detected = True
                    break

                with torch.no_grad():
                    t_logits, _ = teacher.teacher(imgs, wea, mc_dropout=False)

                loss_ce = ce(logits, y)
                loss_kd = kd_loss(logits, t_logits, T=KD_TEMPERATURE)
                loss = (1 - KD_ALPHA) * loss_ce + KD_ALPHA * loss_kd

                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print("警告: 损失为NaN或无穷大，跳过该批次")
                    nan_detected = True
                    break

            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optim)
            scaler.update()

            # 清空梯度
            optim.zero_grad(set_to_none=True)

            teacher.update(model)

            tl += loss.item() * imgs.size(0)
            pred = logits.argmax(1)
            tc += (pred == y).sum().item()
            tn += y.size(0)

        # 如果检测到NaN，跳过该epoch
        if nan_detected or tn == 0:
            print(f"Epoch {epoch + 1}: 检测到NaN，跳过该epoch")
            # 重置模型状态到上一个检查点
            if epoch > 0 and os.path.exists(best_path_ref[0]):
                model.load_state_dict(torch.load(best_path_ref[0]))
            continue

        tr_loss = tl / max(1, tn)
        tr_acc = tc / max(1, tn)
        tr_losses.append(tr_loss);
        tr_accs.append(tr_acc)

        # ---- Val
        model.eval()
        vl = 0.0;
        vc = 0;
        vn = 0
        y_true = [];
        y_pred = []
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            for imgs, wea, y, _ in va_loader:
                imgs = imgs.to(device, non_blocking=True)
                wea = wea.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits, _ = model(imgs, wea, mc_dropout=False)
                loss = ce(logits, y)
                vl += loss.item() * imgs.size(0)
                p = logits.argmax(1)
                vc += (p == y).sum().item();
                vn += y.size(0)
                y_true.extend(y.cpu().tolist());
                y_pred.extend(p.cpu().tolist())
        va_loss = vl / max(1, vn);
        va_acc = vc / max(1, vn)
        va_losses.append(va_loss);
        va_accs.append(va_acc)

        sched.step()
        reduce_lr_scheduler.step(va_loss)  # 基于验证损失调整学习率

        print(
            f"[{epoch + 1:02d}/{EPOCHS}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} lr={optim.param_groups[0]['lr']:.6e}")

        # 早停检查
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"早停: 验证损失在 {patience} 个epoch内未改善")
            break

        if va_acc > best_val:
            best_val = va_acc
            target_primary = os.path.join(CKPT_DIR, "best_model.pth")
            actual_path = safe_save_state_dict(model.state_dict(), target_primary)
            best_path_ref[0] = actual_path
            print(f"  -> 当前最佳验证准确率: {best_val:.4f}")

        # 检查模型参数是否有NaN
        if check_nan_in_model(model):
            print("警告: 模型参数包含NaN，停止训练")
            break

    # 画训练曲线
    plot_training_curves(tr_losses, va_losses, tr_accs, va_accs, os.path.join(FIG_DIR, "图_训练曲线.png"))

    # 混淆矩阵（验证集）
    y_true = [];
    y_pred = []
    best_path = best_path_ref[0]
    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location="cpu")
        model.load_state_dict(sd)
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        for imgs, wea, y, _ in va_loader:
            imgs = imgs.to(device, non_blocking=True)
            wea = wea.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _ = model(imgs, wea, mc_dropout=False)
            p = logits.argmax(1)
            y_true.extend(y.cpu().tolist());
            y_pred.extend(p.cpu().tolist())
    plot_confusion(y_true, y_pred, LABELS, os.path.join(FIG_DIR, "图_验证集混淆矩阵.png"))

    return best_path_ref[0], weather_means


# ===================== 任务三推理 =====================
def parse_attachment3() -> Dict[str, Dict[int, str]]:
    fruits = {}
    if not os.path.isdir(PRED_DIR):
        print(f"⚠️ 未找到附件3目录：{PRED_DIR}")
        return fruits

    subdirs = sorted([d for d in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, d))])
    for fd in subdirs:
        fdir = os.path.join(PRED_DIR, fd)
        weeks = {}
        for img in os.listdir(fdir):
            if not img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            m = re.match(r"week_(\d+)\.", img)
            if not m:
                continue
            wn = int(m.group(1))
            weeks[wn] = os.path.join(fdir, img)
        if weeks:
            fruits[fd] = weeks
    return fruits


def build_input_seq_for_future(fruit_weeks: Dict[int, str], target_week_num: int, seq_len: int = 4):
    needed = []
    cur = target_week_num - 1  # 从目标周的前一周开始
    while len(needed) < seq_len and cur >= 1:
        if cur in fruit_weeks:
            needed.append(fruit_weeks[cur])
        cur -= 1
    if not needed:
        return None
    needed = needed[::-1]
    while len(needed) < seq_len:
        needed = [needed[0]] + needed
    return needed[-seq_len:]


def find_week_key_by_number(weather_means: Dict[str, Dict[str, float]], n: int) -> str:
    for k in weather_means.keys():
        m = re.search(r"week_(\d+)", k)
        if m and int(m.group(1)) == n:
            return k
    cand = []
    for k in weather_means.keys():
        m = re.search(r"week_(\d+)", k)
        if m:
            cand.append((abs(int(m.group(1)) - n), k))
    if cand:
        cand.sort()
        return cand[0][1]
    return ""


def task3_predict(best_ckpt_path: str, weather_means: Dict[str, Dict[str, float]]):
    print("== 任务三：未来四周预测 ==")

    model = MultiModalDiseasePredictor(num_classes=len(LABELS)).to(device)
    if os.path.exists(best_ckpt_path):
        sd = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
    else:
        print(f"警告: 检查点文件不存在: {best_ckpt_path}")
        return

    model.eval()

    fruits = parse_attachment3()
    if not fruits:
        print("⚠️ 附件3未发现任何果实子目录。")
        return
    print(f"检测到果实数量：{len(fruits)}")

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    rows = []
    for fid, wkmap in fruits.items():
        for tgt in [11, 12, 13, 14]:
            img_list = build_input_seq_for_future(wkmap, tgt, seq_len=SEQ_LEN)
            if img_list is None:
                print(f"警告: 无法为果实 {fid} 周 {tgt} 构建输入序列")
                continue

            imgs = []
            for p in img_list:
                try:
                    im = Image.open(p).convert("RGB")
                except:
                    im = Image.new("RGB", (224, 224), (0, 0, 0))
                imgs.append(tf(im))
            imgs = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, L, 3, 224, 224)
            imgs = imgs.to(device, non_blocking=True)

            wk_key = find_week_key_by_number(weather_means, tgt)
            wv = [weather_means.get(wk_key, {}).get(c, 0.0) for c in WEATHER_COLS_PREF]
            wea = torch.tensor([wv] * SEQ_LEN, dtype=torch.float32).unsqueeze(0)
            wea = wea.to(device, non_blocking=True)

            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                probs = []
                for _ in range(MC_PASSES):
                    logits, _ = model(imgs, wea, mc_dropout=True)
                    # 检查logits是否有效
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"警告: 果实 {fid} 周 {tgt} 的预测包含NaN或无穷大")
                        prob = torch.ones(len(LABELS)) / len(LABELS)  # 使用均匀分布作为fallback
                    else:
                        prob = F.softmax(logits, dim=1).cpu().numpy()[0]
                    probs.append(prob)
                probs = np.stack(probs, axis=0)
                mean_prob = probs.mean(axis=0)
                pred_idx = int(mean_prob.argmax())
                pred_phase = LABELS[pred_idx]
                entropy = float(-np.sum(mean_prob * np.log(mean_prob + 1e-9)))

            rows.append({
                "果实编号": fid,
                "周数": f"week_{tgt:02d}",
                "病害阶段": pred_phase,
                "p_健康期": float(mean_prob[PHASE2IDX["健康期"]]),
                "p_初发期": float(mean_prob[PHASE2IDX["初发期"]]),
                "p_发病期": float(mean_prob[PHASE2IDX["发病期"]]),
                "预测熵": entropy
            })

    df = pd.DataFrame(rows)
    for lbl in LABELS:
        c = f"p_{lbl}"
        if c not in df.columns:
            df[c] = 0.0

    if not df.empty:
        df["w"] = df["周数"].str.extract(r"week_(\d+)").astype(int)
        df = df.sort_values(["果实编号", "w"]).drop(columns=["w"])

    out_xlsx_primary = os.path.join(TASK3_DIR, "result3.xlsx")
    try:
        ensure_dir(TASK3_DIR)
        df.to_excel(out_xlsx_primary, index=False)
        print(f"✅ 已保存任务三结果表格：{out_xlsx_primary}（{len(df)} 行）")
        img_dir_use = TASK3_DIR
    except Exception as e:
        out_xlsx_fb = os.path.join(TASK3_DIR_FB, "result3.xlsx")
        ensure_dir(TASK3_DIR_FB)
        df.to_excel(out_xlsx_fb, index=False)
        print(f"✅ 已保存任务三结果表格（回退路径）：{out_xlsx_fb}（{len(df)} 行）；主路径失败：{e}")
        img_dir_use = TASK3_DIR_FB
    if not df.empty:
        vc = df["病害阶段"].value_counts().reindex(LABELS, fill_value=0)
        plt.figure(figsize=(6, 5))
        ax = vc.plot(kind="bar")
        for i, v in enumerate(vc.values):
            ax.text(i, v + 0.3, str(int(v)), ha="center")
        plt.ylabel("数量");
        plt.title("任务三预测阶段分布（week_11~14）")
        plt.tight_layout()
        p1 = os.path.join(img_dir_use, "图_任务三_预测阶段分布.png")
        plt.savefig(p1, dpi=300);
        plt.close()
        print(f"✅ 保存图：{p1}")

        # 图2：按周堆叠分布
        if df["周数"].nunique() > 1:
            ct = pd.crosstab(df["周数"], df["病害阶段"]).reindex(columns=LABELS, fill_value=0).astype(int).sort_index()
            ax = ct.plot(kind="bar", stacked=True, figsize=(8, 5))
            plt.ylabel("数量");
            plt.title("按周的病害阶段堆叠分布（week_11~14）")
            plt.legend(title="病害阶段")
            plt.tight_layout()
            p2 = os.path.join(img_dir_use, "图_任务三_按周堆叠分布.png")
            plt.savefig(p2, dpi=300);
            plt.close()
            print(f"✅ 保存图：{p2}")


def main():
    init_dirs()

    best_ckpt_path, weather_means = run_train()

    fixed_ckpt = os.path.join(OUT_DIR, "best_model.pth")
    if os.path.exists(fixed_ckpt):
        best_ckpt_path = fixed_ckpt

    task3_predict(best_ckpt_path, weather_means)


if __name__ == "__main__":
    main()