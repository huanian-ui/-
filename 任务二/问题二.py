# -*- coding: utf-8 -*-
"""
Task 2 — 病害阶段与气象因子相关性分析（v2 论文级增强版）
Author: ChatGPT

新增/改进（相对上一版）：
- 中文字体自动适配（微软雅黑/黑体 → Arial/DejaVu 回退），修复标题乱码
- 与阶段的 Spearman ρ 热力图：显著优先（FDR≤0.05）、自适应对称色轴、格内数值+显著性星号
- 统一科研配色：['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8', '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
- 新增「有序Logit OR 森林图」（赔率比+95%CI，log 尺度）
- 保留/扩展：描述性表格、单变量检验、Logit / LightGBM（分组CV）、PDP、（可选）SHAP

依赖：
pip install pandas numpy scipy statsmodels scikit-learn lightgbm matplotlib openpyxl
# 可选（若需 SHAP 图）：
pip install shap
"""

import os
import re
import glob
import math
import warnings
from datetime import datetime
from typing import Tuple, List

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import partial_dependence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ============== 可选依赖 ==============
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ============== 配置区域（按需修改） ==============
LABELS_FILE = r"E:\PythonProject\result1.xlsx"       # 任务一结果（中文/英文表头均可）
WEATHER_DIR = r"F:\本科组数据\附件2"                      # 附件二目录（week_XX_*.xlsx）
OUTPUT_DIR  = r"E:\PythonProject\task2_outputs"      # 输出目录

# 论文配色（按你的要求）
PALETTE = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8', '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']

# 阈值
RH_HIGH   = 80.0      # %
TEMP_HIGH = 30.0      # ℃
TEMP_LOW  = 15.0      # ℃
WIND_HIGH = 8.0       # m/s

RANDOM_STATE = 42


# ============== 画图风格 ==============
def set_paper_style():
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.color": "#aaaaaa",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def color_at(i):
    return PALETTE[i % len(PALETTE)]


def diverging_cmap(neg="#83A1E7", zero="#FFFFFF", pos="#90D8A6"):
    """基于给定配色创建发散色图（负-白-正）"""
    return LinearSegmentedColormap.from_list("paper_div", [neg, zero, pos], N=256)


# ============== 工具函数 ==============
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "tables"), exist_ok=True)


def parse_week_from_filename(fname: str) -> Tuple[int, datetime]:
    base = os.path.basename(fname)
    m_week = re.search(r"week[_\- ]?(\d{1,2})", base, flags=re.IGNORECASE)
    week_int = int(m_week.group(1)) if m_week else None
    m_date = re.search(r"\((\d{1,2})[_\-](\d{1,2})[_\-](\d{4})\)", base)
    if not m_date:
        m_date = re.search(r"[_\- ](\d{1,2})[_\-](\d{1,2})[_\-](\d{4})", base)
    d0 = None
    if m_date:
        try:
            d, m, y = int(m_date.group(1)), int(m_date.group(2)), int(m_date.group(3))
            d0 = datetime(y, m, d)
        except Exception:
            d0 = None
    return week_int, d0


def std_colname_map(cols: List[str]) -> dict:
    cmap = {}
    for c in cols:
        lc = str(c).strip().lower()
        aliases = {
            "image": ["image", "img", "filename", "file", "photo", "图像", "图片", "文件名", "图片名", "果实编号", "果实id"],
            "date": ["date", "日期"],
            "time": ["time", "时间"],
            "temp": ["temp", "temperature", "气温", "温度"],
            "dewpt": ["dewpt", "dew_point", "露点"],
            "rh": ["rh", "humidity", "相对湿度"],
            "precip_rate": ["precip_rate", "precip", "rain", "降水", "rain_rate"],
            "solar_rad": ["solar_rad", "solar", "辐照", "solar_radiation"],
            "ghi": ["ghi"],
            "dhi": ["dhi"],
            "dni": ["dni"],
            "pres": ["pres", "pressure", "气压"],
            "wind_spd": ["wind_spd", "wind", "wind_speed", "风速"],
            "weather_desc": ["weather_desc", "weather", "天气"],
            "vis": ["vis", "visibility", "能见度"],
            "week": ["week", "周次", "周数", "周编号"],
        }
        mapped = None
        for k, vs in aliases.items():
            if lc in vs:
                mapped = k
                break
        cmap[c] = mapped if mapped else lc
    return cmap


def normalize_image_name(x):
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", "", s)  # 去扩展名
    return s


def compute_vpd(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    vpd = es * (1 - rh / 100.0)
    return vpd


def fdr_bh(pvals: np.ndarray, alpha=0.05):
    p = np.asarray(pvals)
    n = len(p)
    idx = np.argsort(p)
    ranked = np.arange(1, n + 1)
    thresh = alpha * ranked / n
    passed = p[idx] <= thresh
    return passed[np.argsort(idx)], thresh[-1] if n else alpha


def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if min_rating is None:
        min_rating = int(min(y_true.min(), y_pred.min()))
    if max_rating is None:
        max_rating = int(max(y_true.max(), y_pred.max()))
    num_ratings = int(max_rating - min_rating + 1)
    O = np.zeros((num_ratings, num_ratings), dtype=float)
    for a, p in zip(y_true, y_pred):
        O[a - min_rating, p - min_rating] += 1
    act_hist = np.bincount(y_true - min_rating, minlength=num_ratings).astype(float)
    pred_hist = np.bincount(y_pred - min_rating, minlength=num_ratings).astype(float)
    E = np.outer(act_hist, pred_hist) / max(1.0, len(y_true))
    W = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    num = (W * O).sum()
    den = (W * E).sum() + 1e-12
    return 1.0 - num / den


# ============== 数据读取 ==============
def load_labels(labels_file: str) -> pd.DataFrame:
    """兼容中文表头：周数/果实编号/果实阶段；并完成标准化。"""
    df = pd.read_excel(labels_file, engine="openpyxl")
    orig = list(df.columns)
    colmap = {}
    for c in orig:
        lc = str(c).strip().lower()
        if lc in ["image", "img", "filename", "file", "photo", "图像", "图片", "文件名", "图片名", "果实编号", "果实id"]:
            colmap[c] = "image_or_fruit"
        elif lc in ["week", "周次", "周数", "周编号"]:
            colmap[c] = "week"
        elif lc in ["stage", "label", "阶段", "果实阶段", "病害阶段"]:
            colmap[c] = "stage"
    df = df.rename(columns=colmap)

    # 从“果实编号”生成 image（对齐附件2的 1_1.jpg）
    if "image" not in df.columns and "image_or_fruit" in df.columns:
        df["image"] = df["image_or_fruit"].astype(str).map(normalize_image_name)
    if "image" not in df.columns:
        raise ValueError("任务一结果缺少 `image`/`果实编号` 列，请至少提供其中之一。")

    # 解析 tree_id / fruit_id
    m = df["image"].astype(str).str.extract(r"(?P<tree_id>\d+)[_\-](?P<fruit_id>\d+)")
    if "tree_id" not in df.columns:
        df["tree_id"] = pd.to_numeric(m["tree_id"], errors="coerce")
    if "fruit_id" not in df.columns:
        df["fruit_id"] = pd.to_numeric(m["fruit_id"], errors="coerce")

    # 解析 week（允许 week_01 / 1 / '01' / 'week01' 等）
    if "week" in df.columns:
        df["week"] = df["week"].astype(str).str.extract(r"(\d{1,2})")[0]
        df["week"] = pd.to_numeric(df["week"], errors="coerce")

    # 阶段映射
    if "stage" not in df.columns:
        raise ValueError("任务一结果缺少 `stage`/`果实阶段` 列。")
    stage_map = {
        "健康期": 0, "健康": 0, "0": 0, 0: 0,
        "初发期": 1, "初发": 1, "1": 1, 1: 1,
        "发病期": 2, "发病": 2, "2": 2, 2: 2,
    }

    def map_stage(x):
        s = str(x).strip()
        return stage_map.get(s, stage_map.get(s.replace(" ", ""), pd.to_numeric(x, errors="coerce")))

    df["stage"] = df["stage"].map(map_stage)
    if df["stage"].isna().any():
        bad = df[df["stage"].isna()]
        raise ValueError(f"发现无法识别的阶段标签：{bad.iloc[:5].to_dict(orient='records')}")

    keep = ["image", "tree_id", "fruit_id", "week", "stage"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]


def load_weather_dir(weather_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(weather_dir, "week*.xls*")))
    if not files:
        raise FileNotFoundError(f"未在 {weather_dir} 找到任何 week*.xlsx 文件")
    frames = []
    for f in files:
        try:
            dfw = pd.read_excel(f, engine="openpyxl")
        except Exception:
            dfw = pd.read_excel(f)
        dfw = dfw.rename(columns=std_colname_map(dfw.columns))
        week_int, _ = parse_week_from_filename(f)
        if week_int is not None:
            dfw["week"] = week_int
        if "image" in dfw.columns:
            dfw["image"] = dfw["image"].map(normalize_image_name)

        # 解析日期/时间 → timestamp（可为空）
        if "date" in dfw.columns:
            dfw["date_str"] = dfw["date"].astype(str).str.replace(".", ":", regex=False).str.replace("-",
                                                                                                     ":", regex=False).str.strip()

            def _parse_date(s):
                for fmt in ["%Y:%m:%d", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(str(s), fmt)
                    except Exception:
                        pass
                return pd.NaT

            dfw["date_dt"] = dfw["date_str"].map(_parse_date)
        else:
            dfw["date_dt"] = pd.NaT
        if "time" in dfw.columns:
            def _parse_time(s):
                s = str(s).strip()
                for fmt in ["%H:%M:%S", "%H:%M"]:
                    try:
                        return datetime.strptime(s, fmt).time()
                    except Exception:
                        pass
                return None

            dfw["time_tm"] = dfw["time"].map(_parse_time)
        else:
            dfw["time_tm"] = None

        def _combine_ts(r):
            d = r.get("date_dt", pd.NaT);
            t = r.get("time_tm", None)
            if pd.isna(d) or t is None:
                return pd.NaT
            return datetime.combine(d, t)

        dfw["timestamp"] = dfw.apply(_combine_ts, axis=1)

        # 数值列
        for c in ["temp", "dewpt", "rh", "precip_rate", "solar_rad", "ghi", "dhi", "dni", "pres", "wind_spd", "vis"]:
            if c in dfw.columns:
                dfw[c] = pd.to_numeric(dfw[c], errors="coerce")
        if "temp" in dfw.columns and "rh" in dfw.columns:
            dfw["vpd"] = compute_vpd(dfw["temp"], dfw["rh"])
        frames.append(dfw)

    allw = pd.concat(frames, ignore_index=True)
    if "week" not in allw.columns or allw["week"].isna().all():
        if allw["date_dt"].notna().any():
            allw["week"] = pd.factorize(allw["date_dt"].dt.to_period("W"))[0] + 1
        else:
            allw["week"] = 1
    allw["week"] = pd.to_numeric(allw["week"], errors="coerce")
    return allw


# ============== 特征工程 ==============
def weekly_aggregate_features(dfw: pd.DataFrame) -> pd.DataFrame:
    agg = dfw.copy()
    agg["is_rain"] = (agg.get("precip_rate", 0).fillna(0) > 0).astype(int)
    agg["is_high_rh"] = (agg.get("rh", 0).fillna(0) >= RH_HIGH).astype(int)
    agg["is_high_temp"] = (agg.get("temp", 0).fillna(0) >= TEMP_HIGH).astype(int)
    agg["is_low_temp"] = (agg.get("temp", 0).fillna(0) <= TEMP_LOW).astype(int)
    agg["is_windy"] = (agg.get("wind_spd", 0).fillna(0) >= WIND_HIGH).astype(int)

    gps = []
    for wk, g in agg.groupby("week"):
        d = {"week": wk}
        for c in ["temp", "dewpt", "rh", "vpd", "ghi", "dhi", "dni", "solar_rad", "pres", "wind_spd", "vis"]:
            if c in g.columns:
                d[f"{c}_mean"] = g[c].mean()
                d[f"{c}_min"] = g[c].min()
                d[f"{c}_max"] = g[c].max()
                d[f"{c}_std"] = g[c].std()
        if "precip_rate" in g.columns:
            d["precip_sum"] = g["precip_rate"].fillna(0).sum()
            d["rainy_hours"] = g["is_rain"].sum()
        if "rh" in g.columns:
            d["high_rh_hours"] = g["is_high_rh"].sum()
        if "temp" in g.columns:
            d["high_temp_hours"] = g["is_high_temp"].sum()
            d["low_temp_hours"] = g["is_low_temp"].sum()
            d["temp_range"] = g["temp"].max() - g["temp"].min()
        if "wind_spd" in g.columns:
            d["windy_hours"] = g["is_windy"].sum()
        gps.append(d)
    W = pd.DataFrame(gps).sort_values("week").reset_index(drop=True)
    return W


def add_lag_windows(W: pd.DataFrame, lags=(0, 1, 2)) -> pd.DataFrame:
    W = W.set_index("week").sort_index()
    outs = []
    for lag in lags:
        df_l = W.shift(lag).copy()
        df_l.columns = [f"{c}_" + ("w0" if lag == 0 else f"l{lag}") for c in df_l.columns]
        df_l["week"] = W.index
        outs.append(df_l.reset_index(drop=True))
    M = outs[0]
    for i in range(1, len(outs)):
        M = M.merge(outs[i], on="week", how="left")
    return M


def build_model_dataset(labels: pd.DataFrame, dfw: pd.DataFrame) -> pd.DataFrame:
    # 以 image 精确对齐当前时点微气候
    key_cols = ["image", "week", "temp", "dewpt", "rh", "vpd", "precip_rate", "solar_rad", "ghi", "dhi", "dni", "pres",
                "wind_spd", "weather_desc", "vis"]
    curw = dfw[[c for c in key_cols if c in dfw.columns]].copy()
    if "image" in curw.columns:
        curw = curw.drop_duplicates(subset=["image"], keep="last")
    merged = labels.merge(curw, on="image", how="left", suffixes=("", "_cur"))
    if "week" not in merged.columns or merged["week"].isna().any():
        if "week_cur" in merged.columns:
            merged["week"] = merged["week"].fillna(merged["week_cur"])
            merged = merged.drop(columns=[c for c in ["week_cur"] if c in merged.columns])

    # 周级暴露窗口：当周/前1/前2
    W = weekly_aggregate_features(dfw)
    Wlags = add_lag_windows(W, lags=(0, 1, 2))
    data = merged.merge(Wlags, on="week", how="left")

    if "weather_desc" in data.columns:
        # 标准化字符串
        data["weather_desc"] = (
            data["weather_desc"].astype(str)
            .str.lower()
            .str.strip()
            .replace(["nan", "none", "missing", "na"], np.nan)
        )

        # 选前6个非空类别
        topk = (
            data["weather_desc"].dropna().value_counts().nlargest(6).index.tolist()
        )

        # 已创建的列名集合，防止重复
        created_cols = set()

        for cat in topk:
            safe_cat = re.sub(r"[^a-z0-9_]+", "_", str(cat))  # 清理特殊符号
            new_col = f"wdesc_{safe_cat}"
            if new_col not in created_cols and new_col not in data.columns:
                data[new_col] = (data["weather_desc"] == cat).astype(int)
                created_cols.add(new_col)

    data["stage"] = pd.to_numeric(data["stage"], errors="coerce")
    data = data.dropna(subset=["stage"])
    data["stage"] = data["stage"].astype(int)
    data["is_disease"] = (data["stage"] >= 1).astype(int)
    return data


# ============== 统计检验 ==============
def univariate_tests(df: pd.DataFrame, target_col="stage") -> pd.DataFrame:
    num_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype.kind in "fcbi" and df[c].nunique() > 5:
            num_cols.append(c)

    results = []
    y = df[target_col].values
    for c in num_cols:
        x = df[c].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 10:
            continue
        rho, pval = stats.spearmanr(x[mask], y[mask])
        results.append({"feature": c, "spearman_rho": float(rho), "p_value": float(pval)})
    res_df = pd.DataFrame(results).sort_values("p_value")
    if not res_df.empty:
        sig_mask, _ = fdr_bh(res_df["p_value"].values, alpha=0.05)
        res_df["fdr_significant"] = sig_mask

    # 互信息（非线性相关性）
    if num_cols:
        X = df[num_cols].copy().replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        try:
            mi = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE, discrete_features=False)
            res_df = res_df.merge(pd.DataFrame({"feature": num_cols, "mutual_info": mi}),
                                  on="feature", how="left")
        except Exception as e:
            print(f"[WARN] 互信息计算失败：{e}")
    return res_df


# ============== 建模训练 ==============
def prepare_Xy(df: pd.DataFrame, target_col="stage"):
    y = df[target_col].astype(int).values

    # 严格排除不建模列
    drop_cols = {"image", "weather_desc", target_col}

    # 先取所有数值列
    base_numeric = [c for c in df.columns
                    if c not in drop_cols and df[c].dtype.kind in "fcbi"]

    # 拆分出 one-hot 列，避免和数值列重复
    ohe_cols = [c for c in base_numeric if c.startswith("wdesc_")]
    num_cols = [c for c in base_numeric if not c.startswith("wdesc_")]

    # 合并并“按顺序去重”
    feat_cols = list(dict.fromkeys(num_cols + ohe_cols))

    X = df[feat_cols].copy()
    # 保险：再次去重（以防其它环节引入）
    X = X.loc[:, ~X.columns.duplicated(keep="first")]

    return X, y, feat_cols


def fit_ordered_logit(X, y):
    # 有序 Logit 的截距由 cutpoints 表达，不能再显式加常数
    X = X.copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 传入 DataFrame 以保留列名，便于后续 OR 森林图标注
    try:
        model = OrderedModel(y, X, distr='logit')
        res = model.fit(method='bfgs', disp=False)
        return res
    except Exception as e:
        print(f"[WARN] 有序Logit拟合失败，将回退多项Logit：{e}")
        return None



def fit_multinomial_logit_cv(X, y, groups=None, n_splits=5):
    # 防重
    if not X.columns.is_unique:
        X = X.loc[:, ~X.columns.duplicated(keep="first")]
        print("[INFO] 多项Logit：检测到重复特征名，已自动去重。")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(max_iter=3000, multi_class="multinomial",
                                   solver="lbfgs", random_state=RANDOM_STATE))
    ])
    if groups is not None and len(np.unique(groups)) >= n_splits:
        cv = GroupKFold(n_splits=n_splits); splits = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = cv.split(X, y)

    preds, trues = [], []
    for tr, te in splits:
        pipe.fit(X.iloc[tr], y[tr])
        yp = pipe.predict(X.iloc[te])
        preds.extend(yp); trues.extend(y[te])
    preds = np.asarray(preds); trues = np.asarray(trues)
    metrics = {
        "accuracy": float(accuracy_score(trues, preds)),
        "macro_f1": float(f1_score(trues, preds, average="macro")),
        "qwk": float(quadratic_weighted_kappa(trues, preds)),
    }
    pipe.fit(X, y)
    return pipe, metrics



def fit_lightgbm_cv_and_final(X, y, groups=None, n_splits=5):
    if not HAS_LGB:
        print("[WARN] 未安装 lightgbm，跳过该模型。")
        return None, {}, None

    try:
        # 列名必须唯一
        if not X.columns.is_unique:
            X = X.loc[:, ~X.columns.duplicated(keep="first")]
            print("[INFO] 检测到重复特征名，已自动去重。")

        clf = LGBMClassifier(
            objective="multiclass",
            num_class=len(np.unique(y)),
            learning_rate=0.05,
            n_estimators=500,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE
        )

        # CV 划分
        if groups is not None and len(np.unique(groups)) >= n_splits:
            cv = GroupKFold(n_splits=n_splits)
            splits = cv.split(X, y, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            splits = cv.split(X, y)

        preds, trues = [], []
        for tr, te in splits:
            clf.fit(X.iloc[tr], y[tr])
            yp = clf.predict(X.iloc[te])
            preds.extend(yp); trues.extend(y[te])

        preds = np.asarray(preds); trues = np.asarray(trues)
        metrics = {
            "accuracy": float(accuracy_score(trues, preds)),
            "macro_f1": float(f1_score(trues, preds, average="macro")),
            "qwk": float(quadratic_weighted_kappa(trues, preds)),
        }

        # 全量拟合用于重要性、PDP、SHAP
        clf.fit(X, y)
        importances = clf.feature_importances_

        return clf, metrics, importances

    except Exception as e:
        # 兜底返回，避免 NoneType 解包
        print(f"[ERROR] LightGBM 训练失败：{e}")
        return None, {}, None




# ============== 画图函数（论文风格） ==============
def save_stage_distribution(df, path_png):
    set_paper_style()
    counts = df["stage"].value_counts().sort_index()
    labels = ["健康期(0)", "初发期(1)", "发病期(2)"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(len(counts)), counts.values, color=[color_at(i) for i in range(len(counts))])
    plt.xticks(range(len(counts)), labels)
    plt.ylabel("样本数量")
    plt.title("病害阶段分布")
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=10)
    plt.grid(axis="y")
    plt.tight_layout(); plt.savefig(path_png); plt.close()


def save_weather_histograms(df, feat_cols, path_png):
    set_paper_style()
    feats = [c for c in feat_cols if c in df.columns]
    if len(feats) == 0:
        return
    n = min(len(feats), 6)  # 最多画6个
    cols = 3; rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 4.2, rows * 3.6))
    for i in range(n):
        c = feats[i]
        ax = plt.subplot(rows, cols, i + 1)
        x = df[c].dropna().values
        ax.hist(x, bins=30, color=color_at(i), alpha=0.85, edgecolor="#333333", linewidth=0.6)
        ax.set_title(c)
        ax.grid(True)
    plt.suptitle("关键气象特征分布", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path_png); plt.close()


def save_boxplots_by_stage(df, feat_cols, path_png):
    set_paper_style()
    feats = [c for c in feat_cols if c in df.columns]
    if len(feats) == 0:
        return
    n = min(len(feats), 6)
    cols = 3; rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 4.5, rows * 4.0))
    stage_names = ["0-健康期", "1-初发期", "2-发病期"]
    for i in range(n):
        c = feats[i]
        ax = plt.subplot(rows, cols, i + 1)
        data0 = [df[df["stage"] == k][c].dropna().values for k in [0, 1, 2]]
        bp = ax.boxplot(data0, labels=stage_names, patch_artist=True)
        for j, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=color_at(j), alpha=0.75, edgecolor="#333333")
        for median in bp["medians"]:
            median.set(color="#333333", linewidth=1.2)
        ax.set_title(c); ax.grid(True, axis="y")
    plt.suptitle("按阶段的关键气象特征箱线图", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path_png); plt.close()


def p2stars(p):
    if p <= 1e-3: return "***"
    if p <= 1e-2: return "**"
    if p <= 5e-2: return "*"
    return ""


def save_spearman_heatmap(uni_df, path_png, top_k=20, exclude=("week",)):

    set_paper_style()
    df = uni_df.dropna(subset=["spearman_rho"]).copy()
    if exclude:
        df = df[~df["feature"].isin(list(exclude))]
    if df.empty:
        return

    # 先取显著项
    if "fdr_significant" in df.columns and df["fdr_significant"].any():
        df = df[df["fdr_significant"]].copy()
    else:
        df["abs_rho"] = df["spearman_rho"].abs()
        df = df.sort_values("abs_rho", ascending=False).head(top_k)

    if "abs_rho" not in df.columns:
        df["abs_rho"] = df["spearman_rho"].abs()
    df = df.sort_values("abs_rho", ascending=False)

    rhos = df["spearman_rho"].values.reshape(1, -1)
    cmap = diverging_cmap(neg="#83A1E7", zero="#FFFFFF", pos="#90D8A6")

    vmax = max(0.2, np.nanmax(np.abs(rhos)))  # 至少 ±0.2
    vmin = -vmax

    plt.figure(figsize=(max(8, 0.5 * len(df)), 3.2))
    im = plt.imshow(rhos, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ")

    # 轴标签
    plt.xticks(range(len(df)), df["feature"].tolist(), rotation=55, ha="right")
    plt.yticks([0], ["stage"])
    plt.title("与阶段的 Spearman 相关性（显著优先 / 论文配色）")

    # 在格子内标注 ρ 和显著性
    pmap = {r["feature"]: r.get("p_value", np.nan) for _, r in uni_df.iterrows()}
    for j, feat in enumerate(df["feature"].tolist()):
        rho = df["spearman_rho"].iloc[j]
        p = pmap.get(feat, np.nan)
        stars = p2stars(p) if not np.isnan(p) else ""
        txt = f"{rho:.2f}{stars}"
        plt.text(j, 0, txt, ha="center", va="center", fontsize=10, color="#1f1f1f")

    # 边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#333333"); spine.set_linewidth(1.0)

    plt.tight_layout(); plt.savefig(path_png); plt.close()


def save_significant_features_bar(uni_df, path_png, max_n=25):
    """FDR显著的特征按 |rho| 排序绘制条形图"""
    set_paper_style()
    df = uni_df.copy()
    if "fdr_significant" in df.columns:
        df = df[df["fdr_significant"] == True]
    if df.empty:
        return
    df["abs_rho"] = df["spearman_rho"].abs()
    df = df.sort_values("abs_rho", ascending=False).head(max_n)
    plt.figure(figsize=(8, max(4.0, 0.35 * len(df))))
    colors = [color_at(i) for i in range(len(df))]
    plt.barh(range(len(df)), df["abs_rho"].values[::-1], color=colors[::-1], edgecolor="#333333", linewidth=0.6)
    plt.yticks(range(len(df)), df["feature"].values[::-1])
    plt.xlabel("|Spearman ρ|")
    plt.title("单变量显著特征（FDR≤0.05）")
    plt.tight_layout(); plt.savefig(path_png); plt.close()


def save_lightgbm_importance(importance, feat_names, path_png, top_n=30):
    set_paper_style()
    if importance is None:
        return
    idx = np.argsort(importance)[::-1]
    imp = np.asarray(importance)[idx]
    labs = np.asarray(feat_names)[idx]
    topn = min(len(imp), top_n)
    plt.figure(figsize=(8, max(6, 0.35 * topn)))
    colors = [color_at(i) for i in range(topn)]
    plt.barh(range(topn), imp[:topn][::-1], color=colors[::-1], edgecolor="#333333", linewidth=0.6)
    plt.yticks(range(topn), labs[:topn][::-1])
    plt.xlabel("Gain Importance"); plt.title("LightGBM 特征重要性（Top）")
    plt.tight_layout(); plt.savefig(path_png); plt.close()


def save_pdp_plots(clf, X, feat_names, path_png, top_idx, grid_points=20):
    """对 top_idx 列表中的特征绘制 PDP"""
    set_paper_style()
    if clf is None or len(top_idx) == 0:
        return
    n = min(4, len(top_idx))
    cols = 2; rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 5.0, rows * 4.0))
    for i in range(n):
        j = top_idx[i]
        ax = plt.subplot(rows, cols, i + 1)
        try:
            pdp = partial_dependence(clf, X, features=[j], kind="average", grid_resolution=grid_points)
            xs = pdp["values"][0]; ys = pdp["average"][0].mean(axis=0)  # 多分类取平均
            ax.plot(xs, ys, color=color_at(i), linewidth=2.2)
            ax.set_xlabel(feat_names[j]); ax.set_ylabel("Partial Dependence")
            ax.grid(True); ax.set_title(f"PDP: {feat_names[j]}")
        except Exception as e:
            ax.text(0.5, 0.5, f"PDP失败: {feat_names[j]}\n{e}", ha="center", va="center")
    plt.suptitle("前4特征的部分依赖曲线（LightGBM）", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path_png); plt.close()


def save_or_forest_plot(ord_res, path_png, top_n=30):
    """
    有序Logit OR 森林图（赔率比+95%CI）
    - 横坐标为 OR（对数刻度），竖线为 OR=1
    - 按 p 值由小到大取前 top_n 个
    """
    if ord_res is None:
        return
    set_paper_style()
    params = ord_res.params
    conf = ord_res.conf_int()
    rows = []
    for name, val in params.items():
        if name.startswith("cut"):  # 阈值参数跳过
            continue
        or_val = math.exp(val)
        ci_low = math.exp(conf.loc[name, 0])
        ci_high = math.exp(conf.loc[name, 1])
        p = float(ord_res.pvalues.get(name, np.nan))
        rows.append({"feature": name, "OR": or_val, "CI_low": ci_low, "CI_high": ci_high, "p": p})
    df = pd.DataFrame(rows).sort_values("p").head(top_n)
    if df.empty:
        return

    plt.figure(figsize=(8, max(4.5, 0.35 * len(df))))
    y = np.arange(len(df))
    # 置信区间线
    plt.hlines(y, df["CI_low"], df["CI_high"], color="#333333", linewidth=1.2)
    # 点估计
    plt.scatter(df["OR"], y, s=60, color=[color_at(i) for i in range(len(df))], zorder=3, edgecolors="#333333")
    # 竖线 OR=1
    plt.axvline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    plt.yticks(y, df["feature"])
    plt.xlabel("Odds Ratio (log scale)")
    plt.title("有序Logit 赔率比（OR）森林图")
    plt.xscale("log")
    # 在右侧标注 OR [CI] 与显著性
    for i, r in df.iterrows():
        stars = p2stars(r["p"])
        txt = f'{r["OR"]:.2f} [{r["CI_low"]:.2f}, {r["CI_high"]:.2f}] {stars}'
        plt.text(max(df["CI_high"]) * 1.05, y[i], txt, va="center", fontsize=10)
    # 适配边界
    xmin = min(df["CI_low"].min() * 0.9, 0.5)
    xmax = max(df["CI_high"].max() * 1.3, 1.5)
    plt.xlim(xmin, xmax)
    plt.tight_layout(); plt.savefig(path_png); plt.close()


# ============== 主流程 ==============
def main():
    warnings.filterwarnings("ignore")
    ensure_dirs(); set_paper_style()

    print("=== 路径配置 ===")
    print("LABELS_FILE:", LABELS_FILE)
    print("WEATHER_DIR:", WEATHER_DIR)
    print("OUTPUT_DIR :", OUTPUT_DIR)

    # 1) 读取数据
    print("\n[1/10] 读取任务一标签 ...")
    labels = load_labels(LABELS_FILE)
    print(f"标签样本数：{len(labels)}，列：{list(labels.columns)}")

    print("[2/10] 读取附件2气象 ...")
    dfw = load_weather_dir(WEATHER_DIR)
    print(f"气象记录数：{len(dfw)}；周次范围：{dfw['week'].min()} - {dfw['week'].max()}")

    # 2) 合并与表格导出
    print("[3/10] 构建建模数据（合并 image + 周级窗口） ...")
    data = build_model_dataset(labels, dfw)
    data_path = os.path.join(OUTPUT_DIR, "dataset_task2_merged.csv")
    data.to_csv(data_path, index=False, encoding="utf-8-sig")
    print("Saved:", data_path)

    # 3) 描述性表格
    print("[4/10] 生成描述性统计表 ...")
    tbl_dir = os.path.join(OUTPUT_DIR, "tables")
    stage_counts = data["stage"].value_counts().sort_index()
    stage_counts.to_csv(os.path.join(tbl_dir, "table_stage_counts.csv"), header=["count"], encoding="utf-8-sig")

    # 关键特征（优先当周窗口）
    key_feats = [c for c in [
        "temp_mean_w0", "rh_mean_w0", "vpd_mean_w0", "precip_sum_w0",
        "ghi_mean_w0", "wind_spd_mean_w0", "high_rh_hours_w0", "high_temp_hours_w0"
    ] if c in data.columns]
    if key_feats:
        g = data.groupby("stage")[key_feats].agg(["mean", "std"]).round(3)
        g.to_csv(os.path.join(tbl_dir, "table_feature_by_stage.csv"), encoding="utf-8-sig")

    # 4) 单变量检验
    print("[5/10] 单变量 Spearman/MI/FDR ...")
    uni = univariate_tests(data, target_col="stage")
    uni_path = os.path.join(OUTPUT_DIR, "univariate_results.csv")
    uni.to_csv(uni_path, index=False, encoding="utf-8-sig")
    print("Saved:", uni_path)

    # 5) 建模准备
    print("[6/10] 建模准备 ...")
    X, y, feat_cols = prepare_Xy(data, target_col="stage")
    if "tree_id" in data.columns and data["tree_id"].notna().any():
        groups = data["tree_id"].astype(str).values
    else:
        groups = data["image"].astype(str).values

    # 6) 有序Logit
    print("[7/10] 有序Logit 拟合 ...")
    ord_res = fit_ordered_logit(X[feat_cols], y)
    if ord_res is not None:
        with open(os.path.join(OUTPUT_DIR, "ordered_logit_summary.txt"), "w", encoding="utf-8") as f:
            f.write(str(ord_res.summary()))
        # 系数与 OR
        params = ord_res.params; conf = ord_res.conf_int()
        rows = []
        for name, val in params.items():
            if name.startswith("cut"): continue
            or_val = math.exp(val)
            ci_low = math.exp(conf.loc[name, 0]); ci_high = math.exp(conf.loc[name, 1])
            rows.append({"feature": name, "coef": float(val), "odds_ratio": float(or_val),
                         "ci_low": float(ci_low), "ci_high": float(ci_high),
                         "p_value": float(ord_res.pvalues.get(name, np.nan))})
        pd.DataFrame(rows).sort_values("p_value").to_csv(
            os.path.join(OUTPUT_DIR, "ordered_logit_coeffs.csv"),
            index=False, encoding="utf-8-sig"
        )

    # 7) 多项Logit（CV）
    print("[8/10] 多项Logit（分组CV） ...")
    mnl_model, mnl_metrics = fit_multinomial_logit_cv(X[feat_cols], y, groups=groups, n_splits=5)
    import json
    with open(os.path.join(OUTPUT_DIR, "multinomial_logit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(mnl_metrics, f, ensure_ascii=False, indent=2)

    # 8) LightGBM（CV + 最终模型）
    if HAS_LGB:
        print("[9/10] LightGBM 训练与解释 ...")
        gbm, lgb_metrics, importance = fit_lightgbm_cv_and_final(X[feat_cols], y, groups=groups, n_splits=5)
        with open(os.path.join(OUTPUT_DIR, "lightgbm_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(lgb_metrics, f, ensure_ascii=False, indent=2)
    else:
        gbm, importance = None, None

    # ============ 论文级可视化输出 ============
    fig_dir = os.path.join(OUTPUT_DIR, "figures")

    # 阶段分布
    save_stage_distribution(data, os.path.join(fig_dir, "fig_stage_distribution.png"))

    # 关键特征直方图（当周窗口优先）
    if not key_feats:
        candidates = [c for c in feat_cols if c.endswith("_w0")]
        key_feats = candidates[:6] if candidates else feat_cols[:6]
    save_weather_histograms(data, key_feats, os.path.join(fig_dir, "fig_weather_hist.png"))

    # 按阶段箱线图
    save_boxplots_by_stage(data, key_feats, os.path.join(fig_dir, "fig_boxplots_by_stage.png"))

    # Spearman 热力图（显著优先；若无显著项则 Top20）
    save_spearman_heatmap(uni, os.path.join(fig_dir, "fig_spearman_heatmap.png"), top_k=20, exclude=("week",))

    # 单变量显著条形图
    save_significant_features_bar(uni, os.path.join(fig_dir, "fig_significant_features.png"), max_n=25)

    # LightGBM 重要性 + PDP
    # LightGBM 重要性 + PDP
    if HAS_LGB and importance is not None:
        save_lightgbm_importance(importance, feat_cols, os.path.join(fig_dir, "fig_lightgbm_importance_beautified.png"),
                                 top_n=10)  # 这里改为10

        # 选前4特征做 PDP
        top_idx = list(np.argsort(importance)[::-1][:4])
        try:
            save_pdp_plots(gbm, X[feat_cols], feat_cols, os.path.join(fig_dir, "fig_pdp_top_features.png"), top_idx)
        except Exception as e:
            print("[WARN] PDP 生成失败：", e)

        # 可选：SHAP（若安装）
        if HAS_SHAP:
            try:
                set_paper_style()
                explainer = shap.TreeExplainer(gbm)
                shap_values = explainer.shap_values(X[feat_cols])
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, X[feat_cols], plot_type="bar",
                                  show=False, color=PALETTE[0])
                plt.title("SHAP 全局重要性（LightGBM）")
                plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "fig_shap_summary_bar.png")); plt.close()

                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, X[feat_cols], show=False)
                plt.title("SHAP 总结图（LightGBM）")
                plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "fig_shap_summary.png")); plt.close()
            except Exception as e:
                print("[WARN] SHAP 绘图失败：", e)

    # OR 森林图（仅当有序Logit成功）
    if ord_res is not None:
        save_or_forest_plot(ord_res, os.path.join(fig_dir, "fig_ordered_logit_or_forest.png"))

    # ============ 生成 Markdown 报告 ============
    report_path = os.path.join(OUTPUT_DIR, "report_task2.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 任务二：病害阶段与气象因子相关性分析（自动报告 v2）\n\n")
        f.write("## 数据与方法\n")
        f.write("- 以 `image` 精确对齐拍摄微气候，并叠加当周/前1/前2周周级暴露特征；\n")
        f.write("- 单变量：Spearman ρ + 互信息，BH-FDR 校正；\n")
        f.write("- 多变量：有序Logit（如可拟合）、多项Logit（分组CV）、LightGBM（分组CV）；\n\n")
        f.write("## 主要结果\n")
        if os.path.exists(os.path.join(tbl_dir, "table_stage_counts.csv")):
            f.write("- 阶段分布见 `tables/table_stage_counts.csv`。\n")
        if os.path.exists(os.path.join(OUTPUT_DIR, "ordered_logit_coeffs.csv")):
            f.write("- 有序Logit 系数/赔率比见 `ordered_logit_coeffs.csv`，森林图见 `figures/fig_ordered_logit_or_forest.png`。\n")
        if os.path.exists(os.path.join(OUTPUT_DIR, "multinomial_logit_metrics.json")):
            f.write("- 多项Logit 交叉验证指标见 `multinomial_logit_metrics.json`。\n")
        if HAS_LGB and os.path.exists(os.path.join(OUTPUT_DIR, "lightgbm_metrics.json")):
            f.write("- LightGBM 交叉验证指标见 `lightgbm_metrics.json`，重要性图见 `figures/fig_lightgbm_importance_beautified.png`。\n")
        f.write("\n## 图表清单（可直接用于论文）\n")
        figs = [
            "fig_stage_distribution.png",
            "fig_weather_hist.png",
            "fig_boxplots_by_stage.png",
            "fig_spearman_heatmap.png",
            "fig_significant_features.png",
            "fig_lightgbm_importance_beautified.png",
            "fig_pdp_top_features.png",
            "fig_ordered_logit_or_forest.png",
            "fig_shap_summary_bar.png",
            "fig_shap_summary.png",
        ]
        for g in figs:
            if os.path.exists(os.path.join(os.path.join(OUTPUT_DIR, "figures"), g)):
                f.write(f"- figures/{g}\n")
    print("\n✅ 已完成全部输出。关键目录：")
    print(" - 数据表：", os.path.join(OUTPUT_DIR, "tables"))
    print(" - 图形：  ", os.path.join(OUTPUT_DIR, "figures"))
    print(" - 报告：  ", report_path)


if __name__ == "__main__":
    main()
