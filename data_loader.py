import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 只用于报表关联，绝不进 prompt
ID_FIELDS = ["listingId", "vin", "url"]

# 从 CSV 直接取、允许进 prompt 的字段
RAW_PROMPT_FIELDS = [
    "year", "make", "model", "trim", "bodyStyle", "fuelType",
    "price", "msrp", "mileage", "stockType", "cpoIndicator",
]

# 同款对标计算出来的字段（同样允许进 prompt）
COMP_FIELDS = ["comp_median", "comp_n", "comp_mileage_median", "delta_pct"]

# 唯一允许进 prompt 的字段（白名单）
PROMPT_FIELDS = RAW_PROMPT_FIELDS + COMP_FIELDS

REQUIRED = ["listingId", "year", "make", "model", "price"]

# ---- 可调参数 ----
YEAR_RANGE = (2005, 2025)          # 排除古董/改装车和几乎全新的车
PRICE_RANGE = (3_000, 150_000)     # 排除极端价格
COMP_YEAR_WINDOW = 1               # 同款 = 同 make/model，年份 ±1
MIN_COMPS = 3                      # 同款样本量下限
MIN_DISCOUNT_PCT = 10.0            # 至少比同款中位价低 10%
# 里程护栏：折价若只是因为里程更高，不算"低估"。设为 None 可关闭。
MAX_MILEAGE_RATIO = 1.25           # 里程 <= 同款中位里程 × 1.25 + MILEAGE_SLACK
MILEAGE_SLACK = 5_000


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """列裁剪、过滤新车、去脏数据、修类型、范围过滤。"""
    # 1. 只保留需要的列（列缺失时尽早报错，而不是跑到一半 KeyError）
    keep = ID_FIELDS + RAW_PROMPT_FIELDS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")
    df = df[keep]

    # 2. 过滤新车
    n = len(df)
    df = df[df["stockType"].fillna("").str.lower() != "new"]
    logger.info("过滤新车: %d -> %d", n, len(df))

    # 3. 关键字段缺失的行直接丢弃，并去重
    n = len(df)
    df = df.dropna(subset=REQUIRED).drop_duplicates(subset="listingId")
    logger.info("丢弃缺失/重复: %d -> %d", n, len(df))

    # 4. msrp == 0 是占位值，不是真实价格，转成缺失值
    df = df.assign(msrp=df["msrp"].where(df["msrp"] > 0))

    # 5. 修正类型：有缺失值时 pandas 会把整数列悄悄变成 float
    df = df.astype({"year": "int64", "price": "int64",
                    "msrp": "Int64", "mileage": "Int64"})

    # 6. 年份和价格范围
    n = len(df)
    df = df[df["year"].between(*YEAR_RANGE) & df["price"].between(*PRICE_RANGE)]
    logger.info("范围过滤(year=%s, price=%s): %d -> %d",
                YEAR_RANGE, PRICE_RANGE, n, len(df))
    return df.reset_index(drop=True)


def _add_comps(df: pd.DataFrame) -> pd.DataFrame:
    """为每辆车计算同款对标：同 make/model、年份 ±COMP_YEAR_WINDOW，不含自身。"""
    size = len(df)
    comp_n = np.zeros(size, dtype=int)
    comp_price = np.full(size, np.nan)
    comp_miles = np.full(size, np.nan)

    keys = [df["make"].str.strip().str.lower(), df["model"].str.strip().str.lower()]
    for _, g in df.groupby(keys):
        idx = g.index.to_numpy()
        year = g["year"].to_numpy()
        price = g["price"].to_numpy(dtype=float)
        miles = g["mileage"].to_numpy(dtype=float, na_value=np.nan)
        for k, i in enumerate(idx):
            mask = np.abs(year - year[k]) <= COMP_YEAR_WINDOW
            mask[k] = False                       # 排除自身
            comp_n[i] = mask.sum()
            if comp_n[i]:
                comp_price[i] = np.median(price[mask])
                m = miles[mask]
                if np.isfinite(m).any():
                    comp_miles[i] = np.nanmedian(m)

    return df.assign(
        comp_n=comp_n,
        comp_median=pd.Series(np.round(comp_price)).astype("Int64"),
        comp_mileage_median=pd.Series(np.round(comp_miles)).astype("Int64"),
        delta_pct=(df["price"] / comp_price - 1) * 100,   # 筛选用原始精度，输出时再取整
    )


def load_cars(path: str, limit: int | None = None) -> list[dict]:
    """FSM 的唯一入口：读取 -> 清洗 -> 同款对标 -> 筛出高潜车源 -> 白名单裁剪。

    返回的每个 dict 含 ID_FIELDS + PROMPT_FIELDS，按折价幅度从大到小排序；
    limit 取的是折价最深的前 N 辆。
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")   # 去掉 BOM
    df = _clean(raw)
    pool = len(df)

    df = _add_comps(df)
    comparable = int((df["comp_n"] >= MIN_COMPS).sum())

    # 同款样本量足够，且至少比同款中位价低 MIN_DISCOUNT_PCT
    df = df[(df["comp_n"] >= MIN_COMPS) & (df["delta_pct"] <= -MIN_DISCOUNT_PCT)]
    discounted = len(df)

    # 里程护栏：剔除"便宜只是因为里程高"的车；里程缺失无法验证，同样剔除
    if MAX_MILEAGE_RATIO is not None:
        limit_miles = df["comp_mileage_median"] * MAX_MILEAGE_RATIO + MILEAGE_SLACK
        df = df[(df["mileage"] <= limit_miles).fillna(False).astype(bool)]

    df = df.sort_values("delta_pct")                # 折价最深的排最前
    candidates = len(df)
    if limit:
        df = df.head(limit)

    logger.info("对标结果: 清洗后 %d 辆 | 样本量>=%d 可比 %d 辆 | 折价>=%.0f%% %d 辆 "
                "| 里程护栏后 %d 辆 | 本次取 %d 辆",
                pool, MIN_COMPS, comparable, MIN_DISCOUNT_PCT, discounted,
                candidates, len(df))
    if df.empty:
        logger.warning("没有符合条件的高潜车源，请检查筛选参数")

    df = df.assign(delta_pct=df["delta_pct"].round(1))

    # NaN / pd.NA -> None，避免 "nan" 字符串进入 prompt
    df = df[ID_FIELDS + PROMPT_FIELDS].astype(object)
    df = df.where(df.notna(), None)
    return df.to_dict(orient="records")


def build_prompt_payload(car: dict) -> str:
    """发给模型前的最后一道闸：白名单 + 去空值 + 紧凑 JSON"""
    payload = {k: car[k] for k in PROMPT_FIELDS if car.get(k) is not None}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))