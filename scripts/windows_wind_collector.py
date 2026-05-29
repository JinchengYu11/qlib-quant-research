"""【在 Windows 机器上跑】用 WindPy 拉 A 股财务数据 → 存 CSV.

前置:
  1. Windows 机器, 已装 Wind 终端 + WindPy
  2. 已登录 Wind 终端
  3. Python 3.6+, 装好 pandas

用法:
  python windows_wind_collector.py [<workers>]

输出:
  - 每只股一个 CSV: ./financials_wind/{code}.csv
  - 失败列表: ./financials_wind/_failed.txt

拉完后:
  - zip ./financials_wind 目录
  - 传到 Mac 的 ~/qlib量化研究/data_raw/financials_wind/
  - 联系 Claude 跑下一步

数据格式: 每行一个季度, 列 = 各财务指标 + statDate + pubDate
"""
import os, time, sys, json
from pathlib import Path

# 10 个核心 Wind 指标 (跟 Mac 端 PIT 对齐)
WIND_INDICATORS = [
    "roe_ttm",            # ROE TTM
    "grossprofitmargin",  # 毛利率
    "netprofitmargin",    # 净利率
    "yoynetprofit",       # 净利润同比 (%)
    "yoy_or",             # 营业总收入同比 (%)
    "yoyeps_basic",       # 基本 EPS 同比 (%)
    "opercashflowps",     # 每股经营现金流
    "debttoassets",       # 资产负债率 (%)
    "assetsturn",         # 总资产周转率
    "eps_basic",          # 基本每股收益
]

# 字段名映射 (Wind 名 → 我们的字段名), 跟 factors/alpha158_finance.py 对齐
FIELD_MAP = {
    "roe_ttm":           "roe",
    "grossprofitmargin": "gpm",
    "netprofitmargin":   "npm",
    "yoynetprofit":      "yoy_ni",
    "yoy_or":            "yoy_rev",
    "yoyeps_basic":      "yoy_eps",
    "opercashflowps":    "ocfps",
    "debttoassets":      "debt_ratio",
    "assetsturn":        "asset_turn",
    "eps_basic":         "eps",
}

START = "2010-01-01"
END = "2026-05-29"
OUT_DIR = Path("./financials_wind")
OUT_DIR.mkdir(exist_ok=True)
CODES_FILE = Path("./wind_codes_for_query.txt")  # 同目录


def load_codes():
    """从 wind_codes_for_query.txt 读股票代码 (Wind 格式: 600000.SH)."""
    if not CODES_FILE.exists():
        print(f"❌ 找不到 {CODES_FILE.absolute()}")
        print("   请把 Mac 端 ~/qlib量化研究/data_raw/meta/wind_codes_for_query.txt 拷到当前目录")
        sys.exit(1)
    return [l.strip() for l in CODES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def fetch_one(w, code):
    """用 wsd 拉单股全期所有指标 (季度频率)."""
    fields = ",".join(WIND_INDICATORS)
    data = w.wsd(code, fields, START, END, "Period=Q,Fill=Previous")
    if data.ErrorCode != 0:
        return None, f"Wind ErrorCode={data.ErrorCode}: {data.Data}"
    import pandas as pd
    df = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T
    df.index.name = "statDate"
    df = df.reset_index()
    df["statDate"] = pd.to_datetime(df["statDate"]).dt.strftime("%Y-%m-%d")
    df["code"] = code
    # 重命名为我们的字段名
    df = df.rename(columns=FIELD_MAP)
    return df, None


def main():
    print("=" * 60)
    print("Wind 财务数据采集 (Windows)")
    print("=" * 60)
    try:
        from WindPy import w
    except ImportError:
        print("❌ WindPy 未装. 请先在 Wind 终端安装 Python API")
        sys.exit(1)

    print("启动 WindPy ...")
    ret = w.start()
    if ret.ErrorCode != 0:
        print(f"❌ w.start() 失败: ErrorCode={ret.ErrorCode}")
        print("   请确保 Wind 终端已打开且已登录")
        sys.exit(1)
    print(f"✓ Wind 已连接 (isconnected={w.isconnected()})")

    codes = load_codes()
    print(f"读到 {len(codes)} 个股票代码")

    # 跳过已采
    todo = [c for c in codes if not (OUT_DIR / f"{c}.csv").exists()
            or (OUT_DIR / f"{c}.csv").stat().st_size <= 500]
    print(f"已采 {len(codes)-len(todo)}, 待采 {len(todo)}")
    print(f"输出目录: {OUT_DIR.absolute()}")
    print()

    ok = skip = 0
    fail = []
    t0 = time.time()
    for i, code in enumerate(todo):
        df, err = fetch_one(w, code)
        if df is None or df.empty:
            fail.append((code, err or "empty"))
            print(f"  [{i+1}/{len(todo)}] {code} ❌ {err}")
        else:
            df.to_csv(OUT_DIR / f"{code}.csv", index=False)
            ok += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            eta = (len(todo) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(todo)}] ok={ok} fail={len(fail)} | "
                  f"{rate:.2f}/s | ETA {eta:.1f} min")

    print(f"\n[DONE] ok={ok} fail={len(fail)}")
    if fail:
        (OUT_DIR / "_failed.txt").write_text(
            "\n".join(f"{c}\t{r}" for c, r in fail), encoding="utf-8")
        print(f"失败列表 → {OUT_DIR / '_failed.txt'}")
    print(f"\n请把 {OUT_DIR.absolute()} 整个目录 zip + 传到 Mac:")
    print(f"  ~/qlib量化研究/data_raw/financials_wind/")
    w.stop()


if __name__ == "__main__":
    main()
