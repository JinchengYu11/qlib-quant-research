"""把 10 张图拼成一张总览大图。"""
from pathlib import Path
from PIL import Image

OUT = Path("/Users/cedricyu/qlib量化研究/results/figures")
PAGES = [
    ("gallery.png", [
        ["fig1_cumulative_net.png", None],         # 图1 横跨两列
        ["fig2_yearly_excess.png", None],          # 图2 横跨两列
        ["fig3_summary_bars.png", None],           # 图3 横跨两列
        ["fig4_bias_viz.png", None],               # 图4 横跨两列
        ["fig5_monthly_heatmap.png", None],        # 图5 横跨两列
        ["fig6_rolling_ir.png", "fig7_underwater.png"],
        ["fig8_factor_importance.png", "fig9_ic_timeseries.png"],
        ["fig10_correlation.png", None],
    ]),
]
MARGIN = 20
COL_W = 1600  # 单列目标宽度


def fit_to_width(img, target_w):
    """按比例缩到目标宽度。"""
    w, h = img.size
    new_h = int(h * target_w / w)
    return img.resize((target_w, new_h), Image.LANCZOS)


def make_gallery(out_name, layout):
    rows = []
    for row in layout:
        left = Image.open(OUT / row[0])
        if row[1] is None:
            # 横跨两列
            fit = fit_to_width(left, COL_W * 2 + MARGIN)
            rows.append([fit])
        else:
            right = Image.open(OUT / row[1])
            fl = fit_to_width(left, COL_W)
            fr = fit_to_width(right, COL_W)
            # 高度对齐到较高那张
            h = max(fl.height, fr.height)
            row_img = Image.new("RGB", (COL_W * 2 + MARGIN, h), "white")
            row_img.paste(fl, (0, 0))
            row_img.paste(fr, (COL_W + MARGIN, 0))
            rows.append([row_img])
    # 总宽度
    total_w = COL_W * 2 + MARGIN
    total_h = sum(r[0].height for r in rows) + MARGIN * (len(rows) - 1)
    canvas = Image.new("RGB", (total_w, total_h), "white")
    y = 0
    for r in rows:
        canvas.paste(r[0], (0, y))
        y += r[0].height + MARGIN
    canvas.save(OUT / out_name, optimize=True)
    print(f"✓ {out_name}  尺寸 {canvas.size}, {(OUT / out_name).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    for name, layout in PAGES:
        make_gallery(name, layout)
