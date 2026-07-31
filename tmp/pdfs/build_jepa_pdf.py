from __future__ import annotations

import csv, json, math, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/Users/gabengcui/study/lewm/homework')
RUN = ROOT / 'tmp/pdfs/experiment_summary_20260724'
OUT = ROOT / 'output/pdf/JEPA_experiment_report_20260724.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1654, 2339  # A4 at ~200 dpi
M = 90
BG = 'white'
TEXT = (32, 35, 40)
MUTED = (92, 98, 108)
GRID = (214, 219, 226)
HEADER = (241, 245, 249)
BLUE = (30, 96, 180)
GREEN = (41, 128, 85)
RED = (170, 60, 60)

FONT_REG = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
FONT_CJK = '/System/Library/Fonts/STHeiti Medium.ttc'

def font(size, bold=False):
    path = FONT_CJK if bold else FONT_REG
    return ImageFont.truetype(path, size)

F_TITLE = font(46, True)
F_H1 = font(34, True)
F_H2 = font(28, True)
F_BODY = font(23)
F_SMALL = font(19)
F_TABLE = font(19)
F_TABLE_B = font(19, True)

pages = []
img = Image.new('RGB', (W, H), BG)
d = ImageDraw.Draw(img)
y = M
page_no = 1

def add_page():
    global img, d, y, page_no
    d.text((W - M - 120, H - 58), f'Page {page_no}', fill=MUTED, font=F_SMALL)
    pages.append(img)
    page_no += 1
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    y = M

def ensure(space):
    if y + space > H - M:
        add_page()

def draw_wrapped(text, x, y0, max_width, fnt=F_BODY, fill=TEXT, line_gap=7):
    words = list(text)
    lines = []
    cur = ''
    for ch in words:
        test = cur + ch
        if d.textlength(test, font=fnt) <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    yy = y0
    for line in lines:
        d.text((x, yy), line, fill=fill, font=fnt)
        yy += fnt.size + line_gap
    return yy

def title(text):
    global y
    ensure(80)
    d.text((M, y), text, fill=TEXT, font=F_TITLE)
    y += 78

def h1(text):
    global y
    ensure(70)
    d.text((M, y), text, fill=BLUE, font=F_H1)
    y += 56
    d.line((M, y, W-M, y), fill=GRID, width=2)
    y += 28

def h2(text):
    global y
    ensure(55)
    d.text((M, y), text, fill=TEXT, font=F_H2)
    y += 48

def para(text):
    global y
    ensure(70)
    y = draw_wrapped(text, M, y, W-2*M, F_BODY)
    y += 18

def load_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def load_json(path):
    return json.loads(path.read_text())

def num(v, nd=2):
    try:
        return f'{float(v):.{nd}f}'
    except Exception:
        return 'NA'

def make_table(headers, rows, col_widths, row_h=48, header_h=52, fnt=F_TABLE, fnt_b=F_TABLE_B):
    global y
    total_w = sum(col_widths)
    x0 = M
    needed = header_h + row_h * len(rows) + 20
    ensure(needed)
    d.rectangle((x0, y, x0+total_w, y+header_h), fill=HEADER, outline=GRID)
    x = x0
    for h, cw in zip(headers, col_widths):
        d.text((x+10, y+14), h, fill=TEXT, font=fnt_b)
        d.line((x, y, x, y+header_h+row_h*len(rows)), fill=GRID, width=1)
        x += cw
    d.line((x0+total_w, y, x0+total_w, y+header_h+row_h*len(rows)), fill=GRID, width=1)
    d.line((x0, y+header_h, x0+total_w, y+header_h), fill=GRID, width=1)
    yy = y + header_h
    for row in rows:
        d.rectangle((x0, yy, x0+total_w, yy+row_h), outline=GRID)
        x = x0
        for cell, cw in zip(row, col_widths):
            fill = TEXT
            s = str(cell)
            if s.startswith('+'):
                fill = GREEN
            elif s.startswith('-'):
                fill = RED
            if d.textlength(s, font=fnt) > cw - 18:
                while s and d.textlength(s + '...', font=fnt) > cw - 18:
                    s = s[:-1]
                s += '...'
            d.text((x+10, yy+14), s, fill=fill, font=fnt)
            x += cw
        yy += row_h
    y = yy + 28

def add_image(path, caption=None, max_w=None, max_h=720):
    global y
    path = RUN / path
    if not path.exists():
        para(f'[missing image] {path}')
        return
    im = Image.open(path).convert('RGB')
    max_w = max_w or (W - 2*M)
    scale = min(max_w / im.width, max_h / im.height, 1.0)
    new = im.resize((int(im.width*scale), int(im.height*scale)), Image.LANCZOS)
    need = new.height + (42 if caption else 20)
    ensure(need)
    x = (W - new.width) // 2
    img.paste(new, (x, y))
    y += new.height + 10
    if caption:
        d.text((M, y), caption, fill=MUTED, font=F_SMALL)
        y += 34
    y += 12

def add_two_images(path1, path2, caption1, caption2, max_h=500):
    global y
    im1 = Image.open(RUN / path1).convert('RGB')
    im2 = Image.open(RUN / path2).convert('RGB')
    gap = 28
    max_w = (W - 2*M - gap) // 2
    s1 = min(max_w / im1.width, max_h / im1.height, 1.0)
    s2 = min(max_w / im2.width, max_h / im2.height, 1.0)
    im1 = im1.resize((int(im1.width*s1), int(im1.height*s1)), Image.LANCZOS)
    im2 = im2.resize((int(im2.width*s2), int(im2.height*s2)), Image.LANCZOS)
    need = max(im1.height, im2.height) + 70
    ensure(need)
    x1 = M
    x2 = M + max_w + gap
    img.paste(im1, (x1 + (max_w-im1.width)//2, y))
    img.paste(im2, (x2 + (max_w-im2.width)//2, y))
    yy = y + max(im1.height, im2.height) + 8
    d.text((x1, yy), caption1, fill=MUTED, font=F_SMALL)
    d.text((x2, yy), caption2, fill=MUTED, font=F_SMALL)
    y = yy + 44

def metric_note(rows):
    global y
    make_table(['指标', '含义'], rows, [360, W-2*M-360], row_h=44, header_h=46, fnt=F_SMALL, fnt_b=F_TABLE_B)

summary = load_csv(RUN / 'eval_summary/eval_summary.csv')
by = {r['group']: r for r in summary}

def delta(group, base):
    return float(by[group]['success_rate_mean']) - float(by[base]['success_rate_mean'])

orig_pred = load_json(RUN / 'original_lewm_15/prediction_viz/prediction_report.json')
res_pred = load_json(RUN / 'residual_15/prediction_viz/prediction_report.json')
orig_col = load_json(RUN / 'original_lewm_15/latent_viz/collapse_diagnostics_summary.json')
res_col = load_json(RUN / 'residual_15/latent_viz/collapse_diagnostics_summary.json')

# Cover
page_grad = Image.new('RGB', (W, H), (248, 250, 252))
img.paste(page_grad)
d.rectangle((0, 0, W, 260), fill=(25, 72, 128))
d.text((M, 95), 'JEPA 实验报告', fill='white', font=F_TITLE)
d.text((M, 168), 'PushT / Reacher / Factored 结果汇总', fill=(220, 235, 255), font=F_H2)
y = 330
para('本 PDF 基于 outputs/jepa_viz/experiment_summary_20260724 中的 eval 日志、训练 metrics.csv、latent 导出结果和 tools/jepa_viz 生成的可视化文件。')
make_table(['部分', '内容'], [
    ['PushT', 'Original / Residual / Factored 独立对比'],
    ['Reacher', 'Reacher Original / Residual 独立对比'],
    ['Prediction', 'horizon cosine、alignment heatmap、condition ablation'],
    ['Latent', 'active dims、pairwise cosine、covariance spectrum'],
], [320, W-2*M-320], row_h=54)
add_page()

h1('1. PushT：Original / Residual / Factored 对比')
pusht_rows = []
for name in ['PushT Original 15','PushT Residual 15','PushT Factored 96-96 15','PushT Factored 64-128 15']:
    r = by[name]
    dlt = 0 if name == 'PushT Original 15' else delta(name, 'PushT Original 15')
    pusht_rows.append([name.replace('PushT ',''), f"{num(r['success_rate_mean'])} +/- {num(r['success_rate_std'])}", f"{dlt:+.2f}", f"{num(r['time_per_episode_mean_sec'])}s"])
make_table(['模型','成功率','Δ vs Original','time/ep'], pusht_rows, [440, 360, 300, 260])
add_image('eval_summary_pusht/success_rate_summary.png', 'PushT success rate mean +/- std')
metric_note([['success rate', '多 seed 平均成功率，PushT 内横向比较。'], ['Δ vs Original', '相对 PushT Original 15 的成功率变化。'], ['std', 'seed 间波动，越小越稳定。']])
add_image('eval_summary_pusht/success_rate_by_seed.png', 'PushT per-seed success rate')
add_image('eval_summary_pusht/time_per_episode.png', 'PushT time per episode')

h1('2. Reacher：Original / Residual 对比')
reacher_rows = []
for name in ['Reacher Original 15','Reacher Residual 15']:
    r = by[name]
    dlt = 0 if name == 'Reacher Original 15' else delta(name, 'Reacher Original 15')
    reacher_rows.append([name.replace('Reacher ',''), f"{num(r['success_rate_mean'])} +/- {num(r['success_rate_std'])}", f"{dlt:+.2f}", f"{num(r['time_per_episode_mean_sec'])}s"])
make_table(['模型','成功率','Δ vs Reacher Original','time/ep'], reacher_rows, [440, 360, 400, 220])
add_image('eval_summary_reacher/success_rate_summary.png', 'Reacher success rate mean +/- std')
metric_note([['success rate', '只在 Reacher Original 与 Reacher Residual 之间比较。'], ['eval_budget', 'Reacher 使用 budget=50，不和 PushT budget=300 耗时直接比较。'], ['std', 'Reacher seed 间波动较大，需要关注分布。']])
add_image('eval_summary_reacher/success_rate_by_seed.png', 'Reacher per-seed success rate')
add_image('eval_summary_reacher/time_per_episode.png', 'Reacher time per episode')

h1('3. PushT Prediction 指标')
def pred_row(rep):
    chk = rep['prediction_checks']; heat = rep['alignment_heatmap']; abl = rep['condition_ablation']
    return chk, heat, abl
co, ho, ao = pred_row(orig_pred)
cr, hr, ar = pred_row(res_pred)
make_table(['指标','Original','Residual'], [
    ['mean cosine', num(co['mean_cosine'],5), num(cr['mean_cosine'],5)],
    ['diagonal gap', num(ho['diagonal_gap'],5), num(hr['diagonal_gap'],5)],
    ['top-1 horizon', f"{ho['top1_horizon_matching_accuracy']*100:.2f}%", f"{hr['top1_horizon_matching_accuracy']*100:.2f}%"],
    ['normal MSE', num(ao['normal']['mse'],5), num(ar['normal']['mse'],5)],
    ['shuffled-action MSE', num(ao['condition_shuffled']['mse'],5), num(ar['condition_shuffled']['mse'],5)],
], [500, 420, 420])
add_two_images('original_lewm_15/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png', 'residual_15/prediction_viz/target_pred_cosine_vs_horizon_by_action_norm_bin.png', 'Original horizon cosine', 'Residual horizon cosine')
metric_note([['horizon cosine', 'cos(z_pred, z_target)，越高表示预测越接近目标。'], ['action_norm_bin', '按 action 范数分桶，观察动作强度对预测的影响。']])
add_two_images('original_lewm_15/prediction_viz/target_pred_alignment_heatmap.png', 'residual_15/prediction_viz/target_pred_alignment_heatmap.png', 'Original alignment heatmap', 'Residual alignment heatmap')
metric_note([['diagonal gap', '对角线平均值减去非对角线平均值，越大越能区分 future horizon。'], ['top-1 matching', '预测 horizon 匹配正确 target horizon 的比例。']])
add_two_images('original_lewm_15/prediction_viz/condition_ablation.png', 'residual_15/prediction_viz/condition_ablation.png', 'Original condition ablation', 'Residual condition ablation')
metric_note([['normal', '使用正常 action condition 的预测。'], ['condition shuffled', '打乱 action 后的预测；退化越明显，说明模型越依赖正确 action。']])

h1('4. PushT Latent 表征诊断')
zo = orig_col['z_pred']; zr = res_col['z_pred']
make_table(['指标','Original z_pred','Residual z_pred'], [
    ['active dims', f"{zo['active_dim_count']}/{zo['active_dim_total']}", f"{zr['active_dim_count']}/{zr['active_dim_total']}"],
    ['effective rank', num(zo['effective_rank'],2), num(zr['effective_rank'],2)],
    ['participation ratio', num(zo['participation_ratio'],2), num(zr['participation_ratio'],2)],
    ['top10 explained var', num(zo['top10_explained_variance_ratio'],3), num(zr['top10_explained_variance_ratio'],3)],
    ['pairwise cosine q95', num(zo['pairwise_cosine_q95'],4), num(zr['pairwise_cosine_q95'],4)],
], [500, 420, 420])
add_two_images('original_lewm_15/latent_viz/active_dimension_count.png', 'residual_15/latent_viz/active_dimension_count.png', 'Original active dims', 'Residual active dims')
metric_note([['active dims', 'std 超过阈值的 latent 维度数；过低表示 collapse。'], ['effective rank', '协方差谱有效秩，越高表示维度利用更分散。']])
add_two_images('original_lewm_15/latent_viz/pairwise_cosine_histogram.png', 'residual_15/latent_viz/pairwise_cosine_histogram.png', 'Original pairwise cosine', 'Residual pairwise cosine')
add_two_images('original_lewm_15/latent_viz/covariance_eigenvalue_spectrum.png', 'residual_15/latent_viz/covariance_eigenvalue_spectrum.png', 'Original covariance spectrum', 'Residual covariance spectrum')
add_two_images('original_lewm_15/latent_viz/target_pred_latent_alignment_global.png', 'residual_15/latent_viz/target_pred_latent_alignment_global.png', 'Original target-pred alignment', 'Residual target-pred alignment')

h1('5. 训练曲线索引')
make_table(['模型','主要曲线目录'], [
    ['Reacher Original 15', 'training/reacher_original_15/'],
    ['Reacher Residual 15', 'training/reacher_residual_15/'],
    ['Factored 96/96 15', 'training/factored_96_96_15/'],
    ['Factored 64/128 15', 'training/factored_64_128_15/'],
], [480, 860])
add_two_images('training/reacher_original_15/val_loss.png', 'training/reacher_residual_15/val_loss.png', 'Reacher Original val loss', 'Reacher Residual val loss')
add_two_images('training/factored_96_96_15/val_loss.png', 'training/factored_64_128_15/val_loss.png', 'Factored 96/96 val loss', 'Factored 64/128 val loss')
metric_note([['val loss', '验证集 loss，用于观察泛化。'], ['per-horizon MSE', '每个 future horizon 的 MSE，检查多步预测是否区分。'], ['samples_per_sec', '训练吞吐，不与 learning rate 混在同一 y 轴。']])

h1('6. 结果说明')
para('PushT 上 Residual 15 的成功率均值为 90.0%，Original 15 为 88.8%，差距为 +1.2 个百分点。Factored 两个版本均低于 PushT Original/Residual，其中 96/96 明显优于 64/128。')
para('Reacher 是独立实验，只比较 Reacher Original 与 Reacher Residual。当前 Reacher Residual 15 为 57.6%，Reacher Original 15 为 62.0%，Residual 未带来改善。')
para('PushT latent prediction 中，Original 的 normal MSE 更低、diagonal gap 更大，说明 horizon alignment 更清晰。两个 prediction report 都提示 mean cosine > 0.99，虽然 z_pred 与 z_target 不是完全相同，但后续仍应持续检查 target leakage 或 latent 读取路径。')

add_page()
pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200.0)
print(OUT)
print('pages', len(pages))
