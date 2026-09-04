"""Figuras del README. Genera cada una en version clara y oscura."""
import os, sys
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from models import load_model
from utils import CLASS_NAMES, label_to_color, PALETTE

TH = {
 'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', ink3='#8a8984',
               series='#2a78d6', grid='#e6e5e1'),
 'dark':  dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', ink3='#7d7c74',
               series='#3987e5', grid='#333330'),
}

IOU = [('background',0.7768), ('track',0.7365), ('kart',0.6709),
       ('nitro',0.5352), ('pickup',0.5013), ('bomb',0.0194), ('projectile',0.0000)]
FRAMES_TOTAL = {'background':1000,'track':1000,'kart':1000,'nitro':394,
                'pickup':433,'bomb':168,'projectile':2}

POWER = [(0.20,0.3940), (0.25,0.4565), (0.30,0.4690), (0.35,0.4166),
         (0.40,0.3922), (0.50,0.3985), (0.75,0.3309)]


def style(fig, axes, t):
    fig.patch.set_facecolor(t['surface'])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(t['surface'])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=t['ink2'], labelsize=9, length=0)


def fig_iou(mode):
    t = TH[mode]
    names = [n for n, _ in IOU][::-1]
    vals = [v for _, v in IOU][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.55, color=t['series'], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=t['ink'], fontsize=10)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for yi, v, n in zip(y, vals, names):
        ax.text(v + 0.018, yi, '%.4f' % v, va='center', ha='left',
                color=t['ink'], fontsize=10, fontweight='medium')
        ax.text(1.16, yi, '%s frames' % FRAMES_TOTAL[n], va='center', ha='right',
                color=t['ink3'], fontsize=9)
    ax.text(1.16, len(names) - 0.35, 'en entrenamiento', va='center', ha='right',
            color=t['ink3'], fontsize=8.5, style='italic')
    style(fig, ax, t)
    ax.tick_params(axis='y', labelcolor=t['ink'])
    fig.suptitle('IoU por clase — validación en 2 tracks nunca vistos',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.905, 'mIoU 0.4629 · las dos últimas clases apenas existen en el dataset',
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig('docs/figuras/iou-%s.png' % mode, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_power(mode):
    t = TH[mode]
    xs = [p for p, _ in POWER]; ys = [v for _, v in POWER]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(xs, ys, '-', color=t['series'], lw=2, zorder=3)
    ax.plot(xs, ys, 'o', color=t['series'], ms=8, zorder=4,
            markeredgecolor=t['surface'], markeredgewidth=2)
    best = max(POWER, key=lambda p: p[1])
    ax.annotate('  óptimo  %.4f' % best[1], best, color=t['ink'], fontsize=10,
                fontweight='medium', va='center')
    ax.set_xlabel('exponente del peso de clase   (peso = frecuencia$^{-p}$)',
                  color=t['ink2'], fontsize=9.5)
    ax.set_ylabel('mIoU', color=t['ink2'], fontsize=9.5)
    ax.set_ylim(0.30, 0.50)
    ax.yaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    style(fig, ax, t)
    fig.suptitle('Castigar más el error en las clases raras las EMPEORA',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.885, 'cada punto es una corrida completa de 40 épocas',
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig('docs/figuras/power-%s.png' % mode, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_qual(mode, frames):
    t = TH[mode]
    model = load_model('model.th')
    fig, axes = plt.subplots(len(frames), 3, figsize=(9.8, 3.35 * len(frames)))
    for r, (track, fid, etiqueta, miou) in enumerate(frames):
        fp = 'dataset/dense_data/%s/frame/frame_%s.png' % (track, fid)
        mp = 'dataset/dense_data/%s/combined/mask_combined_%s.png' % (track, fid)
        img = Image.open(fp).convert('RGB')
        x = torch.from_numpy(np.asarray(img, np.uint8).copy()).permute(2,0,1).float().div_(255)[None]
        with torch.no_grad():
            pred = model(x).argmax(1)[0].numpy()
        gt = np.asarray(Image.open(mp))
        for c, (data, title) in enumerate([(np.asarray(img), 'entrada'),
                                           (label_to_color(pred), 'predicción'),
                                           (label_to_color(gt), 'verdad')]):
            ax = axes[r, c]
            ax.imshow(data); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(t['grid']); sp.set_linewidth(1)
            if r == 0:
                ax.set_title(title, color=t['ink'], fontsize=11.5, pad=8)
        axes[r, 0].set_ylabel('%s\nIoU %.2f' % (etiqueta, miou), color=t['ink'],
                              fontsize=10.5, labelpad=10)
    handles = [plt.Rectangle((0,0),1,1, fc=PALETTE[i]/255.0,
               ec=t['grid'], lw=0.5) for i in range(7)]
    fig.legend(handles, CLASS_NAMES, loc='lower center', ncol=7, frameon=False,
               fontsize=9.5, labelcolor=t['ink2'], bbox_to_anchor=(0.5, 0.004))
    fig.patch.set_facecolor(t['surface'])
    for ax in axes.ravel():
        ax.set_facecolor(t['surface'])
    fig.tight_layout(rect=[0, 0.042, 1, 1])
    fig.savefig('docs/figuras/cualitativo-%s.png' % mode, dpi=145, facecolor=t['surface'])
    plt.close(fig)


FRAMES = [('volcano_island', '0125', 'mejor caso',  0.9664),
          ('volcano_island', '0155', 'mediana',     0.5479),
          ('volcano_island', '0240', 'peor caso',   0.2091)]
for mode in ('light', 'dark'):
    fig_iou(mode); fig_power(mode); fig_qual(mode, FRAMES)
    print('generadas figuras', mode)
