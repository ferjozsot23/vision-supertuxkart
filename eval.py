"""Carga un modelo .th entrenado, segmenta imagenes y muestra el resultado.

    python eval.py --model model.th --image ruta/frame_0000.png
    python eval.py --model model.th --image carpeta/ --out resultados/

Funciona con una imagen suelta o con una carpeta. La imagen se pasa CRUDA en
[0,1]: la normalizacion mean/std vive dentro del modelo, no aqui.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from models import load_model
from utils import CLASS_NAMES, NUM_CLASSES, label_to_color, overlay

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def load_image(image_path, size=None):
    img = Image.open(image_path).convert('RGB')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


@torch.no_grad()
def predict(model, image_path, device='cpu', size=None):
    x = load_image(image_path, size).to(device)
    logits = model(x)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def visualize(image_path, pred, out_path=None, truth_path=None):
    import matplotlib
    if out_path:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from utils import PALETTE

    img = np.asarray(Image.open(image_path).convert('RGB').resize(
        (pred.shape[1], pred.shape[0]), Image.BILINEAR))
    panels = [(img, 'original'),
              (label_to_color(pred), 'segmentacion'),
              (overlay(img.transpose(2, 0, 1) / 255.0, pred), 'overlay')]
    if truth_path and os.path.exists(truth_path):
        gt = np.asarray(Image.open(truth_path).resize(
            (pred.shape[1], pred.shape[0]), Image.NEAREST))
        panels.append((label_to_color(gt), 'ground truth'))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.8))
    for ax, (data, title) in zip(np.atleast_1d(axes), panels):
        ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    present = sorted(int(c) for c in np.unique(pred))
    fig.legend(handles=[Patch(facecolor=PALETTE[c] / 255.0, label=CLASS_NAMES[c])
                        for c in present if c < NUM_CLASSES],
               loc='lower center', ncol=min(len(present), 7), frameon=False, fontsize=9)
    fig.suptitle(os.path.basename(image_path), fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print('escrito ->', out_path)
    else:
        plt.show()


def guess_truth_path(image_path):
    d, name = os.path.split(os.path.abspath(image_path))
    if os.path.basename(d) != 'frame' or not name.startswith('frame_'):
        return None
    cand = os.path.join(os.path.dirname(d), 'combined',
                        'mask_combined_' + name[len('frame_'):])
    return cand if os.path.exists(cand) else None


def collect_images(path, limit=None):
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(IMG_EXT))
    else:
        files = [path]
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser(description='Segmenta imagenes con un modelo .th entrenado')
    ap.add_argument('--model', default='model.th')
    ap.add_argument('--image', required=True, help='imagen suelta o carpeta')
    ap.add_argument('--out', default=None, help='archivo o carpeta de salida; sin esto abre una ventana')
    ap.add_argument('--size', type=int, nargs=2, default=None, metavar=('H', 'W'),
                    help='redimensiona antes de predecir; por defecto usa la resolucion nativa')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-truth', action='store_true', help='no buscar la mascara real')
    args = ap.parse_args()

    model = load_model(args.model, device=args.device)
    files = collect_images(args.image, args.limit)
    if not files:
        raise SystemExit('no hay imagenes en %s' % args.image)
    print('modelo %s | %d imagen(es)' % (args.model, len(files)))

    many = len(files) > 1
    for f in files:
        pred = predict(model, f, device=args.device,
                       size=tuple(args.size) if args.size else None)
        counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)
        pct = 100.0 * counts / counts.sum()
        print('%-28s %s' % (os.path.basename(f),
              '  '.join('%s=%.2f%%' % (CLASS_NAMES[c][:4], pct[c])
                        for c in range(NUM_CLASSES) if counts[c] > 0)))
        if args.out is None:
            out = None
        elif many or os.path.isdir(args.out) or args.out.endswith(os.sep):
            out = os.path.join(args.out, 'pred_' + os.path.splitext(os.path.basename(f))[0] + '.png')
        else:
            out = args.out
        visualize(f, pred, out, None if args.no_truth else guess_truth_path(f))


if __name__ == '__main__':
    main()
