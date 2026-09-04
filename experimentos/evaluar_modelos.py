"""Mide de verdad uno o varios .th sobre el conjunto de validacion.

    python3 experimentos/evaluar_modelos.py --data-root dataset/dense_data \
        model.th salidas/barrido/*.th

No se fia de lo que digan los csv del barrido: vuelve a calcular la matriz de
confusion sobre los 500 frames de validacion. Es la comprobacion que decide que
modelo se entrega.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import torch

from models import load_model
from utils import CLASS_NAMES, ConfusionMatrix, load_data


@torch.no_grad()
def score(path, data_root, size, device, batch_size):
    model = load_model(path, device=device)
    base = model.enc1.block[0].weight.shape[0]
    loader = load_data(data_root, 'val', size=size, batch_size=batch_size,
                       num_workers=0, augment=False)
    cm = ConfusionMatrix()
    for img, mask in loader:
        cm.add(model(img.to(device)).argmax(1), mask.to(device))
    return cm, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modelos', nargs='+')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--size', type=int, nargs=2, default=[256, 256])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    paths = []
    for p in args.modelos:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in '*?[') else [p])
    paths = [p for p in paths if os.path.isfile(p)]

    short = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']
    hdr = '%-46s %5s %7s ' % ('modelo', 'base', 'mIoU') + ' '.join('%6s' % s for s in short)
    print(hdr); print('-' * len(hdr))

    rows = []
    for p in paths:
        try:
            cm, base = score(p, args.data_root, tuple(args.size), args.device, args.batch_size)
        except Exception as e:
            print('%-46s  ERROR: %s' % (os.path.basename(p)[:46], e))
            continue
        iou = cm.class_iou
        rows.append((cm.miou, p, base, iou))
        print('%-46s %5d %7.4f ' % (os.path.basename(p)[:46], base, cm.miou)
              + ' '.join('%6.4f' % float(v) if float(v) == float(v) else '   nan' for v in iou),
              flush=True)

    rows.sort(key=lambda r: -r[0])
    if rows:
        m, p, base, iou = rows[0]
        print('-' * len(hdr))
        print('MEJOR MEDIDO: %s' % p)
        print('  base %d   mIoU %.4f' % (base, m))
        print('  ' + ',  '.join('%s %.4f' % (c, float(v)) for c, v in zip(CLASS_NAMES, iou)))


if __name__ == '__main__':
    main()
