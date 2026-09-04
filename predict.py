#!/usr/bin/env python3
"""Segmenta imágenes con el modelo entrenado. Punto de entrada más simple posible.

    python predict.py imagen.png                 # guarda imagen_seg.png al lado
    python predict.py carpeta/                   # procesa la carpeta entera
    python predict.py carpeta/ --out resultados/ --masks

Si falta model.th, se descarga solo. Solo necesita torch, numpy y Pillow;
matplotlib únicamente para el panel comparativo.
"""

import argparse
import os
import sys
import urllib.request

MODEL_URL = os.environ.get('STK_MODEL_URL', 'https://github.com/ferjozsot23/vision-supertuxkart/releases/latest/download/model.th')
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def ensure_model(path):
    if os.path.exists(path):
        return path
    if not MODEL_URL:
        sys.exit(
            "No se encuentra '%s'.\n"
            "  - Puede descargarse desde la sección Releases del repositorio, o\n"
            "  - definir STK_MODEL_URL con su URL para descargarlo automáticamente." % path)
    print('Descargando el modelo (~118 MB)...', flush=True)

    def _p(n, bs, total):
        if total > 0:
            pct = min(100, n * bs * 100 // total)
            print('\r  %d%%' % pct, end='', flush=True)

    urllib.request.urlretrieve(MODEL_URL, path, _p)
    print('\r  completado    ')
    return path


def main():
    ap = argparse.ArgumentParser(
        description='Segmentación semántica de SuperTuxKart en 7 clases.',
        epilog='Ejemplo:  python predict.py ejemplos/ --out salida/')
    ap.add_argument('entrada', help='imagen o carpeta de imágenes')
    ap.add_argument('--out', default=None,
                    help='carpeta de salida (por defecto, junto a la entrada)')
    ap.add_argument('--model', default='model.th')
    ap.add_argument('--masks', action='store_true',
                    help='guarda además la máscara cruda de clases (0..6, 1 canal)')
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda')
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from models import load_model
    from utils import CLASS_NAMES, label_to_color, overlay

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(ensure_model(args.model), device=device)
    print('Modelo cargado en %s' % device)

    if os.path.isdir(args.entrada):
        files = sorted(os.path.join(args.entrada, f) for f in os.listdir(args.entrada)
                       if f.lower().endswith(IMG_EXT))
    else:
        files = [args.entrada]
    if not files:
        sys.exit('No se encontraron imágenes en %s' % args.entrada)

    outdir = args.out or (args.entrada if os.path.isdir(args.entrada)
                          else os.path.dirname(os.path.abspath(args.entrada)))
    os.makedirs(outdir, exist_ok=True)

    for f in files:
        img = Image.open(f).convert('RGB')
        x = torch.from_numpy(np.asarray(img, np.uint8).copy()) \
                 .permute(2, 0, 1).float().div_(255.0)[None].to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)

        stem = os.path.splitext(os.path.basename(f))[0]
        rgb = np.asarray(img)
        panel = np.concatenate([rgb, label_to_color(pred),
                                overlay(rgb.transpose(2, 0, 1) / 255.0, pred)], axis=1)
        out = os.path.join(outdir, stem + '_seg.png')
        Image.fromarray(panel).save(out)
        if args.masks:
            Image.fromarray(pred).save(os.path.join(outdir, stem + '_mask.png'))

        c = np.bincount(pred.ravel(), minlength=7)
        pct = 100.0 * c / c.sum()
        print('%-28s -> %s   %s' % (
            os.path.basename(f), os.path.basename(out),
            ' '.join('%s %.1f%%' % (CLASS_NAMES[i][:4], pct[i])
                     for i in range(7) if c[i] > 0)))

    print('\n%d imagen(es) en %s/' % (len(files), outdir.rstrip('/')))
    print('Panel: original | segmentación | superposición')


if __name__ == '__main__':
    main()
