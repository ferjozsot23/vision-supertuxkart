"""Barrido de hiperparametros: recorre una rejilla y consolida los resultados.

Pensado para lanzarse una vez y volver cuando termine. Cada combinacion escribe
sus epocas en runs.csv y su mejor epoca en sweep_results.csv, y al final imprime
un ranking ordenado por mIoU.

    python3 sweep.py --data-root /data --powers 0.3 0.35 0.4 --scales 1.0 0.85

Una combinacion que reviente NO tumba el barrido: se registra el error y sigue.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import csv
import itertools
import time
import traceback

import torch

from train import train
from utils import CLASS_NAMES

FIELDS = (['combo', 'size', 'lr', 'base', 'weight_power', 'scale_min', 'augment',
           'epochs', 'best_epoch', 'best_miou']
          + ['iou_' + c for c in CLASS_NAMES] + ['minutos', 'estado'])


def build_args(base_args, size, lr, base, power, scale, tag):
    a = copy.deepcopy(base_args)
    a.size = list(size)
    a.lr = lr
    a.base = base
    a.weight_power = power
    a.scale_min = scale
    a.notes = tag
    a.out = os.path.join(base_args.out_dir, 'model_%s.th' % tag)
    a.ckpt_dir = os.path.join(base_args.out_dir, 'ckpt_%s' % tag)
    a.resume = None
    return a


def main():
    ap = argparse.ArgumentParser(description='Barrido de hiperparametros')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[256],
                    help='lados (cuadrados): --sizes 256 384')
    ap.add_argument('--powers', type=float, nargs='+', default=[0.35])
    ap.add_argument('--scales', type=float, nargs='+', default=[1.0])
    ap.add_argument('--bases', type=int, nargs='+', default=[32])
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--early-stop', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--weights', default='class_weights.json')
    ap.add_argument('--weight-clip', type=float, default=10.0)
    ap.add_argument('--out-dir', default='sweep')
    ap.add_argument('--runs-csv', default='runs.csv')
    ap.add_argument('--results-csv', default='sweep_results.csv')
    ap.add_argument('--no-weights', action='store_true')
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true', help='solo listar las combinaciones')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grid = list(itertools.product(args.sizes, args.lrs, args.bases, args.powers, args.scales))
    total = len(grid)
    if args.num_shards > 1:
        grid = grid[args.shard::args.num_shards]

    print('=' * 78)
    if args.num_shards > 1:
        print('BARRIDO: particion %d de %d -> %d combinaciones (de %d en la rejilla)'
              % (args.shard, args.num_shards, len(grid), total))
    else:
        print('BARRIDO: %d combinaciones' % len(grid))
    print('  sizes  : %s' % args.sizes)
    print('  lrs    : %s' % args.lrs)
    print('  bases  : %s' % args.bases)
    print('  powers : %s' % args.powers)
    print('  scales : %s  (1.0 = sin recorte)' % args.scales)
    print('  epocas : %d  (parada temprana tras %d sin mejorar)' % (args.epochs, args.early_stop))
    print('=' * 78)
    for i, (s, lr, b, p, sc) in enumerate(grid, 1):
        print('  %2d/%d  size=%d lr=%.0e base=%d power=%.2f scale=%.2f' % (i, len(grid), s, lr, b, p, sc))
    if args.dry_run:
        return
    print('=' * 78, flush=True)

    results = []
    t_total = time.time()
    for i, (s, lr, b, p, sc) in enumerate(grid, 1):
        tag = 's%d_lr%g_b%d_p%.2f_sc%.2f' % (s, lr, b, p, sc)
        print('\n' + '#' * 78)
        print('# COMBINACION %d/%d  ->  %s' % (i, len(grid), tag))
        print('#' * 78, flush=True)
        a = build_args(args, (s, s), lr, b, p, sc, tag)
        t0 = time.time()
        try:
            r = train(a)
            estado = 'ok'
        except Exception:
            traceback.print_exc()
            r = {'best_miou': float('nan'), 'best_epoch': -1,
                 'class_iou': [float('nan')] * len(CLASS_NAMES)}
            estado = 'ERROR'
        mins = (time.time() - t0) / 60.0
        results.append(dict(zip(FIELDS,
            [tag, '%dx%d' % (s, s), lr, b, p, sc, int(not args.no_augment),
             args.epochs, r['best_epoch'], r['best_miou']]
            + list(r['class_iou']) + [round(mins, 2), estado])))
        _write(args.results_csv, results)
        print('\n>>> %s : mIoU %.4f (epoca %s, %.1f min) [%s]'
              % (tag, r['best_miou'], r['best_epoch'], mins, estado), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _report(results, (time.time() - t_total) / 60.0, args.results_csv)


def _write(path, results):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)


def _nan(v):
    return v != v


def _report(results, mins, path):
    ok = [r for r in results if not _nan(r['best_miou'])]
    ok.sort(key=lambda r: -r['best_miou'])
    short = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']

    print('\n\n' + '=' * 100)
    print('RESULTADOS DEL BARRIDO  (%d combinaciones, %.1f minutos)' % (len(results), mins))
    print('=' * 100)
    hdr = '%-26s %-8s %6s %5s %7s ' % ('combinacion', 'size', 'power', 'scale', 'mIoU')
    hdr += ' '.join('%6s' % s for s in short) + '  %4s' % 'ep'
    print(hdr)
    print('-' * len(hdr))
    for r in ok:
        line = '%-26s %-8s %6.2f %5.2f %7.4f ' % (
            r['combo'], r['size'], r['weight_power'], r['scale_min'], r['best_miou'])
        line += ' '.join('%6.4f' % r['iou_' + c] if not _nan(r['iou_' + c]) else '   nan'
                         for c in CLASS_NAMES)
        print(line + '  %4s' % r['best_epoch'])
    for r in results:
        if _nan(r['best_miou']):
            print('%-26s  [%s]' % (r['combo'], r['estado']))

    if ok:
        b = ok[0]
        print('-' * len(hdr))
        print('GANADORA: %s  ->  mIoU %.4f  (epoca %s)' % (b['combo'], b['best_miou'], b['best_epoch']))
        print('  modelo : %s' % os.path.join(os.path.dirname(path) or '.',
                                              'model_%s.th' % b['combo']))
        print('  por clase: ' + ',  '.join('%s %.4f' % (c, b['iou_' + c]) for c in CLASS_NAMES))
    print('  tabla completa en %s' % path)
    print('=' * 100)


if __name__ == '__main__':
    main()
