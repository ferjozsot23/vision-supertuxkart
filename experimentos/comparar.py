"""Compara las corridas registradas en runs.csv.

    python3 comparar.py              # mejor epoca de cada corrida
    python3 comparar.py --todas      # todas las epocas

Una corrida = un grupo de filas con la misma combinacion de hiperparametros.
Se muestra la epoca de mejor mIoU de cada una, para poder mirar el IoU POR CLASE
y no solo el promedio (el mIoU esconde que nitro/bomb estan en 0 mientras track
esta en 0.9).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from collections import OrderedDict

from utils import CLASS_NAMES


RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

SHORT = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']


def read_runs(path):
    if not os.path.exists(path):
        raise SystemExit('No existe %s. Se descarga con: bash servidor/recoger.sh'
                         % path)
    with open(path) as fh:
        return list(csv.DictReader(fh))


def key(r):
    return (r['size'], r['base'], r['weights_scheme'], r['augment'], r['notes'])


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-csv', default=os.path.join(RESULTADOS, 'runs.csv'))
    ap.add_argument('--todas', action='store_true')
    args = ap.parse_args()

    rows = read_runs(args.runs_csv)
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    hdr = '%-11s %-5s %-4s %-3s %6s %7s %7s ' % ('corrida', 'size', 'base', 'aug', 'ep', 'loss_tr', 'mIoU')
    hdr += ' '.join('%6s' % s for s in SHORT)
    print(hdr)
    print('-' * len(hdr))

    best_overall = None
    for k, rs in groups.items():
        sel = rs if args.todas else [max(rs, key=lambda r: fnum(r['val_miou']))]
        for r in sel:
            miou = fnum(r['val_miou'])
            line = '%-11s %-5s %-4s %-3s %6s %7s %7.4f ' % (
                (r['notes'] or '-')[:11], r['size'], r['base'], r['augment'],
                r['epoch'], r['train_loss'], miou)
            line += ' '.join('%6s' % (r['iou_' + c] if r['iou_' + c] != 'nan' else '  nan')
                             for c in CLASS_NAMES)
            print(line)
            if best_overall is None or miou > fnum(best_overall['val_miou']):
                best_overall = r

    if best_overall:
        print()
        print('mejor mIoU: %.4f  (%s, size %s, epoca %s)'
              % (fnum(best_overall['val_miou']), best_overall['notes'] or '-',
                 best_overall['size'], best_overall['epoch']))
        print('loss_train de esa epoca: %s  <- comparalo con el loss_val del log'
              % best_overall['train_loss'])


if __name__ == '__main__':
    main()
