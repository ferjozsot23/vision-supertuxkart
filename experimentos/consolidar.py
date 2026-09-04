"""Une los sweep_results*.csv de todas las particiones y ordena por mIoU.

    python3 consolidar.py                 # todos los sweep_results*.csv del cwd
    python3 consolidar.py --glob 'sweep_results_g*.csv'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob as globmod

from utils import CLASS_NAMES


RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

SHORT = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default=os.path.join(RESULTADOS, 'sweep_results*.csv'))
    ap.add_argument('--out', default=os.path.join(RESULTADOS, 'barrido_consolidado.csv'))
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit('no encontre ningun fichero que case con %r' % args.glob)

    rows, repetidas = [], {}
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r['_origen'] = os.path.basename(f)
                rows.append(r)

    porcombo = {}
    for r in rows:
        porcombo.setdefault(r['combo'], []).append(r)
    repetidas = {k: v for k, v in porcombo.items() if len(v) > 1}
    rows = [max(v, key=lambda r: fnum(r['best_miou'])) for v in porcombo.values()]

    ok = [r for r in rows if fnum(r['best_miou']) == fnum(r['best_miou'])]
    ok.sort(key=lambda r: -fnum(r['best_miou']))

    hdr = '%-30s %-8s %6s %5s %7s ' % ('combinacion', 'size', 'power', 'scale', 'mIoU')
    hdr += ' '.join('%6s' % s for s in SHORT) + '  %4s %6s' % ('ep', 'min')
    print('=' * len(hdr))
    print('BARRIDO CONSOLIDADO — %d combinaciones de %d fichero(s)' % (len(rows), len(files)))
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for r in ok:
        line = '%-30s %-8s %6.2f %5.2f %7.4f ' % (
            r['combo'], r['size'], fnum(r['weight_power']), fnum(r['scale_min']),
            fnum(r['best_miou']))
        line += ' '.join('%6.4f' % fnum(r['iou_' + c]) if fnum(r['iou_' + c]) == fnum(r['iou_' + c])
                         else '   nan' for c in CLASS_NAMES)
        print(line + '  %4s %6s' % (r['best_epoch'], r['minutos']))
    for r in rows:
        if fnum(r['best_miou']) != fnum(r['best_miou']):
            print('%-30s  [%s]' % (r['combo'], r.get('estado', '?')))

    if repetidas:
        print('-' * len(hdr))
        print('VARIANZA entre corridas de la MISMA configuracion:')
        for k, v in repetidas.items():
            vals = sorted(fnum(x['best_miou']) for x in v)
            print('  %-40s %s   (rango %.4f)'
                  % (k, ' / '.join('%.4f' % x for x in vals), vals[-1] - vals[0]))

    if ok:
        b = ok[0]
        print('-' * len(hdr))
        print('GANADORA: %s   mIoU %.4f  (epoca %s, %s)'
              % (b['combo'], fnum(b['best_miou']), b['best_epoch'], b['_origen']))
        print('  por clase: ' + ',  '.join('%s %.4f' % (c, fnum(b['iou_' + c])) for c in CLASS_NAMES))
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k in ok[0] if k != '_origen'],
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(ok)
        print('  tabla ordenada -> %s' % args.out)
    print('=' * len(hdr))


if __name__ == '__main__':
    main()
