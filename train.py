"""Loop de entrenamiento para la segmentacion semantica de SuperTuxKart.

Disenado para correr en el DGX sin modificaciones y sin nadie mirando:
  - imprime la tabla de IoU por clase cada epoca (unico canal de resultados)
  - guarda checkpoint CADA epoca y soporta --resume
  - rutas explicitas, nunca '~' (el contenedor del DGX es efimero y $HOME=/root)

Uso tipico en el servidor:
    tmux new -s stk
    python train.py --data-root /ruta/en/servidor --epochs 40 --size 128 128 \
        2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

from models import SegModel, save_model
from utils import (CLASS_NAMES, NUM_CLASSES, VAL_TRACKS, ConfusionMatrix,
                   STKSegmentationDataset, compute_class_weights,
                   format_confusion, format_iou_table, load_data, split_tracks)


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    total_loss, nb = 0.0, 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        total_loss += float(criterion(logits, mask))
        nb += 1
        cm.add(logits.argmax(dim=1), mask)
    return cm, (total_loss / max(nb, 1))


def train(args):
    device = pick_device(args.device)
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.abspath(args.out)
    last_ckpt = os.path.join(ckpt_dir, 'last.th')

    print('device       :', device)
    print('torch        :', torch.__version__)
    print('data-root    :', os.path.abspath(args.data_root))
    print('tracks train :', split_tracks(args.data_root, 'train'))
    print('tracks val   :', split_tracks(args.data_root, 'val'))
    print('size         :', tuple(args.size))
    print('out          :', out_path)
    print('ckpt-dir     :', ckpt_dir)

    train_loader = load_data(args.data_root, 'train', size=tuple(args.size),
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             augment=not args.no_augment, limit=args.limit,
                             scale_min=args.scale_min)
    val_loader = load_data(args.data_root, 'val', size=tuple(args.size),
                           batch_size=args.batch_size, num_workers=args.num_workers,
                           augment=False, limit=args.limit)
    print('batches      : train=%d val=%d' % (len(train_loader), len(val_loader)))

    if args.no_weights:
        weights = None
        print('class_weights: DESACTIVADOS')
    else:
        ds = STKSegmentationDataset(args.data_root, split_tracks(args.data_root, 'train'),
                                    size=tuple(args.size))
        weights = compute_class_weights(ds, cache_path=args.weights,
                                        power=args.weight_power,
                                        clip=args.weight_clip).to(device)
        print('class_weights:', [round(float(v), 4) for v in weights])

    model = SegModel(num_classes=NUM_CLASSES, base=args.base).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4)

    start_epoch, best_miou = 0, -1.0
    best_epoch, best_iou = -1, None
    since_best = 0
    if args.resume is not None:
        resume_path = args.resume if args.resume and os.path.isfile(args.resume) else last_ckpt
        if os.path.isfile(resume_path):
            ck = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            start_epoch = ck['epoch'] + 1
            best_miou = ck.get('best_miou', -1.0)
            print('resume desde %s (epoca %d, best_miou %.4f)'
                  % (resume_path, start_epoch, best_miou))
        else:
            print('--resume pedido pero no hay checkpoint en %s; empiezo de cero' % resume_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), mask)
            loss.backward()
            optimizer.step()
            run_loss += float(loss.detach())
            nb += 1
        train_loss = run_loss / max(nb, 1)

        cm, val_loss = evaluate(model, val_loader, device, criterion)
        miou = cm.miou
        scheduler.step(miou)
        lr_now = optimizer.param_groups[0]['lr']

        print()
        print(format_iou_table(
            cm.class_iou, miou,
            extra={'acc_global': '%.4f' % cm.global_accuracy,
                   'loss_train': '%.4f' % train_loss,
                   'loss_val': '%.4f' % val_loss,
                   'lr': '%.1e' % lr_now,
                   'seg': '%.1f' % (time.time() - t0)},
            title='EPOCH %d | VAL' % epoch))

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_miou': best_miou, 'args': vars(args)}, last_ckpt)
        if miou == miou and miou > best_miou:
            best_miou, best_epoch = miou, epoch
            best_iou = [float(v) for v in cm.class_iou]
            since_best = 0
            save_model(model, out_path)
            print('nuevo mejor mIoU %.4f -> %s' % (best_miou, out_path))
        else:
            since_best += 1
            if args.early_stop and since_best >= args.early_stop:
                print('\nparada temprana: %d epocas sin mejorar (mejor: epoca %d, %.4f)'
                      % (since_best, best_epoch, best_miou))
                break

        _log_run(args, epoch, train_loss, miou, cm.class_iou, lr_now)
        last_cm = cm

    print()
    print(format_confusion(last_cm))
    print('\nFIN. mejor mIoU de validacion: %.4f en la epoca %d  (modelo en %s)'
          % (best_miou, best_epoch, out_path))
    return {'best_miou': best_miou, 'best_epoch': best_epoch,
            'class_iou': best_iou or [float('nan')] * NUM_CLASSES,
            'out': out_path}


def _log_run(args, epoch, train_loss, miou, class_iou, lr):
    path = os.path.abspath(args.runs_csv)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['timestamp', 'size', 'lr', 'base', 'epoch', 'weights_scheme',
                        'augment', 'train_loss', 'val_miou']
                       + ['iou_' + c for c in CLASS_NAMES] + ['notes'])
        w.writerow([datetime.now().isoformat(timespec='seconds'),
                    'x'.join(str(s) for s in args.size), '%.2e' % lr, args.base, epoch,
                    'none' if args.no_weights else 'p%.2f' % args.weight_power,
                    '%d/%.2f' % (int(not args.no_augment), args.scale_min),
                    '%.4f' % train_loss, '%.4f' % miou]
                   + ['%.4f' % float(v) if float(v) == float(v) else 'nan' for v in class_iou]
                   + [args.notes])


def main():
    ap = argparse.ArgumentParser(description='Entrena la U-Net de segmentacion de SuperTuxKart')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--size', type=int, nargs=2, default=[128, 128], metavar=('H', 'W'))
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None, help='recorta el dataset (smoke test)')
    ap.add_argument('--out', default='model.th')
    ap.add_argument('--ckpt-dir', default='checkpoints')
    ap.add_argument('--resume', nargs='?', const='', default=None)
    ap.add_argument('--weights', default='class_weights.json')
    ap.add_argument('--no-weights', action='store_true')
    ap.add_argument('--weight-power', type=float, default=0.5,
                    help='0.5 = 1/raiz(freq) (suave), 1.0 = 1/freq (agresivo)')
    ap.add_argument('--weight-clip', type=float, default=10.0)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--scale-min', type=float, default=0.6,
                    help='escala minima del recorte aleatorio; 1.0 lo desactiva')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--runs-csv', default='runs.csv')
    ap.add_argument('--notes', default='')
    ap.add_argument('--early-stop', type=int, default=0,
                    help='corta tras N epocas sin mejorar el mIoU; 0 lo desactiva')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
