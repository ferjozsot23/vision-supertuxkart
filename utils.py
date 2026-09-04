"""Carga, emparejamiento y exploracion del dataset de SuperTuxKart.

Dependencias: torch, numpy, Pillow. Nada mas (no hay torchvision en todas
las maquinas donde esto corre).
"""

import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ['background', 'track', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
NUM_CLASSES = 7

VAL_TRACKS = ['volcano_island', 'lighthouse']

PALETTE = np.array([
    [  0,   0,   0],
    [128, 128, 128],
    [220,  40,  40],
    [ 40, 180,  90],
    [ 60, 130, 250],
    [250, 190,  40],
    [230,  60, 230],
], dtype=np.uint8)

_ID_RE = re.compile(r'(\d+)\.png$')


def _numeric_id(path):
    m = _ID_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def list_tracks(data_root):
    return sorted(d for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d)))


def list_samples(data_root, tracks):
    samples = []
    for track in tracks:
        mask_dir = os.path.join(data_root, track, 'combined')
        frame_dir = os.path.join(data_root, track, 'frame')
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError('no existe %s' % mask_dir)
        for name in sorted(os.listdir(mask_dir)):
            if not name.endswith('.png'):
                continue
            nid = _numeric_id(name)
            if nid is None:
                continue
            frame_path = os.path.join(frame_dir, 'frame_%s.png' % nid)
            if os.path.exists(frame_path):
                samples.append((frame_path, os.path.join(mask_dir, name)))
    return samples


def split_tracks(data_root, split):
    all_tracks = list_tracks(data_root)
    if split == 'all':
        return all_tracks
    val = [t for t in all_tracks if t in VAL_TRACKS]
    if split == 'val':
        return val
    if split == 'train':
        return [t for t in all_tracks if t not in VAL_TRACKS]
    raise ValueError("split debe ser 'train', 'val' o 'all'; llego %r" % split)


class STKSegmentationDataset(Dataset):
    def __init__(self, data_root, tracks, size=(128, 128), augment=False,
                 scale_min=0.6):
        self.data_root = data_root
        self.tracks = list(tracks)
        self.size = tuple(size) if size is not None else None
        self.augment = augment
        self.scale_min = float(scale_min)
        self.samples = list_samples(data_root, self.tracks)
        if not self.samples:
            raise RuntimeError('0 muestras para tracks=%r en %s' % (tracks, data_root))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, mask_path = self.samples[idx]

        img = Image.open(frame_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.augment and self.scale_min < 1.0:
            img, mask = self._random_crop(img, mask)

        if self.size is not None:
            h, w = self.size
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_np = np.asarray(img, dtype=np.uint8)
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        img_t = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float().div_(255.0)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))
        return img_t, mask_t

    def _random_crop(self, img, mask):
        w, h = img.size
        s = random.uniform(self.scale_min, 1.0)
        cw, ch = max(1, int(w * s)), max(1, int(h * s))
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        box = (x, y, x + cw, y + ch)
        return img.crop(box), mask.crop(box)

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        return img, mask


def load_data(data_root, split='train', size=(128, 128), batch_size=16,
              num_workers=4, augment=False, limit=None, shuffle=None,
              scale_min=0.6):
    dataset = STKSegmentationDataset(data_root, split_tracks(data_root, split),
                                     size=size, augment=augment, scale_min=scale_min)
    if limit is not None and limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(limit)))
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False,
                      pin_memory=torch.cuda.is_available())


def label_to_color(mask):
    mask = np.asarray(mask)
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    return PALETTE[np.clip(mask, 0, NUM_CLASSES - 1)]


def overlay(img_chw, mask_hw, alpha=0.55):
    img = (np.asarray(img_chw).transpose(1, 2, 0) * 255).astype(np.uint8)
    color = label_to_color(mask_hw)
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)


class ConfusionMatrix(object):
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self):
        self.matrix.zero_()

    def add(self, preds, labels):
        preds = preds.detach().reshape(-1).cpu()
        labels = labels.detach().reshape(-1).cpu()
        k = (labels >= 0) & (labels < self.num_classes)
        idx = self.num_classes * labels[k].to(torch.int64) + preds[k].to(torch.int64)
        self.matrix += torch.bincount(idx, minlength=self.num_classes ** 2) \
                            .reshape(self.num_classes, self.num_classes)

    @property
    def class_iou(self):
        m = self.matrix.double()
        tp = m.diag()
        denom = m.sum(dim=1) + m.sum(dim=0) - tp
        iou = torch.full((self.num_classes,), float('nan'), dtype=torch.float64)
        present = denom > 0
        iou[present] = tp[present] / denom[present]
        return iou.float()

    @property
    def miou(self):
        iou = self.class_iou
        valid = ~torch.isnan(iou)
        if valid.sum() == 0:
            return float('nan')
        return float(iou[valid].mean())

    @property
    def num_present(self):
        return int((~torch.isnan(self.class_iou)).sum())

    @property
    def global_accuracy(self):
        total = self.matrix.sum()
        if total == 0:
            return float('nan')
        return float(self.matrix.diag().sum().double() / total.double())


def format_iou_table(class_iou, miou, extra=None, title='VAL'):
    lines = ['=== %s ===' % title]
    for name, v in zip(CLASS_NAMES, list(class_iou)):
        v = float(v)
        lines.append('%-12s %s' % (name, 'nan' if v != v else '%.4f' % v))
    lines.append('-' * 25)
    present = int(sum(1 for v in class_iou if float(v) == float(v)))
    lines.append('%-12s %.4f   (%d clases presentes)'
                 % ('mIoU', miou, present) if miou == miou
                 else '%-12s nan' % 'mIoU')
    for k, v in (extra or {}).items():
        lines.append('%-12s %s' % (k, v))
    lines.append('=' * 25)
    return '\n'.join(lines)


def format_confusion(cm, top=3, min_pixels=1):
    m = cm.matrix.double()
    lines = ['--- a donde van los pixeles de cada clase real ---']
    for c, name in enumerate(CLASS_NAMES):
        total = float(m[c].sum())
        if total < min_pixels:
            lines.append('%-12s (no aparece en validacion)' % name)
            continue
        order = sorted(range(cm.num_classes), key=lambda j: -float(m[c, j]))[:top]
        parts = ['%s %.1f%%' % (CLASS_NAMES[j], 100.0 * float(m[c, j]) / total)
                 for j in order if float(m[c, j]) > 0]
        lines.append('%-12s %10d px  ->  %s' % (name, int(total), ',  '.join(parts)))
    return '\n'.join(lines)


def weights_from_counts(counts, power=0.5, clip=10.0):
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / counts.sum()
    w = (1.0 / (freq + 1e-9)) ** power
    w = w / w.mean()
    return np.clip(w, 0, clip)


def compute_class_weights(dataset, cache_path='class_weights.json', verbose=True,
                          power=0.5, clip=10.0):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            data = json.load(fh)
        if 'pixel_counts' in data:
            w = weights_from_counts(data['pixel_counts'], power, clip)
            if verbose:
                print('class_weights de %s (power=%.2f, clip=%.1f)' % (cache_path, power, clip))
            return torch.tensor(w, dtype=torch.float32)
        if verbose:
            print('class_weights leidos de %s' % cache_path)
        return torch.tensor(data['weights'], dtype=torch.float32)

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        counts += np.bincount(mask.numpy().ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        if verbose and (i + 1) % 200 == 0:
            print('  %d/%d frames contados' % (i + 1, len(dataset)))

    freq = counts / counts.sum()
    w = weights_from_counts(counts, power, clip)

    if cache_path:
        with open(cache_path, 'w') as fh:
            json.dump({'class_names': CLASS_NAMES,
                       'pixel_counts': counts.tolist(),
                       'pixel_pct': (100 * freq).round(6).tolist(),
                       'weights': w.round(6).tolist()}, fh, indent=2)
        if verbose:
            print('class_weights escritos en %s' % cache_path)
    return torch.tensor(w, dtype=torch.float32)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--size', type=int, nargs=2, default=[128, 128])
    ap.add_argument('--out', default='salidas/sanity_pairs.png')
    args = ap.parse_args()

    train_tracks = split_tracks(args.data_root, 'train')
    val_tracks = split_tracks(args.data_root, 'val')
    print('tracks train   ->', train_tracks)
    print('tracks val     ->', val_tracks)
    assert not (set(train_tracks) & set(val_tracks)), 'tracks compartidos entre splits'

    train_ds = STKSegmentationDataset(args.data_root, train_tracks, size=args.size)
    val_ds = STKSegmentationDataset(args.data_root, val_tracks, size=args.size)
    print('len(train_ds)  ->', len(train_ds))
    print('len(val_ds)    ->', len(val_ds))
    assert len(train_ds) > 0 and len(val_ds) > 0

    img, mask = train_ds[0]
    print('img.shape      ->', img.shape)
    print('img.dtype      ->', img.dtype)
    print('img.min/max    -> %.4f / %.4f' % (img.min(), img.max()))
    print('mask.shape     ->', mask.shape)
    print('mask.dtype     ->', mask.dtype)
    print('mask.unique()  ->', mask.unique().tolist())
    assert img.dtype == torch.float32 and img.shape[0] == 3
    assert mask.dtype == torch.int64 and mask.shape == img.shape[1:]

    seen = set()
    for ds in (train_ds, val_ds):
        for i in random.Random(0).sample(range(len(ds)), 200):
            seen.update(ds[i][1].unique().tolist())
    print('clases vistas en 400 muestras ->', sorted(seen))
    assert seen <= set(range(NUM_CLASSES)), 'etiqueta fuera de [0,6]: %r' % sorted(seen)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    picks = random.Random(1).sample(range(len(train_ds)), 5)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for col, i in enumerate(picks):
        im, mk = train_ds[i]
        track = os.path.basename(os.path.dirname(os.path.dirname(train_ds.samples[i][0])))
        axes[0, col].imshow(im.permute(1, 2, 0).numpy())
        axes[0, col].set_title('%s\n%s' % (track, os.path.basename(train_ds.samples[i][0])), fontsize=8)
        axes[1, col].imshow(label_to_color(mk.numpy()))
        axes[1, col].set_title('clases: %s' % mk.unique().tolist(), fontsize=8)
        axes[2, col].imshow(overlay(im.numpy(), mk.numpy()))
        axes[2, col].set_title('overlay', fontsize=8)
        for row in range(3):
            axes[row, col].axis('off')
    for row, lab in enumerate(['frame', 'mascara', 'overlay']):
        axes[row, 0].set_ylabel(lab)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print('grilla de sanidad ->', args.out)
