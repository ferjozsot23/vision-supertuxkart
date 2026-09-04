"""U-Net para segmentacion semantica de SuperTuxKart.

Este archivo esta separado de train.py a proposito: permite hacer
`from models import load_model` sin disparar ningun entrenamiento.

Solo depende de torch. Nada de aqui usa APIs introducidas despues de torch 1.4,
para que el .th entrenado en el DGX cargue en cualquier maquina.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 7


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegModel(nn.Module):
    DOWNSAMPLE = 16

    def __init__(self, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        b = base
        self.enc1 = DoubleConv(3, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std

        h, w = x.shape[-2:]
        ph = (-h) % self.DOWNSAMPLE
        pw = (-w) % self.DOWNSAMPLE
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.head(d1)
        return out[..., :h, :w]


def save_model(model, path='model.th'):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    state = {k.replace('module.', ''): v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save(state, path)
    return path


def infer_hparams(state):
    base = state['enc1.block.0.weight'].shape[0]
    num_classes = state['head.weight'].shape[0]
    return int(base), int(num_classes)


def load_model(path='model.th', device='cpu'):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state and 'state_dict' not in state:
        state = state['model']
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    base, num_classes = infer_hparams(state)
    model = SegModel(num_classes=num_classes, base=base)
    model.load_state_dict(state)
    return model.to(device).eval()


if __name__ == '__main__':
    m = SegModel()
    m.eval()
    for hw in [(96, 128), (128, 128), (211, 333), (400, 400), (97, 130), (480, 640)]:
        with torch.no_grad():
            y = m(torch.rand(2, 3, *hw))
        assert y.shape == (2, NUM_CLASSES, *hw), (hw, tuple(y.shape))
        print('in %-12s -> out %s  OK' % (str(hw), tuple(y.shape)))
    n = sum(p.numel() for p in m.parameters())
    print('OK  parametros: %d (%.2f M)' % (n, n / 1e6))

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ruta = os.path.join(tmp, 'prueba.th')
        save_model(m, ruta)
        m2 = load_model(ruta)
        with torch.no_grad():
            x = torch.rand(1, 3, 211, 333)
            assert torch.allclose(m(x), m2(x), atol=1e-6)
    print('save_model/load_model round-trip OK')
