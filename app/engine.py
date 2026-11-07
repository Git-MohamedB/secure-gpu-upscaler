"""Moteur d'inférence GPU Real-ESRGAN (x4plus anime 6B) : FP16 + tiling.

L'architecture RRDBNet est réimplémentée ici (compatible avec les clés des poids
officiels) afin d'éviter la dépendance à basicsr.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class UpscaleEngine:
    SCALE = 4
    TILE_PAD = 10  # recouvrement (px d'entrée) pour éviter les coutures

    def __init__(self, model_path: str, device: str = "cuda", half: bool = True):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais aucun GPU disponible.")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Poids introuvables : {path} (lancer scripts/download_weights.py)"
            )

        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"  # FP16 inutile/lent sur CPU

        model = RRDBNet(num_block=6)  # anime_6B
        # weights_only=True : pas d'exécution de code arbitraire via pickle
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        if self.half:
            model = model.half()
        self.model = model.to(self.device)

    @torch.inference_mode()
    def upscale_image(self, image_np: np.ndarray, tile_size: int = 512) -> np.ndarray:
        """Upscale x4 d'une image RGB uint8 (H, W, 3) par tuiles. Retourne uint8."""
        if image_np.ndim != 3 or image_np.shape[2] != 3 or image_np.dtype != np.uint8:
            raise ValueError("image_np doit être un tableau uint8 de forme (H, W, 3).")
        if tile_size < 16:
            raise ValueError("tile_size doit être >= 16.")

        dtype = torch.float16 if self.half else torch.float32
        img = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        img = (img.to(self.device).to(dtype)) / 255.0

        _, c, h, w = img.shape
        s, pad = self.SCALE, self.TILE_PAD
        output = torch.empty((1, c, h * s, w * s), dtype=dtype, device=self.device)

        try:
            for y0 in range(0, h, tile_size):
                for x0 in range(0, w, tile_size):
                    y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
                    # tuile élargie du padding, bornée à l'image
                    py0, py1 = max(y0 - pad, 0), min(y1 + pad, h)
                    px0, px1 = max(x0 - pad, 0), min(x1 + pad, w)

                    out_tile = self.model(img[:, :, py0:py1, px0:px1])

                    # retire le padding de la sortie
                    oy0, ox0 = (y0 - py0) * s, (x0 - px0) * s
                    oy1, ox1 = oy0 + (y1 - y0) * s, ox0 + (x1 - x0) * s
                    output[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = out_tile[:, :, oy0:oy1, ox0:ox1]
                    del out_tile

            result = (output.clamp_(0, 1) * 255.0).round().to(torch.uint8)
            return result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        finally:
            del img, output
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

