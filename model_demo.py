import gc
import cv2 as cv
import numpy as np
import einops
from skimage import feature

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy, auroc
import timm
import lightning as L
from fvcore.nn import FlopCountAnalysis, parameter_count

# BUG 5 FIX: guard the BNext import so the server can start even when the
# BNext package is not installed (MobileNet backbones do not need it).
try:
    from BNext.src.bnext import BNext
    _BNEXT_AVAILABLE = True
except ImportError:
    _BNEXT_AVAILABLE = False


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DeepfakeDetector(L.LightningModule):

    def __init__(
        self,
        num_classes: int,
        backbone: str = "BNext-T",
        freeze_backbone: bool = True,
        add_magnitude_channel: bool = True,
        add_fft_channel: bool = True,
        add_lbp_channel: bool = True,
        add_gabor_channel: bool = True,
        learning_rate: float = 1e-4,
        pos_weight: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch_outs = []
        self.backbone = backbone

        # ── Backbone ────────────────────────────────────────────────────────
        size_map = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
        if backbone in size_map:
            if not _BNEXT_AVAILABLE:
                raise ImportError(
                    "BNext package is not installed. "
                    "Install it or use a MobileNet backbone instead."
                )
            size = size_map[backbone]
            self.base_model = nn.ModuleDict({"module": BNext(num_classes=1000, size=size)})
            pretrained_state_dict = torch.load(
                f"pretrained/{size}_checkpoint.pth.tar", map_location="cpu"
            )
            self.base_model.load_state_dict(pretrained_state_dict)
            self.base_model = self.base_model.module
            self.inplanes = self.base_model.fc.in_features
            self.base_model.deactive_last_layer = True
            self.base_model.fc = nn.Identity()

        elif backbone == "MobileNetV3-Small":
            self.base_model = timm.create_model(
                "mobilenetv3_small_100",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat = self.base_model.forward_features(dummy)
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                self.inplanes = feat.shape[1]

        elif backbone == "MobileNetV2":
            self.base_model = timm.create_model(
                "mobilenetv2_100",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat = self.base_model.forward_features(dummy)
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                self.inplanes = feat.shape[1]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Feature channel flags ────────────────────────────────────────────
        self.add_magnitude_channel = bool(add_magnitude_channel)
        self.add_fft_channel = bool(add_fft_channel)
        self.add_lbp_channel = bool(add_lbp_channel)
        self.add_gabor_channel = bool(add_gabor_channel)
        self.new_channels = sum(
            [self.add_magnitude_channel, self.add_fft_channel,
             self.add_lbp_channel, self.add_gabor_channel]
        )

        self.pos_weight = pos_weight

        # ── Adapter ──────────────────────────────────────────────────────────
        if self.new_channels > 0:
            in_ch = 3 + self.new_channels
            self.adapter = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
                SELayer(channel=in_ch),
                nn.Conv2d(in_ch, 3, kernel_size=1),
            )
        else:
            self.adapter = nn.Identity()

        # ── Freeze backbone (optional) ───────────────────────────────────────
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # ── Classification head ──────────────────────────────────────────────
        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)

        self.save_hyperparameters()

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, 3, H, W)  — plain RGB input from preprocessing
        if self.new_channels > 0:
            x = self.add_new_channels(x)       # → (B, 3+new_channels, H, W)

        x_adapted = self.adapter(x)            # → (B, 3, H, W)

        # ImageNet normalisation
        mean = torch.as_tensor(
            timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device
        ).view(1, -1, 1, 1)
        std = torch.as_tensor(
            timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device
        ).view(1, -1, 1, 1)
        x_adapted = (x_adapted - mean) / std

        if "MobileNet" in self.backbone:
            features = self.base_model.forward_features(x_adapted)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            features = self.base_model(x_adapted)

        return {"logits": self.fc(features)}

    # ── Optimiser ────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        modules_to_train = [self.adapter, self.fc]
        if not self.freeze_backbone:
            modules_to_train.append(self.base_model)
        optimizer = optim.AdamW(
            [p for m in modules_to_train for p in m.parameters()],
            lr=self.learning_rate,
        )
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=5
        )
        return [optimizer], [scheduler]

    # ── Feature extraction helpers ───────────────────────────────────────────
    def _add_new_channels_worker(self, image: torch.Tensor) -> torch.Tensor:
        """image: (H, W, 3) float32 tensor in [0, 1]"""
        gray = cv.cvtColor(
            (image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_RGB2GRAY
        )

        new_channels = []

        if self.add_magnitude_channel:
            sx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=7)
            sy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=7)
            new_channels.append(np.sqrt(sx ** 2 + sy ** 2))

        if self.add_fft_channel:
            fft_mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9)
            new_channels.append(fft_mag)

        if self.add_lbp_channel:
            lbp = feature.local_binary_pattern(gray, P=3, R=6, method="uniform")
            new_channels.append(lbp)

        if self.add_gabor_channel:
            gabor_responses = []
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                kern = cv.getGaborKernel(
                    (15, 15), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F
                )
                fimg = cv.filter2D(gray, cv.CV_32F, kern)
                gabor_responses.append(np.abs(fimg))
            new_channels.append(np.max(gabor_responses, axis=0))

        stacked = np.stack(new_channels, axis=2) / 255.0   # (H, W, new_channels)
        return torch.from_numpy(stacked).to(self.device).float()

    def add_new_channels(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W)
        images_hwc = einops.rearrange(images, "b c h w -> b h w c")
        new_ch = torch.stack(
            [self._add_new_channels_worker(img) for img in images_hwc], dim=0
        )                                                   # (B, H, W, new_channels)
        combined = torch.cat([images_hwc, new_ch], dim=-1) # (B, H, W, 3+new_channels)
        return einops.rearrange(combined, "b h w c -> b c h w")

    # ── Lightning hooks ───────────────────────────────────────────────────────
    def on_train_start(self):        self._on_start()
    def on_test_start(self):         self._on_start()
    def on_train_epoch_start(self):  self._on_epoch_start()
    def on_test_epoch_start(self):   self._on_epoch_start()
    def on_train_epoch_end(self):    self._on_epoch_end()
    def on_test_epoch_end(self):     self._on_epoch_end()

    def training_step(self, batch, i_batch):   return self._step(batch, i_batch, phase="train")
    def validation_step(self, batch, i_batch): return self._step(batch, i_batch, phase="val")
    def test_step(self, batch, i_batch):       return self._step(batch, i_batch, phase="test")

    def _step(self, batch, i_batch, phase=None):
        images = batch["image"].to(self.device)
        labels = batch["is_real"][:, 0].float().to(self.device)

        outs = self(images)
        outs["phase"] = phase
        outs["labels"] = labels

        if self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(
                input=outs["logits"][:, 0],
                target=labels,
                pos_weight=torch.as_tensor(self.pos_weight, device=self.device),
            )
        else:
            raise NotImplementedError("Only binary classification is implemented.")

        for k in list(outs.keys()):
            if isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()

        log_dict = {"loss": loss.detach().cpu()}
        if phase in {"train", "val"}:
            log_dict["learning_rate"] = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {f"{phase}_{k}": v for k, v in log_dict.items()},
            prog_bar=False, logger=True,
        )

        self.epoch_outs.append(outs)
        return loss

    def _on_start(self):
        with torch.no_grad():
            flops = FlopCountAnalysis(self, torch.randn(1, 3, 224, 224, device=self.device))
            parameters = parameter_count(self)[""]
            self.log_dict(
                {"flops": flops.total(), "parameters": parameters},
                prog_bar=True, logger=True,
            )

    def _on_epoch_start(self):
        self._clear_memory()
        self.epoch_outs = []

    def _on_epoch_end(self):
        self._clear_memory()
        with torch.no_grad():
            labels = torch.cat([b["labels"] for b in self.epoch_outs], dim=0)
            logits = torch.cat([b["logits"] for b in self.epoch_outs], dim=0)[:, 0]
            phases = [p for b in self.epoch_outs for p in [b["phase"]] * len(b["labels"])]
            assert len(labels) == len(logits)
            for phase in ["train", "val", "test"]:
                idx = [i for i, p in enumerate(phases) if p == phase]
                if not idx:
                    continue
                metrics = {
                    "acc": accuracy(preds=logits[idx], target=labels[idx], task="binary", average="micro"),
                    "auc": auroc(preds=logits[idx], target=labels[idx].long(), task="binary", average="micro"),
                }
                self.log_dict(
                    {f"{phase}_{k}": v for k, v in metrics.items()
                     if isinstance(v, (torch.Tensor, int, float))},
                    prog_bar=True, logger=True,
                )

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DeepfakeDetector(
        num_classes=2,
        backbone="MobileNetV3-Small",
        freeze_backbone=False,
        add_magnitude_channel=False,
        add_fft_channel=True,
        add_lbp_channel=True,
        add_gabor_channel=False,
    )
    out = model(torch.randn(2, 3, 224, 224))
    print("logits shape:", out["logits"].shape)   # expected: (2, 1)