import torch.nn.functional as F
#torch dataset should contain tensor already at the point the dataloader passes data to model 

#---- LATE FUSION -----
#for late fusion input channels will be 100
import torch
import torch.nn as nn

CILP_EMB_SIZE = 200
BATCH_SIZE = 32

class Embedder(nn.Module):
    def __init__(self, in_ch, emb_size=CILP_EMB_SIZE):
        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb)

# Please do no alter this file. That will make it harder to pass the assessment!
class Classifier(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        kernel_size = 3
        n_classes = 1
        self.embedder = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def get_embs(self, imgs):
        return self.embedder(imgs)
    
    def forward(self, raw_data=None, data_embs=None):
        assert (raw_data is not None or data_embs is not None), "No images or embeddings given."
        if raw_data is not None:
            data_embs = self.get_embs(raw_data)
        return self.classifier(data_embs)

    
# ---- Fusion models ----

class LateNet(nn.Module):
    """Late fusion: independent RGB + LiDAR encoders; embeddings joined before classification.

    RGB input:   (B, 3,  64, 64)
    LiDAR input: (B, 4,  64, 64)  – XYZA channels
    """
    def __init__(self, emb_size: int = CILP_EMB_SIZE):
        super().__init__()
        self.rgb_enc   = Embedder(in_ch=3, emb_size=emb_size)
        self.lidar_enc = Embedder(in_ch=4, emb_size=emb_size)
        self.head = nn.Sequential(
            nn.Linear(emb_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),   # raw logit
        )

    def forward(self, x_rgb: torch.Tensor, x_xyza: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.rgb_enc(x_rgb), self.lidar_enc(x_xyza)], dim=1)
        return self.head(emb).squeeze(1)


# ---- Intermediate fusion helper encoders ----

class IntermediateBranchEncoder(nn.Module):
    """Per-modality branch encoder for intermediate fusion.

    Maps (B, in_ch, 64, 64) → (B, 100, 16, 16) via two conv + downsample blocks:
        Conv(in_ch→50) → ReLU → down → Conv(50→100) → ReLU → down

    strided=False (default): downsamples with MaxPool2d(2).
    strided=True:            downsamples with Conv2d(C, C, 3, stride=2, padding=1).
    """
    def __init__(self, in_ch: int, strided: bool = False):
        super().__init__()
        k = 3
        if strided:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch,  50, k, padding=1), nn.ReLU(), nn.Conv2d( 50,  50, k, stride=2, padding=1),
                nn.Conv2d(    50, 100, k, padding=1), nn.ReLU(), nn.Conv2d(100, 100, k, stride=2, padding=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch,  50, k, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(    50, 100, k, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedConvEncoder(nn.Module):
    """Post-fusion shared encoder for intermediate fusion.

    Maps (B, in_ch, H, W) → (B, in_ch, H/2, W/2) via one conv + downsample block:
        Conv(in_ch→in_ch) → ReLU → down

    strided=False (default): downsamples with MaxPool2d(2).
    strided=True:            downsamples with Conv2d(C, C, 3, stride=2, padding=1).
    """
    def __init__(self, in_ch: int, strided: bool = False):
        super().__init__()
        k = 3
        if strided:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, padding=1), nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, k, stride=2, padding=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- Intermediate fusion models ----
# Each base class accepts a single strided flag that switches both the branch
# encoders and the shared post-fusion encoder simultaneously:
#
#   Class name          strided
#   <Base>              False    ← default (MaxPool everywhere)
#   <Base>Strided       True     (strided convs everywhere)

class CatIntermediateNet(nn.Module):
    """Intermediate fusion via channel-wise concatenation of parallel feature maps.

    Flow:
        RGB   path:  IntermediateBranchEncoder(rgb_ch)   → (B, 100, 16, 16)
        LiDAR path:  IntermediateBranchEncoder(lidar_ch) → (B, 100, 16, 16)
        Merge:       cat along channel axis               → (B, 200, 16, 16)
        Shared:      SharedConvEncoder(200)               → (B, 200,  8,  8)
        Head:        Flatten → FC → FC → logit
    """
    def __init__(self, rgb_ch: int = 3, lidar_ch: int = 4, strided: bool = False):
        super().__init__()
        self.rgb_conv    = IntermediateBranchEncoder(rgb_ch,   strided=strided)
        self.lidar_conv  = IntermediateBranchEncoder(lidar_ch, strided=strided)
        self.shared_conv = SharedConvEncoder(200, strided=strided)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * 8 * 8, 1000), nn.ReLU(),
            nn.Linear(1000, 100),          nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x_rgb: torch.Tensor, x_xyza: torch.Tensor) -> torch.Tensor:
        f_rgb   = self.rgb_conv(x_rgb)              # (B, 100, 16, 16)
        f_lidar = self.lidar_conv(x_xyza)           # (B, 100, 16, 16)
        x = torch.cat([f_rgb, f_lidar], dim=1)      # (B, 200, 16, 16)  — merge
        x = self.shared_conv(x)                     # (B, 200,  8,  8)  — joint processing
        return self.head(x).squeeze(1)

class CatIntermediateNetStrided(CatIntermediateNet):
    """Cat fusion – strided convs everywhere (no MaxPool)."""
    def __init__(self, rgb_ch: int = 3, lidar_ch: int = 4):
        super().__init__(rgb_ch, lidar_ch, strided=True)


class MatmulIntermediateNet(nn.Module):
    """Intermediate fusion via matrix multiplication of parallel feature maps.

    Flow:
        RGB   path:  IntermediateBranchEncoder(rgb_ch)   → (B, 100, 16, 16)
        LiDAR path:  IntermediateBranchEncoder(lidar_ch) → (B, 100, 16, 16)
        Merge:       torch.matmul                         → (B, 100, 16, 16)
        Shared:      SharedConvEncoder(100)               → (B, 100,  8,  8)
        Head:        Flatten → FC → FC → logit

    matmul fuses the spatial dimensions, allowing each spatial position in one
    modality to interact with every position in the other (within each channel).
    """
    def __init__(self, rgb_ch: int = 3, lidar_ch: int = 4, strided: bool = False):
        super().__init__()
        self.rgb_conv    = IntermediateBranchEncoder(rgb_ch,   strided=strided)
        self.lidar_conv  = IntermediateBranchEncoder(lidar_ch, strided=strided)
        self.shared_conv = SharedConvEncoder(100, strided=strided)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100 * 8 * 8, 1000), nn.ReLU(),
            nn.Linear(1000, 100),          nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x_rgb: torch.Tensor, x_xyza: torch.Tensor) -> torch.Tensor:
        f_rgb   = self.rgb_conv(x_rgb)              # (B, 100, 16, 16)
        f_lidar = self.lidar_conv(x_xyza)           # (B, 100, 16, 16)
        x = torch.matmul(f_rgb, f_lidar)            # (B, 100, 16, 16)  — merge
        x = self.shared_conv(x)                     # (B, 100,  8,  8)  — joint processing
        return self.head(x).squeeze(1)

class MatmulIntermediateNetStrided(MatmulIntermediateNet):
    """Matmul fusion – strided convs everywhere (no MaxPool)."""
    def __init__(self, rgb_ch: int = 3, lidar_ch: int = 4):
        super().__init__(rgb_ch, lidar_ch, strided=True)


class AddIntermediateNet(nn.Module):
    """Intermediate fusion via element-wise addition of parallel feature maps.

    Flow:
        RGB   path:  IntermediateBranchEncoder(rgb_ch)   → (B, 100, 16, 16)
        LiDAR path:  IntermediateBranchEncoder(lidar_ch) → (B, 100, 16, 16)
        Merge:       element-wise addition                → (B, 100, 16, 16)
        Shared:      SharedConvEncoder(100)               → (B, 100,  8,  8)
        Head:        Flatten → FC → FC → logit

    Addition preserves the channel dimension (unlike cat) and is parameter-free,
    making it the lightest fusion operation.
    """
    def __init__(self, rgb_ch: int = 3, lidar_ch: int = 4, strided: bool = False):
        super().__init__()
        self.rgb_conv    = IntermediateBranchEncoder(rgb_ch,   strided=strided)
        self.lidar_conv  = IntermediateBranchEncoder(lidar_ch, strided=strided)
        self.shared_conv = SharedConvEncoder(100, strided=strided)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100 * 8 * 8, 1000), nn.ReLU(),
            nn.Linear(1000, 100),          nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x_rgb: torch.Tensor, x_xyza: torch.Tensor) -> torch.Tensor:
        f_rgb   = self.rgb_conv(x_rgb)              # (B, 100, 16, 16)
        f_lidar = self.lidar_conv(x_xyza)           # (B, 100, 16, 16)
        x = f_rgb + f_lidar                         # (B, 100, 16, 16)  — merge
        x = self.shared_conv(x)                     # (B, 100,  8,  8)  — joint processing
        return self.head(x).squeeze(1)

class AddIntermediateNetStrided(AddIntermediateNet):
    """Add fusion – strided convs everywhere (no MaxPool)."""
    def __init__(self, rgb_ch: int = 3, lidar_ch: int = 4):
        super().__init__(rgb_ch, lidar_ch, strided=True)


# contrastive pre training

"""
Ah I think just take notebook for this
class ContrastivePretraining(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_embedder = Embedder(4).to(device)
        lidar_embedder = Embedder(1).to(device)
        self.img_embedder = img_embedder
        self.lidar_embedder = lidar_embedder
        self.cos = nn.FIXME()

    def forward(self, rgb_imgs, lidar_depths):
        img_emb = self.img_embedder(rgb_imgs)
        lidar_emb = self.lidar_embedder(lidar_depths)

        repeated_img_emb = img_emb.FIXME(len(img_emb), dim=0)
        repeated_lidar_emb = lidar_emb.FIXME(len(lidar_emb), 1)

        similarity = self.cos(repeated_img_emb, repeated_lidar_emb)
        similarity = torch.unflatten(similarity, 0, (BATCH_SIZE, BATCH_SIZE))
        similarity = (similarity + 1) / 2

        logits_per_img = similarity
        logits_per_lidar = similarity.T
        return logits_per_img, logits_per_lidar
        """


# ============================================================
# ABLATION – Late fusion: Strided-convolution variants (MaxPool2d → Conv2d stride=2)
# The intermediate fusion strided variants live alongside their base classes above.
# ============================================================

class EmbedderStrided(nn.Module):
    """Drop-in replacement for Embedder with strided convs instead of MaxPool."""
    def __init__(self, in_ch, emb_size=CILP_EMB_SIZE):
        super().__init__()
        kernel_size = 3

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, 3, stride=2, padding=1),       # replaces MaxPool2d(2)
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, stride=2, padding=1),     # replaces MaxPool2d(2)
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, stride=2, padding=1),     # replaces MaxPool2d(2)
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, stride=2, padding=1),     # replaces MaxPool2d(2)
            nn.Flatten()
        )

        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb)


class LateNetStrided(nn.Module):
    """Late fusion with strided convs instead of MaxPool (ablation of LateNet).

    RGB input:   (B, 3,  64, 64)
    LiDAR input: (B, 4,  64, 64)  – XYZA channels
    """
    def __init__(self, emb_size: int = CILP_EMB_SIZE):
        super().__init__()
        self.rgb_enc   = EmbedderStrided(in_ch=3, emb_size=emb_size)
        self.lidar_enc = EmbedderStrided(in_ch=4, emb_size=emb_size)
        self.head = nn.Sequential(
            nn.Linear(emb_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_rgb: torch.Tensor, x_xyza: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([self.rgb_enc(x_rgb), self.lidar_enc(x_xyza)], dim=1)
        return self.head(emb).squeeze(1)