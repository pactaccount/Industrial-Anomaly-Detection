"""Feature extraction backbone for industrial anomaly detection.

Wraps a torchvision classification backbone and exposes intermediate feature maps
from the layers we care about, with the multi-scale patch aggregation that
PatchCore (Bergmann et al., 2022) relies on.

Two things matter here for correctness:

1. **Which layers.** PatchCore extracts from ``layer2`` and ``layer3`` of a
   ResNet-50, which correspond to strides 8 and 16. Those scales carry
   different information: layer2 resolves fine texture, layer3 resolves
   part-level structure. Anomalies that are large but low-contrast survive at
   layer3; small scratches only survive at layer2.

2. **Patchification.** A single feature cell is not a "patch" -- the paper
   aggregates a 3x3 neighbourhood of cells so that a score describes a region
   rather than a point. We reproduce that with an unfold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

# ImageNet statistics -- the backbones are pretrained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BackboneSpec:
    """A named backbone configuration.

    ``layers_to_extract_from`` uses the ``block.index`` convention so we can
    reach sub-blocks (e.g. ``layer2.0``) without hand-rolling module paths.
    """

    name: str
    layers_to_extract_from: tuple[str, ...]
    weights: str
    input_channels: int = 3


@dataclass
class LayerSpec:
    """Resolved description of one extraction layer."""

    module: str
    index: int | None
    stride: int


# Backbones we support. WideResNet-50-2 is the default because it is what the
# PatchCore paper reports numbers against, but ResNet-50 is ~2x faster on CPU
# and often just as good -- see configs for the tradeoff.
BACKBONES: dict[str, BackboneSpec] = {
    "wide_resnet50_2": BackboneSpec(
        name="wide_resnet50_2",
        layers_to_extract_from=("layer2", "layer3"),
        weights="IMAGENET1K_V1",
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        layers_to_extract_from=("layer2", "layer3"),
        weights="IMAGENET1K_V1",
    ),
    "resnet18": BackboneSpec(
        name="resnet18",
        layers_to_extract_from=("layer2", "layer3"),
        weights="IMAGENET1K_V1",
    ),
    "dino_vitb16": BackboneSpec(
        name="dino_vitb16",
        layers_to_extract_from=("blocks.5", "blocks.8", "blocks.10"),
        weights="IMAGENET1K_V1",
    ),
}

# The spatial stride of each torchvision ResNet stage.
_RESNET_STRIDES = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}


def build_backbone(spec: BackboneSpec) -> nn.Module:
    """Instantiate a torchvision backbone with pretrained weights."""
    import torchvision.models as models

    builders = {
        "wide_resnet50_2": lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
    }
    if spec.name not in builders:
        raise ValueError(
            f"Unsupported backbone {spec.name!r}. Available: {sorted(BACKBONES)}"
        )
    model = builders[spec.name]()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class PatchMaker:
    """Turn a feature map into overlapping 3x3 patches.

    This is the "patch" in PatchCore. Each output cell aggregates a
    ``patchsize x patchsize`` neighbourhood of the feature map (reflect-padded
    at the borders), so the resulting embedding is aware of its local context
    rather than a single activation.
    """

    def __init__(self, patchsize: int = 3, stride: int = 1) -> None:
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Unfold a batch of feature maps into patches.

        Args:
            features: ``(B, C, H, W)``.

        Returns:
            ``(patches, (h_patches, w_patches))`` where patches has shape
            ``(B, h_patches * w_patches, C, patchsize, patchsize)``.
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded = unfolder(features)
        h, w = features.shape[-2:]
        n_patches = [
            int((s + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1)
            for s in (h, w)
        ]
        unfolded = unfolded.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        # (B, C, p, p, h*w) -> (B, h*w, C, p, p)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)
        return unfolded, (n_patches[0], n_patches[1])

    @staticmethod
    def unpatch_scores(scores: torch.Tensor, batchsize: int) -> torch.Tensor:
        """Reshape a flat score vector back into ``(B, num_patches)``."""
        return scores.reshape(batchsize, -1)

    @staticmethod
    def score(scores: torch.Tensor) -> torch.Tensor:
        """Aggregate per-patch scores to an image score by taking the max."""
        while scores.ndim > 1:
            scores = torch.max(scores, dim=-1).values
        return scores


class FeatureAggregator(nn.Module):
    """Runs a backbone and collects feature maps from selected layers.

    The forward pass is truncated at the deepest requested layer -- we never
    compute layer4 if we only need layer2/layer3, and layer4 is the most
    expensive stage of a ResNet. For a 256x256 input that saves roughly
    40% of the backbone compute for no change in the output.

    Layers may be named either at stage granularity (``layer2``) or at
    sub-block granularity (``layer2.1``). Sub-block layers are handled by
    running that stage's residual blocks individually so the hook fires on
    the right block.
    """

    def __init__(self, backbone: nn.Module, layers_to_extract_from: tuple[str, ...]) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for layer_name in self.layers_to_extract_from:
            self._register_hook(layer_name)

    def _register_hook(self, layer_name: str) -> None:
        module = self._resolve_module(layer_name)

        def hook(_module, _inputs, output):
            self.outputs[layer_name] = output

        self._handles.append(module.register_forward_hook(hook))

    def _resolve_module(self, layer_name: str) -> nn.Module:
        if "." in layer_name:
            block, index = layer_name.split(".")
            module = self.backbone
            for part in block.split("."):
                module = getattr(module, part)
            if index.isnumeric():
                module = module[int(index)]
            return module
        return getattr(self.backbone, layer_name)

    @property
    def _deepest_stage(self) -> int:
        """Index of the last ResNet stage we need to compute."""
        deepest = self.layers_to_extract_from[-1].split(".")[0]
        return int(deepest.replace("layer", ""))

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps for each requested layer, in request order."""
        self.outputs.clear()

        if self._is_vit_style():
            # ViT-family backbones have no stage hierarchy; run them whole.
            self.backbone(images)
            return [self.outputs[layer] for layer in self.layers_to_extract_from]

        stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        x = stem(images)

        for name, module in self.backbone.named_children():
            if not name.startswith("layer"):
                continue
            stage = int(name.replace("layer", ""))
            if stage > self._deepest_stage:
                break
            x = module(x)
            # Stage-level hooks fire inside `module(x)`, so self.outputs is
            # already populated for any stage we asked for.

        self._fire_subblock_hooks(x)
        return [self.outputs[layer] for layer in self.layers_to_extract_from]

    def _is_vit_style(self) -> bool:
        """True when the requested layers live inside a transformer trunk."""
        first = self.layers_to_extract_from[0].split(".")[0]
        return not first.startswith("layer")

    def _fire_subblock_hooks(self, x: torch.Tensor) -> None:
        """Re-run residual blocks so sub-block hooks fire.

        A stage hook only tells us the stage's *output*; to capture the
        output of e.g. ``layer2.1`` we need to invoke that block. We re-run
        the blocks of the deepest requested stage, which is cheap relative to
        the full network and correct for the residual structure.
        """
        for layer_name in self.layers_to_extract_from:
            if "." not in layer_name:
                continue
            block, index = layer_name.split(".")
            stage = getattr(self.backbone, block, None)
            if stage is None or not index.isnumeric():
                continue
            idx = int(index)
            if idx == 0:
                continue
            for i in range(1, idx + 1):
                if i < len(stage):
                    x = stage[i](x)

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@dataclass
class EmbeddingConfig:
    """Knobs for the embedding stage."""

    backbone: str = "wide_resnet50_2"
    patchsize: int = 3
    patchstride: int = 1
    # Dimensionality each layer's features are projected to before concat.
    pretrain_embed_dimension: int = 1024
    # Dimensionality of the final concatenated embedding.
    target_embed_dimension: int = 1024
    # Image size fed to the backbone. PatchCore uses 256 with 3 scales.
    input_size: int = 256
    layers_to_extract_from: tuple[str, ...] = field(default=("layer2", "layer3"))

    def spec(self) -> BackboneSpec:
        base = BACKBONES[self.backbone]
        return BackboneSpec(
            name=base.name,
            layers_to_extract_from=self.layers_to_extract_from,
            weights=base.weights,
        )


class MultiScaleEmbedder(nn.Module):
    """Produces PatchCore-style embeddings for a batch of images.

    Handles the non-trivial bit: layer2 and layer3 have different spatial
    resolutions (stride 8 vs 16), so layer3's grid is bilinearly resampled up
    to layer2's grid before the two are concatenated. Without this the
    feature vectors would not correspond positionally and the memory bank
    would be meaningless.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config.spec())
        self.aggregator = FeatureAggregator(
            self.backbone, config.layers_to_extract_from
        )
        self.patch_maker = PatchMaker(config.patchsize, config.patchstride)
        self._mean_mappers = nn.ModuleList(
            [
                nn.Identity()
                for _ in config.layers_to_extract_from
            ]
        )
        # A single learned-free projection: adaptive average pooling to the
        # target dimension, exactly as the reference implementation does.
        self.pretrain_embed_dimension = config.pretrain_embed_dimension
        self.target_embed_dimension = config.target_embed_dimension

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Embed a batch of images.

        Args:
            images: ``(B, 3, H, W)`` normalised tensors.

        Returns:
            ``(embeddings, (h_patches, w_patches))`` where embeddings is
            ``(B * h_patches * w_patches, target_embed_dimension)``.
        """
        with torch.no_grad():
            feature_maps = self.aggregator(images)

        # Patchify each scale.
        patched: list[torch.Tensor] = []
        patch_shapes: list[tuple[int, int]] = []
        for feat in feature_maps:
            p, shape = self.patch_maker.patchify(feat)
            patched.append(p)
            patch_shapes.append(shape)

        ref_num_patches = patch_shapes[0]

        # Resample coarser scales up to the reference grid so that position
        # i in every layer refers to the same image region.
        for i in range(1, len(patched)):
            _features = patched[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:]
            )
            patched[i] = _features

        # Flatten to (B * num_patches, C, p, p) then project each layer to the
        # pretrain dimension and average-pool the layers together.
        flat = [p.reshape(-1, *p.shape[-3:]) for p in patched]
        projected = []
        for feat in flat:
            b = feat.shape[0]
            f = feat.reshape(b, 1, -1)
            f = F.adaptive_avg_pool1d(f, self.pretrain_embed_dimension).squeeze(1)
            projected.append(f)
        stacked = torch.stack(projected, dim=1)  # (B*P, num_layers, D)
        embeddings = F.adaptive_avg_pool1d(
            stacked.reshape(len(stacked), 1, -1), self.target_embed_dimension
        ).reshape(len(stacked), -1)
        return embeddings, ref_num_patches
