# Graphene architecture, adapted from SRFormer and SwinIR.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_


class _DWConvFFN(nn.Module):
    def __init__(self, hidden_features, kernel_size=5, dilation=1):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ConvFFN kernel_size must be a positive odd integer.")
        if dilation < 1:
            raise ValueError("ConvFFN dilation must be a positive integer.")
        padding = dilation * (kernel_size - 1) // 2
        self.hidden_features = hidden_features
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=hidden_features,
            ),
            nn.GELU(),
        )

    def forward(self, x, x_size):
        h, w = x_size
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, h, w).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):


    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, dilation=1, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = _DWConvFFN(
            hidden_features=hidden_features,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        if x_size is None:
            raise ValueError("ConvFFN requires x_size=(H, W).")
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):


    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):


    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):


    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.permuted_window_size = (window_size[0] // 2, window_size[1] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1),
                num_heads,
            )
        )

        coords_h = torch.arange(self.permuted_window_size[0])
        coords_w = torch.arange(self.permuted_window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size[1] - 1
        aligned_relative_position_index = relative_coords.sum(-1)
        aligned_relative_position_index = (
            aligned_relative_position_index.reshape(
                self.permuted_window_size[0],
                self.permuted_window_size[1],
                1,
                1,
                self.permuted_window_size[0] * self.permuted_window_size[1],
            )
            .repeat(1, 1, 2, 2, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(
                4 * self.permuted_window_size[0] * self.permuted_window_size[1],
                self.permuted_window_size[0] * self.permuted_window_size[1],
            )
        )
        self.register_buffer("aligned_relative_position_index", aligned_relative_position_index)

        self.kv = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, return_logits: bool = False):
        b_, n, c = x.shape
        kv = (
            self.kv(x)
            .reshape(
                b_,
                self.permuted_window_size[0],
                2,
                self.permuted_window_size[1],
                2,
                2,
                c // 4,
            )
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(b_, n // 4, 2, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        q = (
            self.q(x)
            .reshape(b_, n, 1, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.permuted_window_size[0] * self.permuted_window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n // 4) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n // 4)
            logits = attn
            attn = self.softmax(attn)
        else:
            logits = attn
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_logits:
            return x, logits
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.permuted_window_size}, num_heads={self.num_heads}"



class IsingRefinementBlock(nn.Module):


    def __init__(self, dim, window_size, cfg):
        super().__init__()
        self.cfg = cfg or {}
        self.K = int(self.cfg.get("K", 3))
        self.signed = bool(self.cfg.get("signed", True))
        self.symmetric = bool(self.cfg.get("symmetric", True))
        self.topk = int(self.cfg.get("topk", 0))

        beta0 = float(self.cfg.get("beta", 1.0))
        gamma0 = float(self.cfg.get("gamma", 1.0))
        gate0 = float(self.cfg.get("gate_scale", 1.0))
        refine_ratio = float(self.cfg.get("refine_ratio", 1.0))

        # Graphene-lightweight learns a per-IRB residual scalar initialized to zero.
        # The full model applies the gated residual without this extra scalar.
        self.residual_scale = None
        if "residual_scale" in self.cfg or bool(
            self.cfg.get("learn_residual_scale", False)
        ):
            residual_scale0 = float(self.cfg.get("residual_scale", 1.0))
            if bool(self.cfg.get("learn_residual_scale", False)):
                self.residual_scale = nn.Parameter(torch.tensor(residual_scale0))
            else:
                self.register_buffer(
                    "fixed_residual_scale", torch.tensor(residual_scale0)
                )

        if bool(self.cfg.get("learn_beta", False)):
            self.beta = nn.Parameter(torch.tensor(beta0))
        else:
            self.register_buffer("beta", torch.tensor(beta0))

        if bool(self.cfg.get("learn_gamma", False)):
            self.gamma = nn.Parameter(torch.tensor(gamma0))
        else:
            self.register_buffer("gamma", torch.tensor(gamma0))

        self.register_buffer("gate_scale", torch.tensor(gate0))

        if isinstance(window_size, tuple):
            window_h, window_w = window_size
        else:
            window_h = window_w = int(window_size)
        if window_h % 2 != 0 or window_w % 2 != 0:
            raise ValueError("IsingRefinementBlock requires even window sizes.")

        coarse_w = window_w // 2
        lift_index = []
        for row in range(window_h):
            for col in range(window_w):
                lift_index.append((row // 2) * coarse_w + (col // 2))
        self.register_buffer("lift_index", torch.tensor(lift_index, dtype=torch.long))

        self.group_area = 4.0
        self.to_h = nn.Linear(dim, 1)
        self.to_m0 = nn.Linear(dim, 1)

        hidden = max(dim, int(dim * refine_ratio))
        self.refine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def _apply_topk(self, J):
        if self.topk <= 0 or self.topk >= J.size(-1):
            return J
        absJ = J.abs()
        _, idx = torch.topk(absJ, k=self.topk, dim=-1)
        mask = torch.zeros_like(J, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return J.masked_fill(~mask, 0.0)

    def _lift_coarse_matrix(self, J):
        return J.index_select(-1, self.lift_index) / self.group_area

    def _lift_pair_mask(self, pair_mask):
        return pair_mask.index_select(-1, self.lift_index)

    def forward(self, x, logits, pair_mask=None):
        h = self.to_h(x)
        m = torch.tanh(self.to_m0(x))

        J = self._lift_coarse_matrix(logits.mean(dim=1))

        if pair_mask is not None:
            # pair_mask is batch-independent with shape
            # [num_windows, tokens, tokens]. Broadcast it over the image
            # batch instead of materializing one copy per image.
            num_windows = pair_mask.shape[0]
            if J.shape[0] % num_windows != 0:
                raise ValueError(
                    "Ising pair-mask windows do not divide the attention batch."
                )
            J_shape = J.shape
            J = J.view(
                J.shape[0] // num_windows,
                num_windows,
                J.shape[-2],
                J.shape[-1],
            )
            J = J.masked_fill(pair_mask.unsqueeze(0), 0.0).view(J_shape)

        J = J - J.mean(dim=-1, keepdim=True)

        if not self.signed:
            J = torch.relu(J)

        if self.symmetric:
            J = 0.5 * (J + J.transpose(-1, -2))

        J = J.clone()
        J.diagonal(dim1=-2, dim2=-1).zero_()

        J = self._apply_topk(J)
        J = self.gamma * J

        for _ in range(max(self.K, 1)):
            m = torch.tanh(self.beta * (h + torch.bmm(J, m)))

        gate = torch.sigmoid(self.gate_scale * m)
        refinement = gate * self.refine(x)
        if self.residual_scale is not None:
            refinement = self.residual_scale * refinement
        elif hasattr(self, "fixed_residual_scale"):
            refinement = self.fixed_residual_scale * refinement
        return x + refinement


class GrapheneBlock(nn.Module):


    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ising_cfg=None,
        convffn_kernel_size=5,
        convffn_dilation=1,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if self.window_size % 2 != 0 or self.shift_size % 2 != 0:
            raise ValueError("GrapheneBlock requires even window_size and shift_size.")
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.permuted_window_size = self.window_size // 2
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.ising = None
        if isinstance(ising_cfg, dict) and bool(ising_cfg.get("enable", False)):
            self.ising = IsingRefinementBlock(dim, self.window_size, ising_cfg)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            kernel_size=convffn_kernel_size,
            dilation=convffn_dilation,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        if attn_mask is not None and self.ising is not None:
            ising_pair_mask = self.ising._lift_pair_mask(attn_mask).lt(0)
        else:
            ising_pair_mask = None
        # Derived from attn_mask and lift_index; never serialize it.
        self.register_buffer(
            "ising_pair_mask",
            ising_pair_mask,
            persistent=False,
        )

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        permuted_window_mask = torch.zeros((1, H // 2, W // 2, 1))
        h_slices = (
            slice(0, -self.permuted_window_size),
            slice(-self.permuted_window_size, -self.shift_size // 2),
            slice(-self.shift_size // 2, None),
        )
        w_slices = (
            slice(0, -self.permuted_window_size),
            slice(-self.permuted_window_size, -self.shift_size // 2),
            slice(-self.shift_size // 2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                permuted_window_mask[:, h, w, :] = cnt
                cnt += 1

        permuted_windows = window_partition(permuted_window_mask, self.permuted_window_size)
        permuted_windows = permuted_windows.view(-1, self.permuted_window_size * self.permuted_window_size)
        attn_mask = mask_windows.unsqueeze(2) - permuted_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, _, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            mask = self.attn_mask
        else:
            mask = self.calculate_mask(x_size).to(x.device)

        if self.ising is None:
            attn_windows = self.attn(x_windows, mask=mask)
        else:
            attn_windows, logits = self.attn(x_windows, mask=mask, return_logits=True)
            pair_mask = None
            if mask is not None:
                if self.input_resolution == x_size:
                    pair_mask = self.ising_pair_mask
                else:
                    pair_mask = self.ising._lift_pair_mask(mask).lt(0)
            attn_windows = self.ising(attn_windows, logits, pair_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )



class GrapheneStageBody(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        ising_cfg=None,
        ising_schedule=None,
        convffn_kernel_size=5,
        convffn_dilations=None,
        shift_sizes=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        if convffn_dilations is None:
            convffn_dilations = [1] * depth
        if len(convffn_dilations) != depth:
            raise ValueError(
                "convffn_dilations must contain one value per block: "
                f"expected {depth}, got {len(convffn_dilations)}."
            )
        convffn_dilations = [int(value) for value in convffn_dilations]
        if any(value < 1 for value in convffn_dilations):
            raise ValueError("All ConvFFN dilation values must be positive integers.")
        if shift_sizes is None:
            shift_sizes = [0 if i % 2 == 0 else window_size // 2 for i in range(depth)]
        if len(shift_sizes) != depth:
            raise ValueError(
                "shift_sizes must contain one value per block: "
                f"expected {depth}, got {len(shift_sizes)}."
            )
        shift_sizes = [int(value) for value in shift_sizes]
        if any(value < 0 or value >= window_size or value % 2 != 0 for value in shift_sizes):
            raise ValueError(
                "All shift_sizes must be even integers in [0, window_size)."
            )
        if ising_schedule is None:
            enabled = isinstance(ising_cfg, dict) and bool(
                ising_cfg.get("enable", False)
            )
            ising_schedule = [enabled] * depth
        if len(ising_schedule) != depth:
            raise ValueError(
                "ising_schedule must contain one value per block: "
                f"expected {depth}, got {len(ising_schedule)}."
            )
        ising_schedule = [bool(value) for value in ising_schedule]

        self.blocks = nn.ModuleList(
            [
                GrapheneBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_sizes[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    ising_cfg=ising_cfg if ising_schedule[i] else None,
                    convffn_kernel_size=convffn_kernel_size,
                    convffn_dilation=convffn_dilations[i],
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"



class GrapheneStage(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection='1conv',
        ising_cfg=None,
        ising_schedule=None,
        convffn_kernel_size=5,
        convffn_dilations=None,
        shift_sizes=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = GrapheneStageBody(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            ising_cfg=ising_cfg,
            ising_schedule=ising_schedule,
            convffn_kernel_size=convffn_kernel_size,
            convffn_dilations=convffn_dilations,
            shift_sizes=shift_sizes,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x



class PatchEmbed(nn.Module):


    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x



class PatchUnEmbed(nn.Module):


    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x



class Upsample(nn.Sequential):


    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):


    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class HistoryAttention(nn.Module):
    def __init__(self, dim, num_stages, init_scale=0.0, recent_bias=6.0):
        super().__init__()
        self.dim = dim
        self.num_stages = num_stages
        self.norm = nn.LayerNorm(dim)
        self.eta_s = nn.Parameter(torch.zeros(num_stages, dim))
        self.lambda_s = nn.Parameter(torch.full((num_stages,), float(init_scale)))
        self.rho = float(recent_bias)
        self.recent_bias = self.rho

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        legacy_query_key = prefix + "query"
        legacy_mix_key = prefix + "mix"
        if legacy_query_key in state_dict and prefix + "eta_s" not in state_dict:
            state_dict[prefix + "eta_s"] = state_dict.pop(legacy_query_key)
        if legacy_mix_key in state_dict and prefix + "lambda_s" not in state_dict:
            state_dict[prefix + "lambda_s"] = state_dict.pop(legacy_mix_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, history, stage_idx):
        if len(history) == 1:
            return history[-1]
        values = torch.stack(history, dim=0)
        keys = self.norm(values)
        eta_s = self.eta_s[stage_idx].type_as(keys)
        logits = torch.einsum('c,nbtc->nbt', eta_s, keys) / math.sqrt(self.dim)
        bias = torch.zeros_like(logits)
        bias[-1] = self.rho
        w_s = torch.softmax(logits + bias, dim=0)
        selected_history = torch.einsum('nbt,nbtc->btc', w_s, values)
        lambda_s = self.lambda_s[stage_idx].type_as(selected_history)
        return history[-1] + lambda_s * (selected_history - history[-1])


class _GrapheneBase(nn.Module):


    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=32, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super(_GrapheneBase, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.historyattention = None


        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)


        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio


        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)


        self.layers = nn.ModuleList()
        self.norm = norm_layer(self.num_features)


        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':

            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))


        if self.upsampler == 'pixelshuffle':

            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':

            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':

            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:

            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if self.historyattention is None:
            for layer in self.layers:
                x = layer(x, x_size)
        else:
            history = [x]
            for i, layer in enumerate(self.layers):
                x = layer(x, x_size)
                history.append(x)
                if i < len(self.layers) - 1:
                    x = self.historyattention(history, i)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        if getattr(self, 'fp32_io', False) and torch.is_autocast_enabled(x.device.type):
            return self._forward_fp32_io(x)
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':

            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':

            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':

            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:

            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def _forward_fp32_io(self, x):
        """BF16 encoder kernels with FP32 stem, reconstruction, and RGB output."""
        H, W = x.shape[2:]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.check_image_size(x.float())
            self.mean = self.mean.to(device=x.device, dtype=torch.float32)
            shallow = self.conv_first((x - self.mean) * self.img_range)
        deep = self.forward_features(shallow)
        with torch.autocast(device_type=x.device.type, enabled=False):
            features = self.conv_after_body(deep.float()) + shallow
            if self.upsampler == 'pixelshuffledirect':
                sr = self.upsample(features)
            elif self.upsampler == 'pixelshuffle':
                sr = self.conv_last(self.upsample(self.conv_before_upsample(features)))
            else:
                raise ValueError('FP32 input/output layers require a PixelShuffle head.')
            sr = sr / self.img_range + self.mean
        return sr[:, :, :H*self.upscale, :W*self.upscale]



class Graphene(_GrapheneBase):


    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=32,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            **kwargs,
        )

        ising_cfg = kwargs.get("ising_cfg", None)
        if ising_cfg is None:
            ising_cfg = {
                "enable": True,
                "K": 3,
                "beta": 1.0,
                "learn_beta": False,
                "gamma": 1.0,
                "learn_gamma": False,
                "signed": True,
                "symmetric": True,
                "topk": 0,
                "gate_scale": 1.0,
                "refine_ratio": 1.0,
            }
        convffn_kernel_size = kwargs.get('convffn_kernel_size', 5)
        convffn_dilation_schedule = kwargs.get('convffn_dilation_schedule', None)
        if convffn_dilation_schedule is None:
            convffn_dilation_schedule = [[1] * depth for depth in depths]
        if len(convffn_dilation_schedule) != self.num_layers:
            raise ValueError(
                "convffn_dilation_schedule must contain one list per stage: "
                f"expected {self.num_layers}, got {len(convffn_dilation_schedule)}."
            )
        normalized_dilation_schedule = []
        for stage_idx, (stage_schedule, depth) in enumerate(
            zip(convffn_dilation_schedule, depths)
        ):
            if len(stage_schedule) != depth:
                raise ValueError(
                    "convffn_dilation_schedule stage "
                    f"{stage_idx} must contain {depth} values, "
                    f"got {len(stage_schedule)}."
                )
            stage_schedule = [int(value) for value in stage_schedule]
            if any(value < 1 for value in stage_schedule):
                raise ValueError(
                    "All convffn_dilation_schedule values must be positive integers."
                )
            normalized_dilation_schedule.append(stage_schedule)
        shift_schedule = kwargs.get('shift_schedule', None)
        if shift_schedule is None:
            shift_schedule = [
                [0 if i % 2 == 0 else window_size // 2 for i in range(depth)]
                for depth in depths
            ]
        if len(shift_schedule) != self.num_layers:
            raise ValueError(
                "shift_schedule must contain one list per stage: "
                f"expected {self.num_layers}, got {len(shift_schedule)}."
            )
        normalized_shift_schedule = []
        for stage_idx, (stage_schedule, depth) in enumerate(zip(shift_schedule, depths)):
            if len(stage_schedule) != depth:
                raise ValueError(
                    "shift_schedule stage "
                    f"{stage_idx} must contain {depth} values, got {len(stage_schedule)}."
                )
            stage_schedule = [int(value) for value in stage_schedule]
            if any(value < 0 or value >= window_size or value % 2 != 0 for value in stage_schedule):
                raise ValueError(
                    "All shift_schedule values must be even integers in [0, window_size)."
                )
            normalized_shift_schedule.append(stage_schedule)
        historyattention_cfg = kwargs.get('historyattention_cfg', kwargs.get('historyattention', None)) or {}
        ising_schedule = kwargs.get('ising_schedule', None)
        if ising_schedule is None:
            ising_enabled = isinstance(ising_cfg, dict) and bool(
                ising_cfg.get("enable", False)
            )
            ising_schedule = [
                [ising_enabled] * depth for depth in depths
            ]
        if len(ising_schedule) != self.num_layers:
            raise ValueError(
                "ising_schedule must contain one list per stage: "
                f"expected {self.num_layers}, got {len(ising_schedule)}."
            )
        normalized_ising_schedule = []
        for stage_idx, (stage_schedule, depth) in enumerate(
            zip(ising_schedule, depths)
        ):
            if len(stage_schedule) != depth:
                raise ValueError(
                    "ising_schedule stage "
                    f"{stage_idx} must contain {depth} values, "
                    f"got {len(stage_schedule)}."
                )
            normalized_ising_schedule.append(
                [bool(value) for value in stage_schedule]
            )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = GrapheneStage(
                dim=embed_dim,
                input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                ising_cfg=ising_cfg,
                ising_schedule=normalized_ising_schedule[i_layer],
                convffn_kernel_size=convffn_kernel_size,
                convffn_dilations=normalized_dilation_schedule[i_layer],
                shift_sizes=normalized_shift_schedule[i_layer],
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        if bool(historyattention_cfg.get('enable', False)):
            self.historyattention = HistoryAttention(
                embed_dim,
                num_stages=max(self.num_layers - 1, 1),
                init_scale=float(historyattention_cfg.get('init_scale', 0.0)),
                recent_bias=float(historyattention_cfg.get('recent_bias', 6.0)),
            )
        else:
            self.historyattention = None
        self.layers.apply(self._init_weights)
        if self.historyattention is not None:
            self.historyattention.apply(self._init_weights)
        self.norm.apply(self._init_weights)
