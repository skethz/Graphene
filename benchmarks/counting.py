"""Params and FLOPs with fvcore (FLOPs = 2 x MACs) and explicit handlers for custom operators.

The handler formulas are the ones behind the published tables:

* Mamba selective scan: 9*B*L*C*N + B*C*L MACs (input u: B x C x L, A: C x N).
* PFT/IET sparse attention (SMM_QmK, SMM_AmV): index B x N x K times the feature width.
* NATTEN na2d_qk / na2d_av: the tables used prod(out) * in[-1] / out[-1]. That is
  B*heads*H*W*D for QK and B*heads*H*W*K^2 for AV, so the K^2 factor (QK) and
  the D factor (AV) are missing. ``natten_exact_macs`` reports the full
  B*heads*H*W*K^2*D for each call. The table formula stays the headline number
  so that it stays comparable with the paper; both are written to the JSON.
* scaled_dot_product_attention (UCAN): B*heads*Nq*Nk*(Dqk + Dv).
* Bicubic/bilinear upsampling: 16/4 MACs per output element (fvcore's
  default for bilinear is also 4).

Operators that do no multiply-accumulate work (elementwise, indexing,
softmax, norms of small tensors, ...) are listed in ``IGNORED_NON_MAC_OPS``.
They are reported but add nothing. Any other operator fvcore cannot count
makes the status ``needs_handler``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch

MACS_TO_FLOPS = 2

IGNORED_NON_MAC_OPS = {
    "aten::_weight_norm", "aten::abs", "aten::add", "aten::add_", "aten::arange", "aten::argmax",
    "aten::avg_pool2d", "aten::cat", "aten::ceil", "aten::clamp", "aten::clamp_", "aten::clamp_min",
    "aten::clone", "aten::contiguous", "aten::cos", "aten::cumsum", "aten::diagonal", "aten::div",
    "aten::div_", "aten::embedding", "aten::empty", "aten::empty_like", "aten::eq", "aten::exp", "aten::expand_as",
    "aten::exponential_", "aten::fill_", "aten::flip", "aten::floor", "aten::gelu", "aten::im2col",
    "aten::index", "aten::index_select", "aten::leaky_relu", "aten::leaky_relu_", "aten::linalg_vector_norm",
    "aten::linspace", "aten::log", "aten::log_softmax", "aten::lt", "aten::masked_fill", "aten::masked_fill_",
    "aten::max", "aten::max_pool2d", "aten::maximum", "aten::mean", "aten::meshgrid", "aten::min", "aten::minimum", "aten::mul",
    "aten::mul_", "aten::ne", "aten::neg", "aten::new_zeros", "aten::one_hot", "aten::ones_like", "aten::pad",
    "aten::pixel_shuffle", "aten::pow", "aten::prelu", "aten::rand_like", "aten::randperm", "aten::reciprocal",
    "aten::relu", "aten::relu_", "aten::repeat", "aten::reshape", "aten::roll", "aten::rsqrt", "aten::rsub",
    "aten::scatter", "aten::scatter_", "aten::searchsorted", "aten::sigmoid", "aten::silu", "aten::sin",
    "aten::softmax", "aten::sort", "aten::sqrt", "aten::stack", "aten::std", "aten::sub", "aten::sum",
    "aten::tanh", "aten::topk", "aten::transpose", "aten::type_as", "aten::unbind", "aten::unfold",
    "aten::unsqueeze", "aten::var", "aten::view", "aten::where", "aten::zero_", "aten::zeros",
    "aten::zeros_like",
}


def _sizes(value: Any) -> list[int] | None:
    try:
        sizes = value.type().sizes()
    except Exception:
        return None
    if sizes is None or any(size is None for size in sizes):
        return None
    return [int(size) for size in sizes]


def selective_scan_macs(inputs: list[Any], outputs: list[Any]) -> float:
    del outputs
    u = _sizes(inputs[0])
    a = _sizes(inputs[2])
    if not u or len(u) != 3 or not a or len(a) < 2:
        return 0.0
    batch, channels, length = u
    return float(9 * batch * length * channels * a[1] + batch * channels * length)


def sparse_macs(inputs: list[Any], outputs: list[Any]) -> float:
    del outputs
    left = _sizes(inputs[0])
    index = _sizes(inputs[2])
    if not left or not index:
        return 0.0
    # QK: A = B x N x D, index = B x N x K. AV: A = B x N x K, value = B x N x D.
    width = left[-1] if len(left) == 3 and left[-1] != index[-1] else (_sizes(inputs[1]) or [0])[-1]
    return float(index[0] * index[1] * index[2] * width)


def natten_table_macs(inputs: list[Any], outputs: list[Any]) -> float:
    inp = _sizes(inputs[0])
    out = _sizes(outputs[0])
    if not inp or not out or len(out) != 5:
        return 0.0
    width = inp[-1] if len(inp) == 5 else (_sizes(inputs[1]) or [0])[-1]
    return float(math.prod(out) * width / out[-1])


def natten_exact_macs(inputs: list[Any], outputs: list[Any]) -> float:
    inp = _sizes(inputs[0])
    out = _sizes(outputs[0])
    if not inp or not out or len(out) != 5 or len(inp) != 5:
        return 0.0
    # QK: out B,h,H,W,K^2 and in[-1] = D. AV: out B,h,H,W,D and in[-1] = K^2.
    return float(math.prod(out) * inp[-1])


def sdpa_macs(inputs: list[Any], outputs: list[Any]) -> float:
    del outputs
    q, k, v = (_sizes(inputs[i]) for i in range(3))
    if not q or not k or not v or len(q) < 2:
        return 0.0
    batch_heads = math.prod(q[:-2])
    return float(batch_heads * q[-2] * k[-2] * (q[-1] + v[-1]))


def upsample_macs(taps: int):
    def handle(inputs: list[Any], outputs: list[Any]) -> float:
        del inputs
        out = _sizes(outputs[0])
        return float(math.prod(out) * taps) if out else 0.0
    return handle


def trace_safe_batched_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
    """IET's batched_isin with the clamp bound as a Python int.

    Under jit tracing, tensor.size() is recorded as a CPU tensor and the official
    `torch.clamp(idx, max=test_elements.size(1) - 1)` fails on CUDA inputs. The function
    only builds an index mask (sort/searchsorted/gather, no MACs); used for counting only.
    """
    test_sorted, _ = test_elements.sort(dim=-1)
    idx = torch.searchsorted(test_sorted, elements, right=False)
    bound = int(test_elements.size(1))
    idx = torch.clamp(idx, max=bound - 1)
    gathered = torch.gather(test_sorted, 1, idx)
    return (gathered == elements) & (idx < bound)


def tracing_patches(method: str) -> list[str]:
    """Apply count-only patches that make an architecture traceable; returns their descriptions."""
    import sys

    if method in ("IET", "IET-light"):
        module = sys.modules["basicsr.archs.iet_arch"]
        module.batched_isin = trace_safe_batched_isin
        return ["iet_arch.batched_isin: clamp bound cast to int for jit tracing (index mask, no MACs)"]
    return []


NATTEN_OPS = (
    "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction",
    "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction",
    "prim::PythonOp.NattenQK",  # names of the pure-PyTorch stand-in used for the original tables
    "prim::PythonOp.NattenAV",
)
SELECTIVE_SCAN_OPS = (
    "prim::PythonOp.SelectiveScanFn", "prim::PythonOp.SelectiveScanMamba", "prim::PythonOp.SelectiveScanOflex",
    "prim::PythonOp.SelectiveScanCore", "prim::PythonOp.SelectiveScan", "prim::PythonOp.SelectiveScanCuda",
)


def count(model: torch.nn.Module, input_size: tuple[int, int], device: torch.device) -> dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    params = sum(p.numel() for p in model.parameters())
    natten_calls: list[dict[str, Any]] = []

    def natten_handle(inputs, outputs):
        table = natten_table_macs(inputs, outputs)
        natten_calls.append({"in": _sizes(inputs[0]), "out": _sizes(outputs[0]),
                             "table_macs": table, "exact_macs": natten_exact_macs(inputs, outputs)})
        return table

    handles: dict[str, Any] = {name: selective_scan_macs for name in SELECTIVE_SCAN_OPS}
    handles.update({name: natten_handle for name in NATTEN_OPS})
    handles.update({
        "prim::PythonOp.SMM_QmK": sparse_macs,
        "prim::PythonOp.SMM_AmV": sparse_macs,
        "aten::scaled_dot_product_attention": sdpa_macs,
        "aten::upsample_bicubic2d": upsample_macs(16),
        "aten::upsample_bilinear2d": upsample_macs(4),
    })

    # torch.utils.checkpoint in forward (ATD-light's official config) does not change the MACs;
    # it is switched off during tracing only.
    switched = [m for m in model.modules() if getattr(m, "use_checkpoint", False) is True]
    for module in switched:
        module.use_checkpoint = False
    try:
        dummy = torch.randn(1, 3, *input_size, device=device)
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, dummy).set_op_handle(**handles)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            macs = float(analysis.total())
            by_operator = {k: float(v) for k, v in analysis.by_operator().items()}
            unsupported = dict(analysis.unsupported_ops())
    finally:
        for module in switched:
            module.use_checkpoint = True

    ignored = {k: v for k, v in unsupported.items() if k in IGNORED_NON_MAC_OPS}
    unknown = {k: v for k, v in unsupported.items() if k not in IGNORED_NON_MAC_OPS}
    natten_table = sum(c["table_macs"] for c in natten_calls)
    natten_exact = sum(c["exact_macs"] for c in natten_calls)
    result: dict[str, Any] = {
        "params": params,
        "macs": macs,
        "flops": MACS_TO_FLOPS * macs,
        "flops_g": MACS_TO_FLOPS * macs / 1e9,
        "by_operator_macs": dict(sorted(by_operator.items())),
        "ignored_non_mac_ops": dict(sorted(Counter(ignored).items())),
        "unknown_ops": dict(sorted(Counter(unknown).items())),
        "status": "needs_handler" if unknown else "ok",
        "input_shape": [1, 3, *input_size],
        "use_checkpoint_switched_off_for_trace": len(switched),
    }
    if natten_calls:
        result["natten"] = {
            "calls": len(natten_calls),
            "table_formula_macs": natten_table,
            "exact_macs": natten_exact,
            "flops_g_with_exact_natten": MACS_TO_FLOPS * (macs - natten_table + natten_exact) / 1e9,
            "per_call": natten_calls,
        }
    return result
