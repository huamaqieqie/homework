import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
LEWM_ROOT = REPO_ROOT
sys.path.insert(0, str(LEWM_ROOT))


try:
    import einops  # noqa: F401
except ModuleNotFoundError:
    einops_stub = types.ModuleType("einops")

    def rearrange(x, pattern, **axes_lengths):
        if pattern == "b t d -> (b t) d":
            return x.reshape(x.size(0) * x.size(1), x.size(2))
        if pattern == "(b t) d -> b t d":
            batch = axes_lengths["b"]
            return x.reshape(batch, -1, x.size(-1))
        raise NotImplementedError(f"sanity fallback does not implement rearrange pattern: {pattern}")

    einops_stub.rearrange = rearrange
    sys.modules["einops"] = einops_stub

from jepa import JEPA  # noqa: E402


class TinyPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, c):
        return self.proj(x + 0.1 * c)


def make_model(mode, scale=1.0, dim=8):
    return JEPA(
        encoder=nn.Identity(),
        predictor=TinyPredictor(dim),
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        predictor_mode=mode,
        residual_scale=scale,
    )


def assert_finite_grads(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise AssertionError(f"non-finite gradient in {name}")


def main():
    torch.manual_seed(7)
    batch, steps, dim = 4, 3, 8
    z_t = torch.randn(batch, steps, dim)
    action = torch.randn(batch, steps, dim)
    target = torch.randn(batch, steps, dim)

    direct = make_model("direct", dim=dim)
    direct_out, direct_raw = direct.predict(z_t, action, return_raw=True)
    expected_direct = direct.predictor(z_t, action)
    if direct_out.shape != z_t.shape:
        raise AssertionError(f"direct output shape changed: {tuple(direct_out.shape)}")
    if not torch.allclose(direct_out, expected_direct):
        raise AssertionError("direct mode must return raw predictor output")
    if not torch.allclose(direct_raw, expected_direct):
        raise AssertionError("direct return_raw must match predictor output")

    residual_scale = 0.25
    residual = make_model("residual", scale=residual_scale, dim=dim)
    residual_out, residual_raw = residual.predict(z_t, action, return_raw=True)
    expected_residual = z_t + residual_scale * residual_raw
    if residual_out.shape != z_t.shape:
        raise AssertionError(f"residual output shape changed: {tuple(residual_out.shape)}")
    if not torch.allclose(residual_out, expected_residual):
        raise AssertionError("residual mode must compute z_t + residual_scale * delta_z")

    loss = F.mse_loss(residual_out, target)
    if not math.isfinite(float(loss.detach())):
        raise AssertionError("loss is not finite")
    loss.backward()
    assert_finite_grads(residual)

    print("PASS")


if __name__ == "__main__":
    main()
