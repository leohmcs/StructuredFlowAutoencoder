"""Flow-matching (FM) baseline for pinwheel (paper Table 1 / Figure 1).

Plain, unconditional flow matching on x. It differs from SFA in one thing only: the network is
SFA's pinwheel vector field `vt` (the same LLK "mlp" network as runner_continuous.PinWheelRunner,
5 x model.ngf hidden units), trained with the same flow-matching loss and probability path
(sfa.FlowMatchingLoss: t=0 data, t=1 noise) and sampled with the same ODE solver, but with the
latent z fixed at 0. There is no posterior, so only the generative metrics are reported.

Registered for pinwheel only: on the other datasets SFA's vector field is a different network, so
this module would no longer be "SFA without z".
"""
import torch

from sfa_llk import LLK
from benchmarks.common import BaselineModule, BaselineRunner


class FMModule(BaselineModule):
    name = "fm"

    def __init__(self, config, args, spec):
        super().__init__(config, args, spec)
        self.sig_min = float(config.flow.sig_min)
        self.vt = LLK(spec.x_dim, 1, model="mlp", hidden_features=[config.model.ngf] * 5,
                      hidden_dim=spec.x_dim)

    def _z(self, x):
        return x.new_zeros(len(x), 1)

    def loss(self, x):
        t = torch.rand(len(x), device=x.device, dtype=x.dtype)
        x1 = torch.randn_like(x)
        tt = t[:, None]
        xt = (1 - tt) * x + (self.sig_min + (1 - self.sig_min) * tt) * x1
        ut = (1 - self.sig_min) * x1 - x
        loss = (self.vt(t, xt, self._z(x)) - ut).square().mean()
        return loss, loss, {}

    def posterior(self, x):
        return None

    @torch.no_grad()
    def generate(self, n):
        x1 = torch.randn(n, self.spec.x_dim, device=self.device, dtype=next(self.parameters()).dtype)
        return self.vt.decode(x1, self._z(x1))


class PinWheelRunner(BaselineRunner):
    dataset, module_cls = "pinwheel", FMModule
