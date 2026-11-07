"""MC dropout and frozen encoders must not touch BatchNorm statistics. Runs in a subprocess with the real torch,
because conftest.py stubs torch for the rest of the suite."""
import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent('''
    import sys
    sys.path.insert(0, "src")
    import torch, torch.nn as nn
    from train_utils import keep_frozen_batchnorm_eval, mc_dropout_predict

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
            self.drop = nn.Dropout2d(0.5)
            self.head = nn.Conv2d(4, 2, 1)
        def forward(self, x):
            return {"out": self.head(self.drop(self.enc(x)))}

    torch.manual_seed(0)
    m = Net()
    m.eval()
    bn = m.enc[1]
    before = (bn.running_mean.clone(), bn.running_var.clone(), int(bn.num_batches_tracked))
    x = torch.randn(1, 3, 8, 8) * 5 + 3
    probs, unc = mc_dropout_predict(m, x, n_passes=5)
    assert torch.equal(bn.running_mean, before[0]) and torch.equal(bn.running_var, before[1])
    assert int(bn.num_batches_tracked) == before[2], "MC dropout updated BatchNorm statistics"
    assert not m.training and not m.drop.training, "previous (eval) mode not restored"
    assert unc.shape == (1, 8, 8) and float(unc.std()) > 0, "dropout was not active"

    m.train()
    mc_dropout_predict(m, x, n_passes=2)
    assert m.training and m.drop.training and bn.training, "previous (train) mode not restored"

    for p in m.enc.parameters():
        p.requires_grad_(False)
    m.train()
    assert keep_frozen_batchnorm_eval(m) == 1
    frozen = bn.running_mean.clone()
    m(x)
    assert torch.equal(bn.running_mean, frozen), "frozen encoder BatchNorm drifted"
    assert m.head.training
    print("OK")
''')


def test_batchnorm_is_left_alone():
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch not installed")
    except ValueError:
        pass
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "No module named 'torch'" in r.stderr:
        pytest.skip("torch not installed")
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
