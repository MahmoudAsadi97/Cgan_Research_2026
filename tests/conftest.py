import numpy as np
import pytest


@pytest.fixture(scope="session")
def archive(tmp_path_factory):
    """An archive in the format of ``data_prep.convert``: six meters per technology group, evening peak,
    midday export for PV."""
    rng = np.random.default_rng(0)
    per_group = 6
    pv, hp, ev = (np.repeat((np.arange(8) >> bit) & 1, per_group) for bit in range(3))
    n, hours = 8 * per_group, np.arange(24)
    evening = np.exp(-0.5 * ((hours - 19) / 2.0) ** 2)
    midday = np.exp(-0.5 * ((hours - 13) / 2.5) ** 2)
    imp = (0.3 + 0.2 * rng.random((n, 1, 1)) + (0.5 + hp[:, None, None]) * evening * rng.gamma(2.0, 0.5, (n, 366, 1))
           + 0.05 * rng.random((n, 366, 24)))
    exp_ = pv[:, None, None] * midday * rng.gamma(2.0, 0.4, (n, 366, 1))
    path = tmp_path_factory.mktemp("data") / "synthetic.npz"
    np.savez_compressed(path, import_kwh=imp.astype(np.float32), export_kwh=exp_.astype(np.float32),
                        count=np.full((n, 366, 24), 4, np.int8), pv=pv, hp=hp, ev=ev, ean=np.arange(1, n + 1),
                        source=np.full(n, "synthetic"))
    return path
