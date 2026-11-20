"""Stage 11 audit machinery on a SYNTHETIC market (re-uses the Stage 8 fixture builder).

* the honest pipeline passes the truncation check;
* two deliberately leaky features (future value, whole-day statistic) must FAIL it;
* the inference contract (only date and date-1, no targets) is enforced.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hft500 import align as A, causality as C, pipeline as P  # noqa: E402
import test_features_cross as TX  # noqa: E402

D2 = TX.D2


class FakeModel:
    """Linear in a few features; enough to carry any leak into the predictions."""
    def predict(self, fr):
        x = (fr["imb_l1"].to_numpy() + 0.1 * fr["ret_2s"].to_numpy() + 0.01 * fr["xret_eth_0.5s"].to_numpy())
        return {h: x * (k + 1) for k, h in enumerate(P.PRED)}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return TX.make_root(tmp_path_factory.mktemp("audit"))


def _fake_models():
    # the synthetic fixture only has BTC/ETH; other instruments get all-NaN features -> predictions 0
    return {i: FakeModel() for i in P.INSTRUMENTS}


def run(feature_fn):
    return lambda r, d: P.wide(P.predict_day(r, d, _fake_models(), feature_fn=feature_fn))


def leak_future(r, inst, d, tick):
    f = P.features(r, inst, d, tick)
    return f.with_columns(pl.col("imb_l1").shift(-1))                       # value from t + 500 ms


def leak_day_stat(r, inst, d, tick):
    f = P.features(r, inst, d, tick)
    return f.with_columns(pl.col("ret_2s") - pl.col("ret_2s").mean())       # whole-day mean


def test_output_contract(root):
    out = P.predict_day(root, D2, _fake_models())
    assert out.height == 691_200 and out.columns == ["ts", "instrument", "pred_500ms", "pred_2s", "pred_8s"]
    assert all(np.isfinite(out[c].to_numpy()).all() for c in P.PRED.values())


def test_honest_pipeline_passes(root):
    d0 = A.day_start(D2)
    res = C.truncation_check(run(P.features), root, D2, n=2, seed=1, Ts=[d0, d0 + 500, d0 + 10_400])
    assert all(r["ok"] for r in res)


@pytest.mark.parametrize("leaky", [leak_future, leak_day_stat])
def test_leaky_canaries_fail(root, leaky):
    d0 = A.day_start(D2)
    res = C.truncation_check(run(leaky), root, D2, n=0, Ts=[d0 + 3_600_000])
    assert not res[0]["ok"]


def test_contract_enforced(root):
    rr = P.RestrictedRoot(root, D2)
    P.predict_day(rr, D2, _fake_models())
    assert {d for _, d, _ in rr.reads} <= {D2, TX.D1}
    with pytest.raises(P.ContractViolation):
        rr.read("targets", D2, "BTCUSDT_PERP")
    with pytest.raises(P.ContractViolation):
        rr.read("book", "2025-09-17", "BTCUSDT_PERP")
