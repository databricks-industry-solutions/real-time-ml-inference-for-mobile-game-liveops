import numpy as np

from offer_recommender.data.hashing import md5_uniform_np, md5_uniform_str


def test_md5_uniform_matches_known_values():
    # Locked to the empirically-verified Python/Spark-identical formula.
    assert abs(md5_uniform_str("player_003182:0:17") - 0.305583751295) < 1e-9


def test_md5_uniform_np_vectorized_matches_scalar():
    keys = np.array(["a:1:17", "b:2:17", "c:3:17"])
    got = md5_uniform_np(keys)
    exp = np.array([md5_uniform_str(k) for k in keys])
    np.testing.assert_allclose(got, exp)
    assert ((got >= 0) & (got < 1)).all()
