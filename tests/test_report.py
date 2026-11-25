"""Parts of the result tables that need no trained model."""
import numpy as np
import pytest

from report import household_shares


def test_household_shares_of_two_households():
    days = np.repeat(np.array([[0.0], [2.0], [3.0], [5.0]], np.float32), 24, axis=1)    # means 1 and 4, all 2.5
    largest, between = household_shares(days, np.array([7, 7, 9, 9]))
    assert largest == pytest.approx(0.5)                         # 6.5 of the 13 squared deviations per hour each
    assert between == pytest.approx(9 / 13)                      # 2 * 1.5 ** 2 + 2 * 1.5 ** 2 of the 13
    assert household_shares(days, np.array([1, 2, 3, 4]))[1] == pytest.approx(1.0)
    assert household_shares(days, np.array([1, 1, 1, 1])) == pytest.approx((1.0, 0.0))
