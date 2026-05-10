from __future__ import annotations

from typing import Sequence, Tuple

CoordinatePair = Tuple[float, float]

class OutOfBoundsError(ValueError):
    """Raised when the input x value is outside the allowed range [0, 1]."""

def lookup_y_from_x(
    pairs: Sequence[CoordinatePair],
    x_value: float,
) -> float:
    """
    Look up y from x using a sorted table of (x, y) coordinate pairs.

    Behavior:
    - Requires x_value to be between 0 and 1 inclusive.
    - Uses binary search on x values.
    - Returns exact y if x_value is found.
    - Otherwise linearly interpolates between the two surrounding points.

    Args:
        pairs:
            Sorted sequence of (x, y) pairs, sorted ascending by x.
        x_value:
            The x value to look up.

    Returns:
        The corresponding y value, exact or interpolated.

    Raises:
        OutOfBoundsError:
            If x_value is outside [0, 1].
        ValueError:
            If pairs is empty, unsorted, or interpolation cannot be performed.
        TypeError:
            If x_value is not numeric.
    """
    if not isinstance(x_value, (int, float)):
        raise TypeError(f"x_value must be numeric, got {type(x_value).__name__}")

    x_value = float(x_value)

    if not (0.0 <= x_value <= 1.0):
        raise OutOfBoundsError(
            f"Invalid input: x_value {x_value} is out of bounds. Expected 0 <= x <= 1."
        )

    if not pairs:
        raise ValueError("pairs cannot be empty")

    # Optional sanity checks on endpoints against the table itself
    first_x, first_y = pairs[0]
    last_x, last_y = pairs[-1]

    if x_value < first_x or x_value > last_x:
        raise OutOfBoundsError(
            f"x_value {x_value} is outside the table range [{first_x}, {last_x}]."
        )

    # Binary search for exact match or insertion point
    left = 0
    right = len(pairs) - 1

    while left <= right:
        mid = (left + right) // 2
        mid_x, mid_y = pairs[mid]

        if mid_x == x_value:
            return mid_y
        elif mid_x < x_value:
            left = mid + 1
        else:
            right = mid - 1

    # Not found:
    # left is the insertion point
    # right == left - 1
    lower_index = right
    upper_index = left

    if lower_index < 0 or upper_index >= len(pairs):
        raise ValueError(
            f"Could not bracket x_value {x_value} for interpolation."
        )

    x0, y0 = pairs[lower_index]
    x1, y1 = pairs[upper_index]

    if x1 == x0:
        raise ValueError(
            f"Cannot interpolate because bracketing x values are identical: {x0}"
        )

    # Linear interpolation
    y_value = y0 + (x_value - x0) * (y1 - y0) / (x1 - x0)
    return y_value

from engineering.unifac_vle_data_etoh_h2o import UNIFAC_VLE_DATA_ETOH_H2O

def get_y(x_value: float) -> float:
    return lookup_y_from_x(UNIFAC_VLE_DATA_ETOH_H2O, x_value)

if __name__ == "__main__":
    y1 = get_y(.9999)
    y2 = get_y(.99989)
    y3 = get_y(.25)
    print(y1)
    print(y2)
    print(y3)
