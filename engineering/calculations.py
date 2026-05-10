"""Deterministic engineering tools for Rayleigh, balance, sweep, and plotting math.

This module contains numerical/distillation calculations and plotting helpers.
It is a tool layer used by the app and agent-facing orchestration code, not an
agent or language-reasoning module.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from engineering.y_lookup_from_x import get_y
import engineering.conversions as conv

def validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")

def validate_positive(value: float, name: str) -> None:
    validate_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")

def validate_nonnegative(value: float, name: str) -> None:
    validate_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")

def validate_mole_fraction(x: float, name: str) -> None:
    validate_finite(x, name)
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {x}.")

def validate_mole_balance(W0: float, B: float, D: float, rel_tol: float = 1e-6) -> None:
    if not math.isclose(W0, B + D, rel_tol=rel_tol, abs_tol=rel_tol):
        raise ValueError(
            f"Inconsistent mole balance: W0 should equal B + D, "
            f"but got W0={W0}, B={B}, D={D}, B+D={B + D}."
        )

def rayleigh_integrand(x: float) -> float:
    """
    Integrand for the Rayleigh batch distillation equation. Given a liquid mole
    fraction of ethanol, x, gets the equilibrium vapor mole fraction, y(x), and
    returns the value of the integrand:

    f(x) = 1 / (y(x) - x)

    x: liquid mole fraction of ethanol
    y(x): equilibrium vapor mole fraction of ethanol

    Args:
        x: Liquid mole fraction of ethanol in the still.
    Returns:
        The value of the Rayleigh integrand at the given x.
    """
    validate_mole_fraction(x, "x")

    y = get_y(x)
    validate_mole_fraction(y, "y(x)")

    if y <= x:
        raise ValueError(
            f"Invalid Rayleigh integrand at x={x:.6f}: y={y:.6f}, x={x:.6f}. "
            "For this simple Rayleigh calculation using ethanol as the more "
            "volatile component, the equilibrium vapor composition y(x) must "
            "be greater than the liquid composition x."
        )

    return 1.0 / (y - x)

def composite_simpson_rule_rayleigh(x0: float, xB: float, n: int = 100) -> float:
    """Composite Simpson's Rule approximation for the Rayleigh integral.
    Approximates the integral of the Rayleigh integrand from xB to x0 using
    n subintervals. Note that n must be a positive even integer for Simpson's
    rule to be valid. The default value of n is set to 100, which should
    provide a good balance between accuracy and computational efficiency.
    Assumes x0, the initial mole fraction, and xB, the final mole fraction, are
    known.

    Args:
        x0: Initial mole fraction of ethanol in the still.
        xB: Final mole fraction of ethanol in the still.
        n: Number of subintervals to use in the composite Simpson's rule.
    
    Returns:
        Approximation of the integral of the Rayleigh integrand from xB to x0.
    """
    validate_mole_fraction(x0, "x0")
    validate_mole_fraction(xB, "xB")

    if xB >= x0:
        raise ValueError(f"Expected xB < x0, got xB={xB}, x0={x0}.")

    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer.")

    h = (x0 - xB) / n

    total = rayleigh_integrand(xB) + rayleigh_integrand(x0)

    for i in range(1, n):
        x_i = xB + i * h
        weight = 4.0 if i % 2 == 1 else 2.0
        total += weight * rayleigh_integrand(x_i)

    return (h / 3.0) * total

def validate_xDavg_for_azeotrope(xDavg: float) -> bool:
    isValid = False
    if xDavg < 0.89:
        isValid = True
    return isValid

def get_B_rayleigh(W0: float, x0: float, xB: float, n: int = 100) -> float:
    """
    Calculate final moles in the still, B, using the Rayleigh equation.
    Parameter n controls the number of subintervals used in the composite
    Simpson's rule approximation of the Rayleigh integral. Note that n must be
    a positive even integer for Simpson's rule to be valid.
    Assumes W0, the initial moles in the still, is known.
    Assumes x0, the initial mole fraction, and xB, the final mole fraction,
    are known.

    Rayleigh equation:

        ln(W0 / B) = integral from xB to x0 of dx / (y - x)

    Therefore:

        B = W0 / exp(integral)

    Args:
        W0: Initial moles in the still at the start of the process.
        x0: Initial mole fraction of ethanol in the still.
        xB: Final mole fraction of ethanol in the still.
        n: Number of subintervals to use in the composite Simpson's rule.

    Returns:
        Moles in the still at the end of the process, B.
    """
    validate_mole_fraction(x0, "x0")
    validate_mole_fraction(xB, "xB")
    validate_positive(W0, "W0")

    integral = composite_simpson_rule_rayleigh(x0, xB, n=n)
    B = W0 * math.exp(-integral)
    validate_positive(B, "B")

    return B

def get_W0_rayleigh(B: float, x0: float, xB: float, n: int = 100) -> float:
    """
    Calculate initial moles in the still, W0, using the Rayleigh equation.
    Parameter n controls the number of subintervals used in the composite
    Simpson's rule approximation of the Rayleigh integral. Note that n must be
    a positive even integer for Simpson's rule to be valid.
    Assumes B, the final moles in the still, is known.
    Assumes x0, the initial mole fraction, and xB, the final mole fraction,
    are known.

        W0 = B * exp(integral)

    Args:
        B: Moles in the still at the end of the process.
        x0: Initial mole fraction of ethanol in the still.
        xB: Final mole fraction of ethanol in the still.
        n: Number of subintervals to use in the composite Simpson's rule.

    Returns:
        Initial moles in the still at the start of the process, W0.
    """
    validate_mole_fraction(x0, "x0")
    validate_mole_fraction(xB, "xB")
    validate_positive(B, "B")

    integral = composite_simpson_rule_rayleigh(x0, xB, n=n)
    W0 = B * math.exp(integral)
    validate_positive(W0, "W0")

    return W0

def get_W0_mole_balance(B: float, D: float) -> float:
    """Calculate the initial moles, W0, in the still using a mole balance.
    Assumes B, the moles in the still at the end of the process, and D, the
    moles in the distillate at the end of the process, are known.
    
    Args:
        B: Moles in the still at the end of the process.
        D: Moles in the distillate at the end of the process.
    
    Returns:
        Initial moles in the still at the start of the process, W0.
    """
    validate_positive(B, "B")
    validate_positive(D, "D")
    W0 = B + D

    return W0

def get_B_mole_balance(W0: float, D: float) -> float:
    """Calculate the moles in the still at the end of the process using a mole
    balance. Assumes W0, the initial moles in the still, and D, the moles in the
    distillate at the end of the process, are known.
    
    Args:
        W0: Initial total moles of both components in the still.
        D: Moles in the distillate at the end of the process.
    
    Returns:
        Moles in the still at the end of the process, B.
    """
    validate_positive(W0, "W0")
    validate_positive(D, "D")
    B = W0 - D
    validate_positive(B, "B")

    return B

def get_D_mole_balance(W0: float, B: float) -> float:
    """Calculate the moles in the distillate, D, at the end of the process
    using a mole balance. Assumes W0, the initial moles in the still, and B,
    the moles in the still at the end of the process, are known.
    
    Args:
        W0: Initial total moles of both components in the still.
        B: Moles in the still at the end of the process.
    
    Returns:
        Moles in the distillate at the end of the process, D.
    """
    validate_positive(W0, "W0")
    validate_positive(B, "B")
    D = W0 - B
    validate_positive(D, "D")

    return D

def get_x0_mole_balance(W0: float, B: float, D: float, xB: float, xDavg: float) -> float:
    """Calculate the initial mole fraction of EtOH, x0, in the still using a
    mole balance. Assumes W0, the initial moles in the still, B, the moles in
    the still at the end of the process, D, the moles in the distillate at
    the end of the process, xB, the final mole fraction of EtOH in the still,
    and xDavg, the average mole fraction of EtOH in the distillate, are known.
    
    Args:
        W0: Initial total moles of both components in the still.
        B: Moles in the still at the end of the process.
        D: Moles in the distillate at the end of the process.
        xB: Final mole fraction of EtOH in the still.
        xDavg: Average mole fraction of EtOH in the distillate.
    """
    validate_positive(W0, "W0")
    validate_positive(B, "B")
    validate_positive(D, "D")
    validate_mole_fraction(xB, "xB")
    validate_mole_fraction(xDavg, "xDavg")
    validate_mole_balance(W0, B, D)

    mol_etoh_in_B = B * xB
    mol_etoh_in_D = D * xDavg
    x0 = (mol_etoh_in_B + mol_etoh_in_D) / W0
    validate_mole_fraction(x0, "x0")

    return x0

def get_xB_mole_balance(W0: float, B: float, D: float, x0: float, xDavg: float) -> float:
    """Calculate the final mole fraction of EtOH in the still using a mole
    balance. Assumes W0, the initial moles in the still, B, the moles in
    the still at the end of the process, D, the moles in the distillate at
    the end of the process, x0, the initial mole fraction of EtOH in the still,
    and xDavg, the average mole fraction of EtOH in the distillate, are known.
    
    Args:
        W0: Initial total moles of both components in the still.
        B: Moles in the still at the end of the process.
        D: Moles in the distillate at the end of the process.
        x0: Initial mole fraction of EtOH in the still.
        xDavg: Average mole fraction of EtOH in the distillate.

    Returns:
        Final mole fraction of EtOH in the still, xB.
    """
    validate_positive(W0, "W0")
    validate_positive(B, "B")
    validate_positive(D, "D")
    validate_mole_fraction(x0, "x0")
    validate_mole_fraction(xDavg, "xDavg")
    validate_mole_balance(W0, B, D)

    mol_etoh_in_W = W0 * x0
    mol_etoh_in_D = D * xDavg
    xB = (mol_etoh_in_W - mol_etoh_in_D) / B
    validate_mole_fraction(xB, "xB")

    return xB

def get_xDavg_mole_balance_1(W0: float, B: float, D: float, x0: float, xB: float):
    """Calculate the average mole fraction of EtOH in the distillate, xDavg,
    using a mole balance. Assumes W0, the initial moles in the still, B, the
    moles in the still at the end of the process, D, the moles in the
    distillate, x0, the initial mole fraction of EtOH in the still, and xB, the
    final mole fraction of EtOH in the still, are known.

    Args:
        W0: Initial total moles of both components in the still.
        B: Moles in the still at the end of the process.
        D: Moles in the distillate at the end of the process.
        x0: Initial mole fraction of EtOH in the still.
        xB: Final mole fraction of EtOH in the still.

    Returns:
        Average mole fraction of EtOH in the distillate, xDavg.    
    """
    validate_positive(W0, "W0")
    validate_positive(B, "B")
    validate_positive(D, "D")
    validate_mole_fraction(x0, "x0")
    validate_mole_fraction(xB, "xB")
    validate_mole_balance(W0, B, D)

    mol_etoh_in_W = W0 * x0
    mol_etoh_in_B = B * xB
    xDavg = (mol_etoh_in_W - mol_etoh_in_B) / D
    validate_mole_fraction(xDavg, "xDavg")

    return xDavg

def get_xDavg_rayleigh(x0: float, xB: float, n: int = 100):
    """Calculate the average mole fraction of EtOH in the distillate, xDavg,
    using Rayleigh's equations. Assumes x0 and xB are known.

    Args:
        x0: Initial mole fraction of EtOH in the still.
        xB: Final mole fraction of EtOH in the still.

    Returns:
        Average mole fraction of EtOH in the distillate, xDavg.    
    """
    validate_mole_fraction(x0, "x0")
    validate_mole_fraction(xB, "xB")

    integral = composite_simpson_rule_rayleigh(x0, xB, n=n)
    B_over_W0 = math.exp(-integral)
    xDavg = (x0 - xB * B_over_W0) * (1 / (1 - B_over_W0))

    return xDavg

# def find_W0_x0_combinations_from_xDavg_D(
#     D_target: float,
#     xDavg_target: float,
#     x0_min: float = 0.001,
#     x0_max: float = 0.89,
#     num_x0: int = 200,
#     num_xB: int = 200,
#     tolerance: float = 0.01,
#     n: int = 100,
# ):
#     """
#     Find possible W0 and x0 combinations that could produce a target
#     distillate amount D_target and average distillate composition xDavg_target.

#     Uses existing Rayleigh/VLE functions:
#         composite_simpson_rule_rayleigh(x0, xB, n)
#     """

#     results = []

#     x0_values = np.linspace(x0_min, x0_max, num_x0)

#     for x0 in x0_values:

#         # xB must be less than x0
#         xB_values = np.linspace(0.001, x0 - 0.001, num_xB)

#         for xB in xB_values:

#             if xB <= 0 or xB >= x0:
#                 continue

#             try:
#                 integral = composite_simpson_rule_rayleigh(x0, xB, n=n)

#                 B_over_W0 = math.exp(-integral)

#                 # avoid divide-by-zero or tiny D cases
#                 if B_over_W0 >= 1.0:
#                     continue

#                 xDavg = get_xDavg_rayleigh(x0, xB)

#                 if abs(xDavg - xDavg_target) <= tolerance:
#                     W0 = D_target / (1.0 - B_over_W0)
#                     B = W0 * B_over_W0
#                     D = W0 - B

#                     results.append({
#                         "W0": W0,
#                         "x0": x0,
#                         "B": B,
#                         "xB": xB,
#                         "D": D,
#                         "xDavg": xDavg,
#                         "error": xDavg - xDavg_target,
#                         "rayleigh_integral": integral,
#                     })

#             except ValueError:
#                 continue

#     return results

def find_W0_x0_combinations_from_xDavg_D(
    D_target: float,
    xDavg_target: float,
    x0_min: float = 0.001,
    x0_max: float = 0.89,
    num_x0: int = 250,
    num_xB: int = 250,
    tolerance: float = 0.01,
    n: int = 100,
    select_middle_duplicate_x: bool = True,
):
    """
    Find possible W0 and x0 combinations that could produce a target
    distillate amount D_target and average distillate composition xDavg_target.

    Uses existing Rayleigh/VLE functions:
        composite_simpson_rule_rayleigh(x0, xB, n)

    If select_middle_duplicate_x=True, then for each x0 value with multiple
    possible W0 values, keep the row whose W0 is closest to the median W0
    for that x0.
    """

    results = []

    x0_values = np.linspace(x0_min, x0_max, num_x0)

    for x0 in x0_values:

        if x0 <= 0.001:
            continue

        # xB must be less than x0
        xB_values = np.linspace(0.001, x0 - 0.001, num_xB)

        for xB in xB_values:

            if xB <= 0 or xB >= x0:
                continue

            try:
                integral = composite_simpson_rule_rayleigh(x0, xB, n=n)

                B_over_W0 = math.exp(-integral)

                # avoid divide-by-zero or tiny D cases
                if B_over_W0 >= 1.0:
                    continue

                xDavg = get_xDavg_rayleigh(x0, xB)

                if abs(xDavg - xDavg_target) <= tolerance:
                    W0 = D_target / (1.0 - B_over_W0)
                    B = W0 * B_over_W0
                    D = W0 - B

                    results.append({
                        "W0": W0,
                        "x0": x0,
                        "B": B,
                        "xB": xB,
                        "D": D,
                        "xDavg": xDavg,
                        "error": xDavg - xDavg_target,
                        "rayleigh_integral": integral,
                    })

            except ValueError:
                continue

    if select_middle_duplicate_x:
        results = select_middle_y_for_each_x(
            results,
            x_key="x0",
            y_key="W0",
        )

    return results

def select_middle_y_for_each_x(
    results: list[dict],
    x_key: str,
    y_key: str,
) -> list[dict]:
    """
    For each unique x value, keep one row.

    If multiple rows share the same x value, choose the row whose y value
    is closest to the median y value for that x group.

    Example:
        x0 = 0.20 has W0 values [50, 60, 100]
        median W0 = 60
        keep the row where W0 = 60

        x0 = 0.20 has W0 values [50, 100]
        median W0 = 75
        both are equally far, so this keeps the lower-index/first closest row.
    """

    grouped = defaultdict(list)

    for row in results:
        grouped[row[x_key]].append(row)

    selected_results = []

    for x_value, rows in grouped.items():

        if len(rows) == 1:
            selected_results.append(rows[0])
            continue

        y_values = [row[y_key] for row in rows]
        middle_y = float(np.median(y_values))

        selected_row = min(
            rows,
            key=lambda row: abs(row[y_key] - middle_y),
        )

        selected_row = selected_row.copy()
        selected_row["middle_selection_y"] = middle_y
        selected_row["num_candidate_points"] = len(rows)

        selected_results.append(selected_row)

    selected_results.sort(key=lambda row: row[x_key])

    return selected_results

def create_plot_W0_vs_x0_combinations(
    results: list[dict],
    D_target: float,
    xDavg_target: float,
    color_by: str = "Final mole fraction of EtOH in the still, xB",
    show_grid: bool = True,
) -> None:
    """
    Plot required initial charge W0 as a function of initial ethanol mole
    fraction x0 for results from find_W0_x0_combinations_from_xDavg_D().

    Args:
        results:
            List of result dictionaries from find_W0_x0_combinations_from_xDavg_D().
            Each row should contain at least "W0" and "x0".
        color_by:
            Optional result field to color points by. Common choices:
            - "xB"
            - "xDavg"
            - "error"
            - None
        show_grid:
            Whether to show grid lines.
    """
    if not results:
        print("No results to plot.")
        return

    x0_values = [row["x0"] for row in results]
    W0_values = [row["W0"] for row in results]

    plt.figure()

    if color_by is not None and color_by in results[0]:
        color_values = [row[color_by] for row in results]
        scatter = plt.scatter(x0_values, W0_values, c=color_values)
        plt.colorbar(scatter, label=color_by)
    else:
        plt.scatter(x0_values, W0_values)

    plt.xlabel("Initial ethanol mole fraction, x0")
    plt.ylabel("Required initial charge, W0 [mol]")
    plt.title(f"Required feed amount and feed composition for D = {D_target}, xDavg = {xDavg_target}")
    plt.grid(show_grid)

def create_plot_W0_vs_x0_combinations_in_L_ABV(
    results: list[dict],
    D_target: float,
    xDavg_target: float,
    color_by: str | None = "xB",
    show_grid: bool = True,
    show_points: bool = True,
    fit_degree: int = 2,
) -> None:
    """
    Plot required initial feed volume W0 [L] as a function of initial feed
    strength x0 [% ABV] using a best-fit line.

    Converts:
        W0 [mol] + x0 [mole fraction] -> W0 [L]
        x0 [mole fraction] -> x0 [% ABV]
        D_target [mol] + xDavg_target [mole fraction] -> D_target [L]
        xDavg_target [mole fraction] -> xDavg_target [% ABV]

    Args:
        results:
            List of result dictionaries. Each row should contain at least
            "W0" and "x0".
        D_target:
            Target distillate/product amount in moles.
        xDavg_target:
            Target average distillate ethanol mole fraction.
        color_by:
            Optional result field to color the points by. Common choices:
            "xB", "x0", "xDavg", "error", or None.
        show_grid:
            Whether to show grid lines.
        show_points:
            Whether to show the original result points.
        fit_degree:
            Polynomial degree for the fit line.
            Use 1 for linear best fit.
            Use 2 for quadratic fit.
    """
    if not results:
        print("No results to plot.")
        return

    converted_rows = []

    for row in results:
        W0 = row["W0"]
        x0 = row["x0"]

        feed_volume_L = conv.get_mixture_volume_L_from_moles(
            total_moles=W0,
            x_etoh=x0,
        )

        feed_abv = conv.get_abv_from_mol_frac(x0)

        converted_rows.append({
            **row,
            "feed_volume_L": feed_volume_L,
            "feed_abv": feed_abv,
        })

    # Sort by x-axis value so fit-line plotting is clean.
    converted_rows.sort(key=lambda row: row["feed_abv"])

    x0_abv_values = np.array(
        [row["feed_abv"] for row in converted_rows],
        dtype=float,
    )

    W0_volume_L_values = np.array(
        [row["feed_volume_L"] for row in converted_rows],
        dtype=float,
    )

    D_target_volume_L = conv.get_mixture_volume_L_from_moles(
        total_moles=D_target,
        x_etoh=xDavg_target,
    )

    xDavg_target_abv = conv.get_abv_from_mol_frac(xDavg_target)

    plt.figure()

    if show_points:
        if color_by is not None and color_by in converted_rows[0]:
            color_values = []

            for row in converted_rows:
                value = row[color_by]

                if color_by in {"xB", "x0", "xDavg"}:
                    value = conv.get_abv_from_mol_frac(value)

                color_values.append(value)

            scatter = plt.scatter(
                x0_abv_values,
                W0_volume_L_values,
                c=color_values,
            )

            color_label = color_by
            if color_by in {"xB", "x0", "xDavg"}:
                color_label = f"{color_by} [% ABV]"

            plt.colorbar(scatter, label=color_label)

        else:
            plt.scatter(x0_abv_values, W0_volume_L_values)

    # Best-fit line.
    if len(x0_abv_values) >= fit_degree + 1:
        coefficients = np.polyfit(
            x0_abv_values,
            W0_volume_L_values,
            deg=fit_degree,
        )

        fit_function = np.poly1d(coefficients)

        x_fit = np.linspace(
            x0_abv_values.min(),
            x0_abv_values.max(),
            200,
        )

        y_fit = fit_function(x_fit)

        # plt.plot(
        #     x_fit,
        #     y_fit,
        #     label=f"Best-fit line, degree {fit_degree}",
        # )

        plt.legend()
    else:
        print(
            f"Not enough points for degree {fit_degree} fit. "
            f"Need at least {fit_degree + 1} points."
        )

    plt.xlabel("Initial feed strength, x0 [% ABV]")
    plt.ylabel("Required initial feed volume, W0 [L]")
    plt.title(
        f"Required feed volume and feed strength\n"
        f"Target product = {D_target_volume_L:.2f} L at {xDavg_target_abv:.2f}% ABV"
    )
    plt.grid(show_grid)

def create_plot_W0_vs_xB_combinations(
    results: list[dict],
    D_target: float,
    xDavg_target: float,
    color_by: str = "Initial ethanol mole fraction, x0",
    show_grid: bool = True,
) -> None:
    """
    Plot required initial charge W0 as a function of final still ethanol mole
    fraction xB for results from find_W0_x0_combinations_from_xDavg_D().
    """
    if not results:
        print("No results to plot.")
        return

    xB_values = [row["xB"] for row in results]
    W0_values = [row["W0"] for row in results]

    plt.figure()

    if color_by is not None and color_by in results[0]:
        color_values = [row[color_by] for row in results]
        scatter = plt.scatter(xB_values, W0_values, c=color_values)
        plt.colorbar(scatter, label=color_by)
    else:
        plt.scatter(xB_values, W0_values)

    plt.xlabel("Final mole fraction of EtOH in the still, xB")
    plt.ylabel("Required initial charge, W0 [mol]")
    plt.title(f"Required feed amount vs final still composition for D = {D_target}, xDavg = {xDavg_target}")
    plt.grid(show_grid)

def find_D_xDavg_combinations_from_W0_x0(
    W0: float,
    x0: float,
    num_xB: int = 100,
    xB_min: float = 0.001,
    xB_buffer: float = 0.001,
    n: int = 100,
) -> list[dict]:
    """
    Sweep final still ethanol mole fraction xB and calculate possible
    distillate amount D and average distillate composition xDavg for a
    known initial charge W0 and initial ethanol mole fraction x0.

    Args:
        W0: Initial total moles in the still.
        x0: Initial ethanol mole fraction in the still.
        num_xB: Number of xB values to test.
        xB_min: Lowest xB value to test.
        xB_buffer: Keeps the maximum xB slightly below x0.
        n: Number of Simpson-rule intervals for Rayleigh integration.

    Returns:
        A list of dictionaries containing xB, B, D, xDavg, and D/W0.
    """
    validate_positive(W0, "W0")
    validate_mole_fraction(x0, "x0")

    if num_xB < 2:
        raise ValueError("num_xB must be at least 2.")

    xB_max = x0 - xB_buffer

    if xB_max <= xB_min:
        raise ValueError(
            "x0 is too small for the requested xB range. "
            "Need x0 - xB_buffer > xB_min."
        )

    # Start near x0 and move downward toward xB_min.
    xB_values = np.linspace(xB_max, xB_min, num_xB)

    results = []

    for xB in xB_values:
        validate_mole_fraction(xB, "xB")

        if xB <= 0 or xB >= x0:
            continue

        B = get_B_rayleigh(W0, x0, xB, n=n)
        D = get_D_mole_balance(W0, B)
        xDavg = get_xDavg_rayleigh(x0, xB, n=n)

        results.append(
            {
                "xB": xB,
                "B": B,
                "D": D,
                "xDavg": xDavg,
                "D_over_W0": D / W0,
                "D_percent_of_feed": 100.0 * D / W0,
            }
        )

    return results

def create_plot_D_vs_xB(
    results: list[dict],
    W0: float,
    x0: float,
    invert_x_axis: bool = True
) -> None:
    """
    Plot distillate amount D as a function of final still ethanol mole fraction xB.

    Args:
        results:
            List of dictionaries from find_D_xDavg_combinations_from_W0_x0().
            Each row should contain "xB" and "D".
        invert_x_axis:
            If True, plot xB decreasing from left to right.
    """
    xB_values = [row["xB"] for row in results]
    D_values = [row["D"] for row in results]

    plt.figure()
    plt.plot(xB_values, D_values)
    plt.xlabel("Final still ethanol mole fraction, xB")
    plt.ylabel("Distillate collected, D [mol]")
    plt.title(
        f"Distillate collected vs xB for W0 = {W0:.2f} mol, x0 = {x0:.3f}"
    )
    plt.grid(True)

    if invert_x_axis:
        plt.gca().invert_xaxis()

def create_plot_xDavg_vs_xB(
    results: list[dict],
    W0: float,
    x0: float,
    invert_x_axis: bool = True
) -> None:
    """
    Plot average distillate ethanol mole fraction xDavg as a function of
    final still ethanol mole fraction xB.

    Args:
        results:
            List of dictionaries from find_D_xDavg_combinations_from_W0_x0().
            Each row should contain "xB" and "xDavg".
        invert_x_axis:
            If True, plot xB decreasing from left to right.
    """
    xB_values = [row["xB"] for row in results]
    xDavg_values = [row["xDavg"] for row in results]

    plt.figure()
    plt.plot(xB_values, xDavg_values)
    plt.xlabel("Final still ethanol mole fraction, xB")
    plt.ylabel("Average ethanol mole fraction, xDavg, in the distillate, D")
    plt.title(f"Average distillate composition vs xB for W0 = {W0:.2f} mol, x0 = {x0:.3f}")
    plt.grid(True)

    if invert_x_axis:
        plt.gca().invert_xaxis()

def create_plot_D_vs_xDavg(
    results: list[dict],
    W0: float,
    x0: float,
) -> None:
    """
    Plot distillate amount D as a function of average distillate composition xDavg.

    Each point corresponds to one xB value from the Rayleigh batch calculation.
    """
    if not results:
        print("No results to plot.")
        return

    D_values = [row["D"] for row in results]
    xDavg_values = [row["xDavg"] for row in results]
    xB_values = [row["xB"] for row in results]

    plt.figure()
    scatter = plt.scatter(xDavg_values, D_values, c=xB_values)

    plt.xlabel("Average distillate ethanol mole fraction, xDavg")
    plt.ylabel("Distillate collected, D [mol]")
    plt.title(
        f"D vs xDavg for W0 = {W0:.2f} mol, x0 = {x0:.3f}"
    )
    plt.grid(True)

    plt.colorbar(scatter, label="Final still ethanol mole fraction, xB")

def create_plot_D_vs_xDavg_in_L_ABV(
    results: list[dict],
    W0: float,
    x0: float,
) -> None:
    """
    Plot distillate volume D [L] as a function of average distillate strength
    xDavg [% ABV].

    Each point corresponds to one xB value from the Rayleigh batch calculation.

    Conversions:
        D [mol] + xDavg [mole fraction] -> D [L]
        xDavg [mole fraction] -> xDavg [% ABV]
        xB [mole fraction] -> xB [% ABV]
        W0 [mol] + x0 [mole fraction] -> W0 [L]
        x0 [mole fraction] -> x0 [% ABV]
    """
    if not results:
        print("No results to plot.")
        return

    D_volume_L_values = [
            conv.get_mixture_volume_L_from_moles(
                total_moles=row["D"],
                x_etoh=row["xDavg"],
            )
            for row in results
        ]
    
    xDavg_abv_values = [
        conv.get_abv_from_mol_frac(row["xDavg"])
        for row in results
    ]

    xB_abv_values = [
        conv.get_abv_from_mol_frac(row["xB"])
        for row in results
    ]

    W0_volume_L = conv.get_mixture_volume_L_from_moles(
        total_moles=W0,
        x_etoh=x0,
    )

    x0_abv = conv.get_abv_from_mol_frac(x0)

    plt.figure()
    scatter = plt.scatter(xDavg_abv_values, D_volume_L_values, c=xB_abv_values)

    plt.xlabel("Average distillate ethanol content, xDavg [% ABV]")
    plt.ylabel("Distillate collected, D [L]")
    plt.title(
        f"Distillate vs EtOH composition for feed volume = {W0_volume_L:.2f} L at = {x0_abv:.2f}% ABV"
    )
    plt.grid(True)

    plt.colorbar(scatter, label="Final still ethanol content, xB [% ABV]")

if __name__ == "__main__":

    D = 1003.7
    xDavg = 0.192048

    combo_results_W0_x0 = find_W0_x0_combinations_from_xDavg_D(D, xDavg)
    create_plot_W0_vs_x0_combinations(combo_results_W0_x0, D, xDavg)
    create_plot_W0_vs_x0_combinations_in_L_ABV(combo_results_W0_x0, D, xDavg)
    plt.show()


    W0 = 1000
    x0 = 0.05

    # combo_results_D_xDavg = find_D_xDavg_combinations_from_W0_x0(W0, x0)
    # create_plot_xDavg_vs_xB(combo_results_D_xDavg, W0, x0)
    # create_plot_D_vs_xB(combo_results_D_xDavg, W0, x0)
    # create_plot_D_vs_xDavg(combo_results_D_xDavg, W0, x0)
    # create_plot_D_vs_xDavg_in_L_ABV(combo_results_D_xDavg, W0, x0)
    plt.show()
