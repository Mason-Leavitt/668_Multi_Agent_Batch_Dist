import math

import engineering.formulas as f

def solve_batch_given_W0_x0_xB(W0: float, x0: float, xB: float, n: int = 100) -> dict:
    B = f.get_B_rayleigh(W0, x0, xB, n=n)
    D = f.get_D_mole_balance(W0, B)
    xDavg = f.get_xDavg_mole_balance(W0, B, D, x0, xB)

    return {
        "W0": W0,
        "B": B,
        "D": D,
        "x0": x0,
        "xB": xB,
        "xDavg": xDavg,
    }

def solve_D_given_W0_x0_xDavg(
    W0: float,
    x0: float,
    xDavg_target: float,
    n: int = 100,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> dict:
    """
    Solve a simple binary batch distillation problem when the initial still
    charge, initial composition, and desired average distillate composition are known.

    This answers:

        Given W0, x0, and desired xDavg, how much distillate D can be collected?

    Known:
        W0: Initial moles in the still.
        x0: Initial mole fraction of ethanol in the still.
        xDavg_target: Desired average mole fraction of ethanol in the distillate.

    Solves for:
        xB: Final mole fraction of ethanol in the still.
        B: Final moles remaining in the still.
        D: Total moles collected as distillate.
    """
    f.validate_positive(W0, "W0")
    f.validate_mole_fraction(x0, "x0")
    f.validate_mole_fraction(xDavg_target, "xDavg_target")

    if x0 <= 0.0:
        raise ValueError(
            f"x0 must be greater than 0 for ethanol batch distillation, got x0={x0}."
        )

    if xDavg_target <= x0:
        raise ValueError(
            "For simple ethanol-water batch distillation, the desired average "
            "distillate composition should usually be greater than the initial "
            f"still composition. Got xDavg_target={xDavg_target}, x0={x0}."
        )

    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer.")

    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")

    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")

    def evaluate_trial_xB(xB: float) -> dict:
        B = f.get_B_rayleigh(W0=W0, x0=x0, xB=xB, n=n)
        D = f.get_D_mole_balance(W0=W0, B=B)
        xDavg = f.get_xDavg_mole_balance(W0=W0, B=B, D=D, x0=x0, xB=xB)

        return {
            "xB": xB,
            "B": B,
            "D": D,
            "xDavg": xDavg,
        }

    def residual(xB: float) -> float:
        trial = evaluate_trial_xB(xB)
        return trial["xDavg"] - xDavg_target

    # Avoid xB extremely close to 0 or x0.
    lower = max(1e-6, x0 * 1e-4)
    upper = x0 * (1.0 - 1e-6)

    if upper <= lower:
        raise ValueError(
            f"Could not create a valid xB search interval. "
            f"x0={x0}, lower={lower}, upper={upper}."
        )

    f_lower = residual(lower)
    f_upper = residual(upper)

    if f_lower * f_upper > 0:
        raise ValueError(
            "Could not bracket a solution for xB. "
            "This usually means the requested W0, x0, and xDavg_target are not "
            "physically reachable with the current Rayleigh equation and VLE data. "
            f"Search interval: [{lower}, {upper}]. "
            f"Residual at lower bound: {f_lower}. "
            f"Residual at upper bound: {f_upper}. "
            f"xDavg_target={xDavg_target}."
        )

    for iteration in range(1, max_iter + 1):
        mid = (lower + upper) / 2.0
        f_mid = residual(mid)

        if abs(f_mid) <= tol:
            final = evaluate_trial_xB(mid)

            return {
                "W0": W0,
                "B": final["B"],
                "D": final["D"],
                "x0": x0,
                "xB": final["xB"],
                "xDavg": final["xDavg"],
                "xDavg_target": xDavg_target,
                "xDavg_error": final["xDavg"] - xDavg_target,
                "n": n,
                "iterations": iteration,
                "method": "bisection on xB",
                "assumptions": [
                    "Simple binary batch distillation.",
                    "Ethanol is treated as the more volatile component.",
                    "Vapor-liquid equilibrium y(x) is supplied by get_y(x).",
                    "No vapor holdup, liquid holdup, entrainment, or heat loss corrections are included.",
                    "Rayleigh equation is integrated numerically using composite Simpson's Rule.",
                    "xB was solved numerically so that calculated xDavg matches xDavg_target.",
                ],
                "percent_recovery_ethanol": 100.0 * (final["D"] * final["xDavg"]) / (W0 * x0),
                "fraction_of_charge_distilled": final["D"] / W0,
            }

        if f_lower * f_mid < 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid

    raise RuntimeError(
        f"solve_D_given_W0_x0_xDavg did not converge after {max_iter} iterations. "
        f"Last search interval was [{lower}, {upper}]. "
        f"Last residuals were f_lower={f_lower}, f_upper={f_upper}."
    )

def check_batch_consistency(
    W0: float,
    B: float,
    D: float,
    x0: float,
    xB: float,
    xDavg: float,
    n: int = 100,
    tol: float = 1e-6,
) -> dict:
    f.validate_mole_balance(W0, B, D)

    total_balance_error = W0 - (B + D)
    component_balance_error = W0 * x0 - (B * xB + D * xDavg)

    rayleigh_integral = f.composite_simpson_rule_rayleigh(x0, xB, n=n)
    rayleigh_lhs = math.log(W0 / B)
    rayleigh_error = rayleigh_lhs - rayleigh_integral

    is_total_balance_consistent = abs(total_balance_error) <= tol
    is_component_balance_consistent = abs(component_balance_error) <= tol
    is_rayleigh_consistent = abs(rayleigh_error) <= tol

    is_fully_consistent = (
        is_total_balance_consistent
        and is_component_balance_consistent
        and is_rayleigh_consistent
    )

    return {
        "is_total_balance_consistent": is_total_balance_consistent,
        "is_component_balance_consistent": is_component_balance_consistent,
        "is_rayleigh_consistent": is_rayleigh_consistent,
        "is_fully_consistent": is_fully_consistent,
        "total_balance_error": total_balance_error,
        "component_balance_error": component_balance_error,
        "rayleigh_lhs_ln_W0_over_B": rayleigh_lhs,
        "rayleigh_integral": rayleigh_integral,
        "rayleigh_error": rayleigh_error,
        "tol": tol,
    }


def prototype_design_given_D_xDavg(
    D_target: float,
    xDavg_target: float,
    x0_options: list[float] | None = None,
    xB_options: list[float] | None = None,
    n: int = 100,
) -> dict:
    f.validate_positive(D_target, "D_target")
    f.validate_mole_fraction(xDavg_target, "xDavg_target")

    if x0_options is None:
        x0_options = [0.05, 0.10, 0.15]

    if xB_options is None:
        xB_options = [0.0025, 0.005, 0.01]

    scenarios = []
    notes = []

    for x0 in x0_options:
        f.validate_mole_fraction(x0, "x0_option")
        for xB in xB_options:
            f.validate_mole_fraction(xB, "xB_option")

            if xB >= x0:
                notes.append(
                    f"Skipped invalid illustrative pair x0={x0:.6f}, xB={xB:.6f} because xB must be less than x0."
                )
                continue

            denominator = x0 - xB
            if denominator == 0:
                notes.append(
                    f"Skipped illustrative pair x0={x0:.6f}, xB={xB:.6f} because the design equation denominator is zero."
                )
                continue

            W0 = D_target * (xDavg_target - xB) / denominator
            B = W0 - D_target

            if W0 <= D_target or B <= 0:
                notes.append(
                    f"Skipped illustrative pair x0={x0:.6f}, xB={xB:.6f} because it produced non-physical W0={W0:.6f} or B={B:.6f}."
                )
                continue

            try:
                check = check_batch_consistency(
                    W0=W0,
                    B=B,
                    D=D_target,
                    x0=x0,
                    xB=xB,
                    xDavg=xDavg_target,
                    n=n,
                )
                scenarios.append(
                    {
                        "x0": x0,
                        "xB": xB,
                        "W0": W0,
                        "B": B,
                        "D": D_target,
                        "xDavg_target": xDavg_target,
                        "is_fully_consistent": check["is_fully_consistent"],
                        "is_total_balance_consistent": check["is_total_balance_consistent"],
                        "is_component_balance_consistent": check["is_component_balance_consistent"],
                        "is_rayleigh_consistent": check["is_rayleigh_consistent"],
                        "rayleigh_error": check["rayleigh_error"],
                    }
                )
            except Exception as exc:
                notes.append(
                    f"Skipped illustrative pair x0={x0:.6f}, xB={xB:.6f} because the consistency check failed: {exc}"
                )

    return {
        "D_target": D_target,
        "xDavg_target": xDavg_target,
        "x0_options": x0_options,
        "xB_options": xB_options,
        "scenarios": scenarios,
        "notes": notes,
        "is_illustrative_only": True,
    }


if __name__ == "__main__":
    result = solve_D_given_W0_x0_xDavg(
        W0=1000.0,
        x0=0.05,
        xDavg_target=0.20,
        n=100,
    )

    for key, value in result.items():
        print(f"{key}: {value}")

    check = check_batch_consistency(
        W0=result["W0"],
        B=result["B"],
        D=result["D"],
        x0=result["x0"],
        xB=result["xB"],
        xDavg=result["xDavg"],
        n=result["n"],
    )
    print("-" * 50)
    for key, value in check.items():
        print(f"{key}: {value}")
