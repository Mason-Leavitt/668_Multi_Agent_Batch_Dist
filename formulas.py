import math

from y_lookup_from_x import get_y

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

def get_xDavg_mole_balance(W0: float, B: float, D: float, x0: float, xB: float):
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

if __name__ == "__main__":
    W0 = 100.0
    x0 = 0.40
    xB = 0.10

    B = get_B_rayleigh(W0, x0, xB)
    D = get_D_mole_balance(W0, B)
    xDavg = get_xDavg_mole_balance(W0, B, D, x0, xB)

    print(f"W0: {W0}")
    print(f"B: {B}")
    print(f"D: {D}")
    print(f"x0: {x0}")
    print(f"xB: {xB}")
    print(f"xDavg: {xDavg}")
    print(f"mole balance check: W0 = {B + D}")
    print(f"EtOH balance check: W0*x0 = {B*xB + D*xDavg}")