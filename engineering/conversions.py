import engineering.constant_references as refs
import math

def get_grams_of_etoh(moles_of_etoh):
    return moles_of_etoh * refs.ETOH_MOLAR_MASS

def get_grams_of_h2o(moles_of_water):
    return moles_of_water * refs.H2O_MOLAR_MASS

def get_moles_of_etoh(grams_of_etoh):
    return grams_of_etoh / refs.ETOH_MOLAR_MASS

def get_moles_of_h2o(grams_of_h2o):
    return grams_of_h2o / refs.H2O_MOLAR_MASS

def get_mass_percent_etoh_from_mole_fraction(x_etoh: float) -> float:
    """
    Convert ethanol mole fraction to ethanol mass percent.

    Assumes a binary ethanol-water mixture.

    Args:
        x_etoh: Mole fraction ethanol, 0 to 1.

    Returns:
        Ethanol mass percent, 0 to 100.
    """
    if not (0.0 <= x_etoh <= 1.0):
        raise ValueError("x_etoh must be between 0 and 1.")

    # Use a 1 mole total mixture basis.
    moles_etoh = x_etoh
    moles_h2o = 1.0 - x_etoh

    grams_etoh = get_grams_of_etoh(moles_etoh)
    grams_h2o = get_grams_of_h2o(moles_h2o)

    total_grams = grams_etoh + grams_h2o

    if total_grams <= 0:
        raise ValueError("Total mass must be positive.")

    return 100.0 * grams_etoh / total_grams

def get_mixture_density(
    percent_etoh: float,
    density_lookup_list: list[float],
    step: float = 0.1,
) -> float:
    """
    Linearly interpolate density from a percent-indexed lookup array.

    Assumes:
        index 0 -> 0.0%
        index 1 -> 0.1%
        index 2 -> 0.2%
        ...

    Returns:
        Density in g/cm^3.

    Note:
        g/cm^3 is numerically equal to kg/L.
    """
    if not density_lookup_list:
        raise ValueError("density_lookup_list cannot be empty.")

    if not (0.0 <= percent_etoh <= 100.0):
        raise ValueError("percent_etoh must be between 0 and 100.")

    exact_index = percent_etoh / step

    lower_index = math.floor(exact_index)
    upper_index = math.ceil(exact_index)

    if lower_index < 0:
        raise IndexError("lower_index is below lookup range.")

    if lower_index >= len(density_lookup_list):
        return density_lookup_list[-1]

    if upper_index >= len(density_lookup_list):
        return density_lookup_list[-1]

    # Handle exact index.
    if lower_index == upper_index:
        return density_lookup_list[lower_index]

    lower_value = density_lookup_list[lower_index]
    upper_value = density_lookup_list[upper_index]

    fraction = exact_index - lower_index

    return lower_value + fraction * (upper_value - lower_value)

def get_mixture_volume_L_from_moles(
    total_moles: float,
    x_etoh: float,
) -> float:
    """
    Convert a binary ethanol-water mixture from total moles and ethanol mole
    fraction to volume in liters.

    Uses density at 20 C from the mass-percent ethanol lookup.

    Args:
        total_moles:
            Total moles of ethanol + water.
        x_etoh:
            Mole fraction ethanol.

    Returns:
        Mixture volume in liters.
    """
    if total_moles <= 0:
        raise ValueError("total_moles must be positive.")

    if not (0.0 <= x_etoh <= 1.0):
        raise ValueError("x_etoh must be between 0 and 1.")

    moles_etoh = total_moles * x_etoh
    moles_h2o = total_moles * (1.0 - x_etoh)

    grams_etoh = get_grams_of_etoh(moles_etoh)
    grams_h2o = get_grams_of_h2o(moles_h2o)

    total_mass_g = grams_etoh + grams_h2o
    total_mass_kg = total_mass_g / 1000.0

    mass_frac_etoh = get_mass_percent_etoh_from_mole_fraction(x_etoh)

    density_kg_per_L = get_mixture_density(mass_frac_etoh, refs.DENSITY_BY_MASS_PERCENT_20C)

    volume_L = total_mass_kg / density_kg_per_L

    return volume_L

def get_percent_from_density(
    density: float,
    lookup_values: list[float],
    step: float = 0.1,
) -> float:
    """
    Find the percent corresponding to a density using a percent-indexed
    density lookup table.

    Assumes:
        index 0    -> 0.0%
        index 1    -> 0.1%
        ...
        index 1000 -> 100.0%

    Works for increasing or decreasing density tables.
    """
    if not lookup_values:
        raise ValueError("lookup_values cannot be empty.")

    min_density = min(lookup_values)
    max_density = max(lookup_values)

    if not (min_density <= density <= max_density):
        raise ValueError(
            f"density={density} is outside lookup range "
            f"[{min_density}, {max_density}]."
        )

    for i in range(len(lookup_values) - 1):
        density_1 = lookup_values[i]
        density_2 = lookup_values[i + 1]

        density_is_between = (
            density_1 <= density <= density_2
            or density_2 <= density <= density_1
        )

        if density_is_between:
            percent_1 = i * step
            percent_2 = (i + 1) * step

            if density_1 == density_2:
                return percent_1

            fraction = (density - density_1) / (density_2 - density_1)
            return percent_1 + fraction * (percent_2 - percent_1)

    # Fallback: closest point.
    closest_index = min(
        range(len(lookup_values)),
        key=lambda i: abs(lookup_values[i] - density),
    )

    return closest_index * step

def get_moles_and_etoh_frac_from_volume_L_and_abv(
    volume_L: float,
    abv_percent: float,
    density_lookup_by_mass_percent: list[float] = refs.DENSITY_BY_MASS_PERCENT_20C,
    density_lookup_by_vol_percent: list[float] = refs.DENSITY_BY_VOL_PERCENT_20C,
    step: float = 0.1
) -> dict:

    if volume_L <= 0:
        raise ValueError("volume_L must be positive.")

    if not (0.0 <= abv_percent <= 100.0):
        raise ValueError("abv_percent must be between 0 and 100.")

    density = get_mixture_density(
        abv_percent,
        density_lookup_by_vol_percent,
        step=step
    )

    total_mass_kg = volume_L * density
    total_mass_g = total_mass_kg * 1000.0

    mass_percent_etoh = get_percent_from_density(
        density=density,
        lookup_values=density_lookup_by_mass_percent,
        step=step,
    )

    grams_etoh = total_mass_g * (mass_percent_etoh / 100.0)
    grams_h2o = total_mass_g - grams_etoh

    moles_etoh = get_moles_of_etoh(grams_etoh)
    moles_h2o = get_moles_of_h2o(grams_h2o)
    total_moles = moles_etoh + moles_h2o

    x_etoh = moles_etoh / total_moles

    results = {
        "volume_L": volume_L,
        "abv_percent": abv_percent,
        "density_kg_per_L": density,
        "total_mass_g": total_mass_g,
        "mass_percent_etoh": mass_percent_etoh,
        "grams_etoh": grams_etoh,
        "grams_h2o": grams_h2o,
        "moles_etoh": moles_etoh,
        "moles_h2o": moles_h2o,
        "total_moles": total_moles,
        "x_etoh": x_etoh,
    }
    
    return results

def get_abv_from_mol_frac(x_etoh:float) -> float:
    mass_frac = get_mass_percent_etoh_from_mole_fraction(x_etoh)
    density = get_mixture_density(mass_frac, refs.DENSITY_BY_MASS_PERCENT_20C)
    abv = get_percent_from_density(density, refs.DENSITY_BY_VOL_PERCENT_20C)
    return abv


if __name__ == "__main__":
    
    
    info = get_moles_and_etoh_frac_from_volume_L_and_abv(
        volume_L=100.0,
        abv_percent=5.0,
    )

    print(f"Volume: {info['volume_L']:.3f} L")
    print(f"ABV: {info['abv_percent']:.3f}%")
    print(f"Density: {info['density_kg_per_L']:.6f} kg/L")
    print(f"Mass % EtOH: {info['mass_percent_etoh']:.6f}%")
    print(f"Total moles: {info['total_moles']:.3f} mol")
    print(f"Moles EtOH: {info['moles_etoh']:.3f} mol")
    print(f"Moles H2O: {info['moles_h2o']:.3f} mol")
    print(f"xEtOH: {info['x_etoh']:.6f}")

    print(info['x_etoh'])