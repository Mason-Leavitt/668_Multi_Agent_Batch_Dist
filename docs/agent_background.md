# Batch Distillation Agent Background

## Purpose

This app helps users explore approximate ethanol-water batch distillation tasks.
It uses:
- an LLM for request classification and conversational explanation
- deterministic engineering functions for calculations

## Supported Workflows

- `feed_to_product_sweep`
- `product_to_feed_sweep`

Other goals may be classified and planned, but not all are executable yet.

## Variable Glossary

- `W0`: total initial feed or still-charge amount in moles of the ethanol-water mixture
- `W0_volume_L`: total initial feed or still-charge volume in liters of the ethanol-water mixture
- `x0`: ethanol mole fraction in the feed
- `x0_abv_percent`: estimated feed ABV percent
- `D`: total distillate or product amount in moles of the ethanol-water mixture
- `D_volume_L`: total distillate or product volume in liters of the ethanol-water mixture
- `xDavg`: average ethanol mole fraction in the collected distillate mixture
- `xDavg_abv_percent`: estimated ABV percent of the collected distillate mixture
- `B`: total remaining bottoms amount in moles of the ethanol-water mixture
- `xB`: ethanol mole fraction in the remaining still bottoms
- `xB_abv_percent`: estimated ABV percent of the remaining still bottoms

## Internal Units vs User-Friendly Units

- Internal engineering calculations use total moles and mole fractions.
- User-facing values may use liters, gallons, and ABV percent.
- User-friendly inputs are normalized to internal units before deterministic calculations run.

## Terminology Rules

- `D` is total product mixture amount in moles, not moles of ethanol.
- `D_volume_L` is total product mixture volume in liters, not liters of ethanol.
- `W0` is total feed mixture amount in moles, not moles of ethanol.
- `W0_volume_L` is total feed mixture volume in liters, not liters of ethanol.
- `x0`, `xB`, and `xDavg` are ethanol mole fractions in mixtures.
- `x0_abv_percent`, `xB_abv_percent`, and `xDavg_abv_percent` are estimated ABV percentages of mixtures.
- If discussing ethanol content, refer to ethanol fraction or ethanol concentration unless an explicit ethanol-only amount is provided.

## Current Limitations

- The LLM should not perform new calculations.
- Deterministic engineering functions are the source of truth for numeric results.
- Result-table follow-up lookup is limited and currently supports only a small set of deterministic queries.
