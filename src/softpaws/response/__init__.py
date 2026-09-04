"""Forward models mapping a neutrino flux to an observed event rate.

Provides two interchangeable paths for comparison:

- the soft-volume model built from :mod:`softpaws.transport`, and
- the published IceCube IRF model (effective area convolved with the energy and
  angular smearing matrices).

:mod:`softpaws.response.effective_area` is the per-site effective-area engine
the paper's comparisons run on.
"""

from .effective_area import (
    ARCA_FIT_BAND,
    ARCA_LOG10_E,
    IC_FIT_BAND,
    IC_LOG10_E,
    N_DEC,
    N_ZENITH,
    arca_effective_area,
    cylinder_projected_area_km2,
    default_cross_section,
    effective_volume_km3,
    first_passage_length_table,
    fit_reach_law,
    ic_effective_area_regenerated,
    ic_effective_area_tau_channel,
    ic_mean_target_volume_cm3,
    ic_required_radius_km,
    ic_target_volume_cm3,
    ic_upgoing_columns,
    required_footprint_radius_km,
    truncated_range_from_table_km,
    truncated_range_km,
)
from .first_principles import (
    DEFAULT_FLAVOURS,
    SPECIES,
    IsoscalarCrossSection,
    arca_effective_area_cm2,
    build_model,
    column_profile,
    detector_curves,
    fit_reach,
    ic_effective_area_cm2,
    residuals,
    rock_range_ratio,
)
from .irfs import EffectiveArea, PointSpreadFunction, SmearingMatrix
from .light_reach import (
    DEFAULT_MIN_MODULES,
    attenuation_length_m,
    effective_body_km,
    hit_count,
    hit_probability,
    hit_radius_m,
    instrumented_chord_km,
    module_charge_pe,
    muon_threshold_gev,
    reach_offset_m,
)
from .soft_volume import SoftVolumeResponse, power_law_flux

__all__ = [
    "ARCA_FIT_BAND",
    "ARCA_LOG10_E",
    "DEFAULT_FLAVOURS",
    "DEFAULT_MIN_MODULES",
    "EffectiveArea",
    "IC_FIT_BAND",
    "IC_LOG10_E",
    "IsoscalarCrossSection",
    "N_DEC",
    "N_ZENITH",
    "PointSpreadFunction",
    "SPECIES",
    "SmearingMatrix",
    "SoftVolumeResponse",
    "arca_effective_area",
    "arca_effective_area_cm2",
    "attenuation_length_m",
    "build_model",
    "column_profile",
    "cylinder_projected_area_km2",
    "default_cross_section",
    "detector_curves",
    "effective_body_km",
    "effective_volume_km3",
    "first_passage_length_table",
    "fit_reach",
    "fit_reach_law",
    "hit_count",
    "hit_probability",
    "hit_radius_m",
    "ic_effective_area_cm2",
    "ic_effective_area_regenerated",
    "ic_effective_area_tau_channel",
    "ic_mean_target_volume_cm3",
    "ic_required_radius_km",
    "ic_target_volume_cm3",
    "ic_upgoing_columns",
    "instrumented_chord_km",
    "module_charge_pe",
    "muon_threshold_gev",
    "power_law_flux",
    "reach_offset_m",
    "required_footprint_radius_km",
    "residuals",
    "rock_range_ratio",
    "truncated_range_from_table_km",
    "truncated_range_km",
]
