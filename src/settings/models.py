"""Typed models describing every configuration file of the project."""

from pydantic import BaseModel


class PathSettings(BaseModel):
    data_dir: str
    world_dir: str
    texts_dir: str
    dataset_dir: str
    features_dir: str
    results_dir: str
    models_dir: str


class RunSettings(BaseModel):
    seed: int
    session_count: int
    target_conversion: float
    practical_intent_share: float
    practical_seed_offset: int
    margin_grid: list[float]
    paths: PathSettings
    dataset_columns: list[str]
    log_only_columns: list[str]


class SessionSettings(BaseModel):
    device_names: list[str]
    device_probabilities: list[float]
    device_alpha_shift: dict[str, float]
    hour_floor: float
    evening_center: float
    evening_width: float
    noon_center: float
    noon_weight: float
    noon_width: float
    weekday_weights: list[float]
    month_lengths: list[int]
    departure_component_probabilities: list[float]
    departure_lower_days: list[int]
    departure_upper_days: list[int]
    nights_values: list[int]
    nights_probabilities: list[float]
    adults_values: list[int]
    adults_probabilities: list[float]
    couple_adults: int
    group_minimum_adults: int
    kids_probability_with_partner: float
    kids_probability_single: float
    kid_count_values: list[int]
    kid_count_probabilities: list[float]
    wish_noise_standard_deviation: float
    wish_sparsify_threshold: float
    extra_wish_probability: float
    extra_wish_low: float
    extra_wish_high: float
    kid_family_wish_low: float
    kid_family_wish_high: float
    budget_noise_standard_deviation: float
    country_filter_temperature: float
    country_filter_probability: float
    minimum_stars_filter_probability: float
    maximum_stops_filter_probability: float
    board_filter_probability: float
    minimum_stars_values: list[int]
    minimum_stars_probabilities: list[float]
    maximum_stops_values: list[int]
    maximum_stops_probabilities: list[float]
    board_filter_values: list[str]
    board_filter_probabilities: list[float]


class AspectSettings(BaseModel):
    names: list[str]
    beach: str
    mountains: str
    city: str
    family: str
    luxury: str


class CatalogueSettings(BaseModel):
    aspects: AspectSettings
    star_probabilities: list[float]
    hotels_base: float
    hotels_scale: float
    size_base: float
    size_competition_weight: float
    type_level_exponent: float
    default_type_level: float
    level_noise_standard_deviation: float
    luxury_destination_weight: float
    luxury_star_weight: float
    luxury_noise_standard_deviation: float
    hotel_effect_standard_deviation: float
    rating_base: float
    rating_star_step: float
    rating_effect_weight: float
    rating_noise_standard_deviation: float
    rating_minimum: float
    rating_maximum: float
    rating_decimals: int
    board_keep_probability: float
    price_noise_standard_deviation: float
    name_maximum_tries: int
    name_prefixes: list[str]
    name_nouns: list[str]


class SeasonSettings(BaseModel):
    base_temperature: float
    latitude_curvature: float
    amplitude_per_degree: float
    northern_peak_month: float
    southern_peak_month: float
    beach_threshold: float
    beach_scale: float
    snow_months: list[int]
    shoulder_months: list[int]
    summer_months: list[int]
    city_peak_months: list[int]
    ski_factor_snow: float
    ski_factor_shoulder: float
    ski_factor_summer: float
    ski_factor_other: float
    non_ski_mountain_factor: float
    ski_demand_snow: float
    ski_demand_summer: float
    ski_demand_other: float
    city_demand_peak: float
    city_demand_off: float
    price_base: float
    price_gain: float
    flight_price_exponent: float


class PoolSettings(BaseModel):
    size: int
    tier_quotas: list[int]
    fit_noise_standard_deviation: float
    distance_penalty_per_kilometers: float
    maximum_packages_per_hotel: int
    minimum_eligible_packages: int
    minimum_eligible_hotels: int
    direct_flight_maximum_kilometers: float
    filter_order: list[str]


class FlightSettings(BaseModel):
    speed_kilometers_per_hour: float
    overhead_hours: float
    layover_base_hours: float
    layover_standard_deviation_hours: float
    duration_noise_standard_deviation: float
    duration_decimals: int
    fare_base: float
    fare_per_kilometers: float
    stop_discount: float
    fare_noise_standard_deviation: float
    distance_unit_kilometers: float
    direct_base: float
    direct_slope: float
    direct_minimum: float
    direct_maximum: float
    two_stop_slope: float
    two_stop_maximum: float
    maximum_stops: int
    lead_days: list[int]
    lead_factors: list[float]


class CompetitorSettings(BaseModel):
    markup_high_competition: float
    markup_low_competition: float
    markup_standard_deviation: float
    markup_minimum: float
    price_noise_standard_deviation: float


class LoggingPolicySettings(BaseModel):
    base: float
    per_star: float
    reference_stars: int
    board: float
    last_minute: float
    last_minute_days: int
    premium: float
    premium_price_level: float
    bandwidth: float
    mix_uniform: float
    lower: float
    upper: float


class PricingSettings(BaseModel):
    price_scale: float
    star_factors: list[float]
    board_names: list[str]
    board_factors: list[float]
    board_offer_weights: list[float]
    board_half_board: str
    board_all_inclusive: str
    board_room_only: str
    child_hotel_share: float
    child_flight_share: float
    free_cancellation_base: float
    free_cancellation_far_bonus: float
    free_cancellation_far_days: int
    free_cancellation_star_bonus: float
    free_cancellation_star_limit: int
    total_noise_standard_deviation: float
    money_decimals: int
    flight: FlightSettings
    competitor: CompetitorSettings
    logging_policy: LoggingPolicySettings


class ChoiceSettings(BaseModel):
    fit_amplitude: float
    fit_slope: float
    fit_center: float
    star_coefficient: float
    star_reference: int
    rating_coefficient: float
    rating_reference: float
    flexibility_coefficient: float
    flexibility_far_days: int
    stops_coefficient: float
    kids_stops_factor: float
    short_trip_nights: int
    short_trip_stops_factor: float
    duration_coefficient: float
    duration_free_hours: float
    price_slope_below: float
    price_slope_above: float
    price_sharpness: float
    fit_price_moderation: float
    alpha_z_coefficient: float
    urgency_floor: float
    urgency_gain: float
    urgency_center_days: float
    urgency_scale_days: float
    trust_threshold: float
    trust_coefficient: float
    budget_coefficient: float
    budget_base_per_night: float
    child_person_share: float
    noise_sigma: float
    outside_c0: float
    outside_device_shift: dict[str, float]
    outside_night_limit_hour: int
    outside_night_shift: float
    outside_far_days: int
    outside_far_shift: float
    outside_last_minute_days: int
    outside_last_minute_shift: float
    outside_weekend_start: int
    outside_weekend_shift: float
    outside_noise_standard_deviation: float
    calibration_lower: float
    calibration_upper: float
    calibration_iterations: int
