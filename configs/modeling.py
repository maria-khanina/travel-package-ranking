"""Settings of the split, the booking model and its ablation experiments."""

from src.settings.modeling_models import (
    BootstrapSettings,
    CalibrationSettings,
    ExplainSettings,
    MarginSettings,
    CatBoostSettings,
    ExperimentRecord,
    ModelingSettings,
    SplitSettings,
    TruthSettings,
)

AMENITY_COLUMNS: list[str] = [
    "room_view",
    "beach_distance_m",
    "center_distance_m",
    "room_size_sqm",
    "has_kids_club",
    "has_pool",
    "has_spa",
    "free_transfer",
    "free_wifi",
    "has_gym",
    "has_parking",
    "pets_allowed",
    "has_balcony",
]

QWEN = "qwen3_small_instruct"
COSINES: list[str] = [
    f"sim_hotel_{QWEN}",
    f"sim_place_{QWEN}",
    f"sim_room_{QWEN}",
    f"sim_stay_{QWEN}",
]
LABEL_PREFIX = "sim_label_"
LABEL_MATCH: list[str] = ["label_match"]
QWEN_DIRECT = f"emb_{QWEN}_"
BGE_DIRECT = "emb_bge_large_instruct_"
POINTWISE = "Logloss"
YETI = "YetiRank"
SOFTMAX = "QuerySoftMax"

RANKS_TABULAR: list[str] = [
    "price_ratio_vs_competitor",
    "selling_price",
    "price_per_person_night",
    "net_cost",
    "stars",
    "rating",
    "flight_duration_hours",
    "number_of_stops",
    "beach_distance_m",
    "center_distance_m",
    "room_size_sqm",
]
RANKS_ALL: list[str] = RANKS_TABULAR + COSINES + LABEL_MATCH


Inputs = tuple[list[str], list[str], list[str]]


def experiment(
    name: str,
    inputs: Inputs,
    loss: str,
) -> ExperimentRecord:
    """Builds an experiment from text columns, column prefixes and rank columns."""
    return ExperimentRecord(
        name=name,
        text_columns=inputs[0],
        dropped_columns=[],
        include_hotel_id=False,
        extra_prefixes=inputs[1],
        relative_columns=inputs[2],
        loss=loss,
    )


NO_INPUTS: Inputs = ([], [], [])
LABELED: Inputs = (COSINES + LABEL_MATCH, [LABEL_PREFIX], [])
LABELED_RANKED: Inputs = (COSINES + LABEL_MATCH, [LABEL_PREFIX], RANKS_ALL)

EXPERIMENTS: list[ExperimentRecord] = [
    ExperimentRecord(
        name="tabular_without_amenities",
        text_columns=[],
        dropped_columns=AMENITY_COLUMNS,
        include_hotel_id=False,
        extra_prefixes=[],
        relative_columns=[],
        loss=POINTWISE,
    ),
    experiment("tabular", NO_INPUTS, POINTWISE),
    experiment("tabular_ranks", ([], [], RANKS_TABULAR), POINTWISE),
    experiment("cosines", (COSINES, [], []), POINTWISE),
    experiment("cosines_and_labels", LABELED, POINTWISE),
    experiment("direct_qwen", ([], [QWEN_DIRECT], []), POINTWISE),
    experiment("direct_bge", ([], [BGE_DIRECT], []), POINTWISE),
    experiment("direct_qwen_ranks", ([], [QWEN_DIRECT], RANKS_TABULAR), POINTWISE),
    experiment("cosines_labels_ranks", LABELED_RANKED, POINTWISE),
    experiment("cosines_labels_ranks_ranker_yeti", LABELED_RANKED, YETI),
    experiment("cosines_labels_ranks_ranker_softmax", LABELED_RANKED, SOFTMAX),
    experiment("cosines_and_labels_ranker_yeti", LABELED, YETI),
    experiment("cosines_and_labels_ranker_softmax", LABELED, SOFTMAX),
    ExperimentRecord(
        name="cosines_and_labels_hotel_id",
        text_columns=COSINES + LABEL_MATCH,
        dropped_columns=[],
        include_hotel_id=True,
        extra_prefixes=[LABEL_PREFIX],
        relative_columns=[],
        loss=POINTWISE,
    ),
]

MODELING = ModelingSettings(
    split=SplitSettings(
        seed=11,
        test_share=0.2,
        fold_count=5,
        cold_hotel_share=0.15,
        pool_size=30,
    ),
    bootstrap=BootstrapSettings(
        replicates=2000,
        seed=5,
        lower_percentile=2.5,
        upper_percentile=97.5,
        baseline="tabular",
    ),
    truth=TruthSettings(
        draws=100,
        chunk_sessions=10000,
        seed=31,
        output_file="truth_metrics.csv",
    ),
    calibration=CalibrationSettings(
        experiment="cosines_and_labels",
        bins=30,
        session_bins=20,
        confidence_z=1.96,
        price_quantiles=[0.33, 0.67],
        newton_iterations=50,
        newton_tolerance=1e-8,
        probability_floor=1e-9,
        figure_folder="figures",
        model_folder="calibration",
    ),
    explain=ExplainSettings(
        experiment="cosines_and_labels",
        sessions_per_fold=700,
        seed=41,
        top_features=8,
        dependence_points=5000,
        exact_groups={
            "price": [
                "price_ratio_vs_competitor",
                "selling_price",
                "margin_dollars",
                "margin_percentage",
                "net_cost",
                "price_per_person_night",
            ],
            "hotel quality": ["stars", "rating"],
            "amenities": AMENITY_COLUMNS,
            "board and room type": ["board_type", "room_category"],
            "flight": ["number_of_stops", "flight_duration_hours"],
            "destination": ["country", "city"],
            "traveler and trip": [
                "user_device",
                "user_country",
                "search_hour",
                "search_dow",
                "days_to_departure",
                "nights",
                "n_adults",
                "n_children",
                "departure_month",
                "free_cancellation",
            ],
            "query length": ["intent_word_count"],
            "label match": ["label_match"],
        },
        prefix_groups={
            "sim_label_": "label similarities",
            "sim_": "text similarities",
        },
        no_effect_columns=["free_wifi", "has_gym", "has_parking", "pets_allowed"],
        figure_name="shap_dependence.png",
        importance_file="shap_importance.csv",
        group_file="shap_groups.csv",
    ),
    margin=MarginSettings(
        experiment="cosines_and_labels",
        sample_sessions=20000,
        window_sessions=40000,
        seed=53,
        slate_size=3,
        fixed_margin_policies=True,
        draws=100,
        chunk_sessions=2000,
        probability_floor=1e-6,
        bins=10,
        policy_file="margin_policies.csv",
        paired_file="margin_policies_paired.csv",
        response_file="margin_response_check.csv",
        response_price_file="margin_response_by_price.csv",
    ),
    catboost=CatBoostSettings(
        iterations=2000,
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=100,
        thread_count=-1,
        task_type="GPU",
        random_seed=0,
        verbose=0,
    ),
    experiments=EXPERIMENTS,
    top_k_values=[1, 3, 5, 10, 20],
    diagnostic_rows=300000,
    label_column="is_booked",
    hotel_column="hotel_id",
    register_column="hid_intent_register",
)
