"""Command line entry points of the data generator."""

import argparse
import asyncio
from collections.abc import Callable

import pandas as pd

from configs.report_settings import EXPLAIN_TOP_ROWS, REPORT_DECIMALS
from src.calibration.pipeline import run_calibration
from src.dataset import build_dataset, build_placeholder_texts, load_texts
from src.explain.pipeline import run_explanation
from src.features.build import build_feature_table, save_feature_table
from src.features.comparison import pivot_report, run_comparison
from src.features.evaluation import evaluate_text_features
from src.features.inputs import load_feature_inputs, load_truth_for_scores
from src.labels.importance import run_importance
from src.labels.screening import run_screening
from src.margin.pipeline import run_margin
from src.matching.pipeline import run_matches
from src.modeling.cold_report import run_cold_report
from src.modeling.pipeline import run_bootstrap, run_cross_validation
from src.modeling.truth import compute_true_table
from src.modeling.truth_pipeline import run_truth_metrics
from src.settings.loader import Settings, load_settings
from src.storage import (
    dataset_directory,
    features_directory,
    load_world,
    results_directory,
    save_dataset,
    save_world,
    texts_directory,
    world_directory,
)
from src.texts.imperfections import apply_imperfections
from src.texts.jobs import build_text_jobs
from src.texts.models import JOB_NAMES, TextBundle, TextJobs
from src.texts.runner import JobRequest, run_job
from src.world.builder import build_world
from src.world.confounder import apply_confounder

TEXT_SUFFIX = ".jsonl"
TEXT_SCORES_FILE = "text_feature_scores.csv"
TEXT_PREFIXES = ("sim_",)
PLAIN_COMMANDS: list[str] = [
    "features",
    "bootstrap",
    "matches",
    "compare-embeddings",
    "screen-labels",
    "label-importance",
    "true-probabilities",
    "truth-metrics",
    "cold-report",
    "calibrate",
    "explain",
]

Handler = Callable[[argparse.Namespace, Settings], None]


def add_texts_arguments(texts: argparse.ArgumentParser) -> None:
    """Defines the arguments of the command that generates texts."""
    texts.add_argument("--job", required=True, choices=JOB_NAMES)
    texts.add_argument("--model", required=True)
    texts.add_argument("--limit", type=int, default=None)
    texts.add_argument("--output-name", default=None)
    texts.add_argument("--keep-thinking", action="store_true")
    texts.add_argument("--repeat-share", type=float, default=None)


def add_world_arguments(world: argparse.ArgumentParser) -> None:
    """Defines the arguments of the command that builds the world."""
    world.add_argument("--session-count", type=int, default=None)


def add_simulate_arguments(simulate: argparse.ArgumentParser) -> None:
    """Defines the arguments of the command that simulates bookings."""
    simulate.add_argument("--placeholder-texts", action="store_true")


def add_train_arguments(train: argparse.ArgumentParser) -> None:
    """Defines the arguments of the command that trains the models."""
    train.add_argument("--only", nargs="*", default=[])


def add_margin_arguments(margin: argparse.ArgumentParser) -> None:
    """Defines the arguments of the command that compares pricing policies."""
    margin.add_argument("--experiment", default=None)


def build_parser() -> argparse.ArgumentParser:
    """Defines all commands of the project and the scenario switch."""
    parser = argparse.ArgumentParser(description="Synthetic travel dataset generator")
    parser.add_argument("--confounded", action="store_true")
    add_commands(parser)
    return parser


def add_commands(parser: argparse.ArgumentParser) -> None:
    """Adds every command with its own arguments."""
    commands = parser.add_subparsers(dest="command", required=True)
    add_world_arguments(commands.add_parser("world"))
    add_texts_arguments(commands.add_parser("texts"))
    add_simulate_arguments(commands.add_parser("simulate"))
    add_train_arguments(commands.add_parser("train"))
    add_margin_arguments(commands.add_parser("margin"))
    for name in PLAIN_COMMANDS:
        commands.add_parser(name)


def run_world(arguments: argparse.Namespace, settings: Settings) -> None:
    """Builds the world and its text briefs and saves them."""
    if arguments.session_count is not None:
        settings.run.session_count = arguments.session_count
    world = build_world(settings, settings.run.seed)
    jobs = build_text_jobs(settings.run.seed, world, settings)
    save_world(world, jobs, world_directory(settings))
    print(f"sessions={len(world.sessions)} rows={len(world.rows)}")


def run_texts(arguments: argparse.Namespace, settings: Settings) -> None:
    """Generates the texts of one job with a language model server."""
    _, jobs = load_world(world_directory(settings))
    if arguments.repeat_share is not None:
        settings.text.repeat_share = arguments.repeat_share
    name = arguments.output_name or f"{arguments.job}{TEXT_SUFFIX}"
    request = JobRequest(
        job=arguments.job,
        model=arguments.model,
        limit=arguments.limit,
        output_path=texts_directory(settings) / name,
        disable_thinking=not arguments.keep_thinking,
    )
    print(asyncio.run(run_job(request, jobs, settings)))


def select_texts(
    arguments: argparse.Namespace,
    settings: Settings,
    jobs: TextJobs,
) -> TextBundle:
    """Loads or fabricates the texts and applies their imperfections."""
    if arguments.placeholder_texts:
        texts = build_placeholder_texts(jobs)
    else:
        texts = load_texts(texts_directory(settings))
    return apply_imperfections(texts, jobs, settings)


def run_simulate(arguments: argparse.Namespace, settings: Settings) -> None:
    """Simulates the bookings and writes the final dataset."""
    world, jobs = load_world(world_directory(settings))
    if settings.confounder.enabled:
        apply_confounder(world, settings)
    texts = select_texts(arguments, settings, jobs)
    bundle = build_dataset(world, texts, settings, None)
    save_dataset(bundle, dataset_directory(settings))
    print(f"outside_constant={bundle.outside_constant:.3f} rows={len(bundle.public)}")


def report_text_scores(table: pd.DataFrame, settings: Settings) -> None:
    """Scores the text features against the hidden fit and saves the result."""
    names: list[str] = [
        name for name in table.columns if name.startswith(TEXT_PREFIXES)
    ]
    truth = load_truth_for_scores(dataset_directory(settings))
    scores = evaluate_text_features(table, truth, names)
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / TEXT_SCORES_FILE, index=False)
    print(scores.to_string(index=False))


def run_features(arguments: argparse.Namespace, settings: Settings) -> None:
    """Builds the feature table and reports the text feature quality."""
    del arguments
    inputs = load_feature_inputs(dataset_directory(settings), settings.features)
    cache = features_directory(settings)
    table = build_feature_table(inputs, settings.features, cache)
    print(save_feature_table(table, cache))
    report_text_scores(table, settings)


def run_train(arguments: argparse.Namespace, settings: Settings) -> None:
    """Cross-validates the booking models and prints their metrics."""
    results = run_cross_validation(settings, arguments.only)
    print(results.round(REPORT_DECIMALS).T.to_string())


def run_matches_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Computes the match scores and prints their correlations."""
    del arguments
    print(run_matches(settings).round(REPORT_DECIMALS).to_string(index=False))


def run_comparison_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Compares the text models and prints the agreement of all registers."""
    del arguments
    for target, report in pivot_report(run_comparison(settings), "all").items():
        print(target)
        print(report.round(REPORT_DECIMALS).to_string())


def run_screening_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Screens the query labels and prints the per-label and summary tables."""
    del arguments
    table, summary = run_screening(settings)
    print(table.round(REPORT_DECIMALS).to_string(index=False))
    print(summary.round(REPORT_DECIMALS).to_string(index=False))


def run_importance_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Measures and ranks the importance of the tastes from the true formulas."""
    del arguments
    print(run_importance(settings).round(REPORT_DECIMALS).to_string(index=False))


def run_true_probabilities_command(
    arguments: argparse.Namespace,
    settings: Settings,
) -> None:
    """Computes the true booking probability of every package."""
    del arguments
    print(len(compute_true_table(settings)), "rows")


def run_truth_metrics_command(
    arguments: argparse.Namespace,
    settings: Settings,
) -> None:
    """Evaluates the saved scores against the true booking probabilities."""
    del arguments
    intervals, paired = run_truth_metrics(settings)
    print(intervals.round(REPORT_DECIMALS).to_string(index=False))
    print(paired.round(REPORT_DECIMALS).to_string(index=False))


def run_cold_report_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Prints the comparison on bookings of cold and warm hotels."""
    del arguments
    intervals, paired = run_cold_report(settings)
    print(intervals.round(REPORT_DECIMALS).to_string(index=False))
    print(paired.round(REPORT_DECIMALS).to_string(index=False))


def run_calibration_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Studies the calibration and prints the comparison and the group tables."""
    del arguments
    comparison, groups = run_calibration(settings)
    print(comparison.round(REPORT_DECIMALS).T.to_string())
    print(groups.round(REPORT_DECIMALS).to_string(index=False))


def run_explain_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Computes Shapley values and prints the importance of features and groups."""
    del arguments
    importance, groups = run_explanation(settings)
    top = importance.head(EXPLAIN_TOP_ROWS)
    print(top.round(REPORT_DECIMALS).to_string(index=False))
    print(groups.round(REPORT_DECIMALS).to_string(index=False))


def run_margin_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Compares the pricing policies and prints their expected revenue."""
    if arguments.experiment is not None:
        settings.modeling.margin.experiment = arguments.experiment
    table, paired = run_margin(settings)
    print(table.round(REPORT_DECIMALS).to_string(index=False))
    print(paired.round(REPORT_DECIMALS).to_string(index=False))


def run_bootstrap_command(arguments: argparse.Namespace, settings: Settings) -> None:
    """Prints the bootstrap intervals and the paired differences."""
    del arguments
    intervals, paired = run_bootstrap(settings)
    print(intervals.round(REPORT_DECIMALS).to_string(index=False))
    print(paired.round(REPORT_DECIMALS).to_string(index=False))


HANDLERS: dict[str, Handler] = {
    "world": run_world,
    "texts": run_texts,
    "simulate": run_simulate,
    "features": run_features,
    "train": run_train,
    "bootstrap": run_bootstrap_command,
    "matches": run_matches_command,
    "compare-embeddings": run_comparison_command,
    "screen-labels": run_screening_command,
    "label-importance": run_importance_command,
    "true-probabilities": run_true_probabilities_command,
    "truth-metrics": run_truth_metrics_command,
    "cold-report": run_cold_report_command,
    "calibrate": run_calibration_command,
    "explain": run_explain_command,
    "margin": run_margin_command,
}


def enable_confounder(settings: Settings) -> None:
    """Switches to the hidden demand shock scenario and its own data folder."""
    settings.confounder.enabled = True
    settings.run.paths.data_dir = settings.confounder.data_folder


def main() -> None:
    """Runs the command chosen on the command line."""
    arguments = build_parser().parse_args()
    settings = load_settings()
    if arguments.confounded:
        enable_confounder(settings)
    HANDLERS[arguments.command](arguments, settings)


if __name__ == "__main__":
    main()
