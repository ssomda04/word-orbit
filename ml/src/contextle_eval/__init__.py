"""Reusable components for Contextle embedding evaluation."""

from contextle_eval.dataset import EvaluationDataset, EvaluationItem, load_dataset
from contextle_eval.metrics import cosine_similarity, evaluate
from contextle_eval.rank_table import (
    RankEntry,
    RankTable,
    build_rank_table,
    build_rank_table_from_file,
    load_vocabulary,
    normalize_vocabulary,
)

__all__ = [
    "EvaluationDataset",
    "EvaluationItem",
    "RankEntry",
    "RankTable",
    "build_rank_table",
    "build_rank_table_from_file",
    "cosine_similarity",
    "evaluate",
    "load_dataset",
    "load_vocabulary",
    "normalize_vocabulary",
]
