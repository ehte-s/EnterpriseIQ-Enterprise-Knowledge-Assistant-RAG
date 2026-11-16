"""API route handlers for running evaluations and retrieving benchmark reports."""

import json
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.evaluation.experiments import ExperimentRunner
from app.retrieval.embeddings import EmbeddingFactory

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


def _run_evaluation_task():
    """Background task to run retrieval evaluation experiments and write results."""
    embedding_provider = EmbeddingFactory.get_provider(provider_type="sentence-transformers")
    runner = ExperimentRunner(embedding_provider=embedding_provider)
    results = runner.run_all_experiments()

    out_file = Path("datasets/evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger RAG evaluation benchmark run",
    description="Asynchronously executes the 100-question evaluation benchmark comparing Baseline, Intermediate, and Optimized configurations.",
)
def trigger_evaluation(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_evaluation_task)
    return {
        "status": "accepted",
        "message": "Evaluation suite queued in background. Results will be saved to datasets/evaluation_results.json",
    }


@router.get(
    "/results",
    summary="Retrieve evaluation benchmark results",
    description="Returns the empirical comparison metrics from the latest benchmark run.",
)
def get_evaluation_results():
    results_path = Path("datasets/evaluation_results.json")
    if not results_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation results found. Trigger POST /api/v1/evaluate to generate results."
        )

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading evaluation results: {str(e)}"
        )
