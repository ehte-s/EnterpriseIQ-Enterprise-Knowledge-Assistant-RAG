"""CLI script to run retrieval experiments and output reproducible evaluation benchmarks."""

import json
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.logging import get_logger
from app.evaluation.experiments import ExperimentRunner
from app.retrieval.embeddings import EmbeddingFactory

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print(" EnterpriseIQ: Running 100-Question Retrieval Benchmark Experiments")
    print("=" * 70)

    # Use SentenceTransformers embedding provider (or fallback gracefully)
    embedding_provider = EmbeddingFactory.get_provider(provider_type="sentence-transformers")
    runner = ExperimentRunner(embedding_provider=embedding_provider)

    results = runner.run_all_experiments()

    report_md = runner.generate_markdown_comparison(results)
    print("\n" + report_md + "\n")

    # Save results to json
    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        # Save serializable version
        json.dump(results, f, indent=2)

    print(f"Detailed benchmark results saved to: {out_file}")


if __name__ == "__main__":
    main()
