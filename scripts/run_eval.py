"""Run the eval set and score it."""
import _bootstrap  # noqa: F401
import asyncio
import json
from dotenv import load_dotenv

from evals.harness.runner import run_eval_set
from evals.harness.scorer import score_run

load_dotenv()


async def main():
    results_path = await run_eval_set("evals/questions/seed.json")
    print("\n" + "=" * 60)
    print("Scoring...")
    scores = score_run(results_path, k=10)

    print(json.dumps(scores["aggregate"], indent=2))
    print("\nPer-question:")
    for q in scores["per_question"]:
        print(f"  {q.get('question_id')} ({q.get('category', '?')}): {q}")


if __name__ == "__main__":
    asyncio.run(main())