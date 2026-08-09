from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
from inspect_ai import task, task_with
from inspect_ai import eval

from evals.bridges.xteaming import INSTRUCTIONS, xteaming_bridge_tools
from evals.scorers.xteaming_stats import (
    xteaming_success_counter,
    reset_stats,
    print_final_stats,
)
from evals.common import (
    load_harmbench_goals,
    build_harmbench_dataset,
    base_supplement_instructions,
    XTEAMING_INITIALIZE_PARAMS,
    default_model_roles,
    make_audit_task,
)

dotenv.load_dotenv()

_CURRENT_DIR = Path(__file__).resolve().parent


@task
def xteaming_harmbench(max_rows: int | None = None):
    """Create X-Teaming harmbench evaluation task."""
    goals = load_harmbench_goals(max_rows)
    dataset = build_harmbench_dataset(
        goals,
        INSTRUCTIONS,
        _CURRENT_DIR / "xteaming_transcripts",
        supplement=base_supplement_instructions(
            trailing=XTEAMING_INITIALIZE_PARAMS.strip()
        ),
    )

    base_task = make_audit_task(
        dataset=dataset,
        extra_tools=xteaming_bridge_tools(),
        scorer=xteaming_success_counter(),
        target_tools="none",
    )

    return task_with(
        base_task,
        model_roles=default_model_roles(auditor_max_connections=30),
    )


if __name__ == "__main__":
    reset_stats()
    eval(
        tasks=xteaming_harmbench(),
        log_dir="logs/xteaming_harmbench",
        max_connections=30,
        max_retries=10,
        retry_on_error=3,
        fail_on_error=0.3,
    )
    print_final_stats()
