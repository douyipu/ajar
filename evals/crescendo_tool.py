from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
from inspect_ai import task, task_with
from inspect_ai import eval

from evals.sandbox import (
    execute_code_in_sandbox,
    execute_web_search,
    cleanup_all_sandboxes,
)
from evals.bridges.crescendo import INSTRUCTIONS, crescendo_bridge_tools
from evals.scorers.crescendo_stats import (
    crescendo_success_counter,
    reset_stats,
    print_final_stats,
)
from evals.common import (
    load_harmbench_goals,
    build_harmbench_dataset,
    tool_supplement_instructions_user,
    default_model_roles,
    make_audit_task,
)

dotenv.load_dotenv()

_CURRENT_DIR = Path(__file__).resolve().parent


@task
def crescendo_harmbench(max_rows: int | None = None):
    """Create crescendo harmbench evaluation task (tool-support version)."""
    goals = load_harmbench_goals(max_rows)
    dataset = build_harmbench_dataset(
        goals,
        INSTRUCTIONS,
        _CURRENT_DIR / "crescendo_tool_transcripts",
        supplement=tool_supplement_instructions_user(),
    )

    base_task = make_audit_task(
        dataset=dataset,
        extra_tools=[
            execute_code_in_sandbox(),
            execute_web_search(),
            *crescendo_bridge_tools(),
        ],
        scorer=crescendo_success_counter(),
        target_tools="synthetic",
    )

    return task_with(
        base_task,
        model_roles=default_model_roles(
            auditor_max_connections=50, target_max_connections=10
        ),
    )


if __name__ == "__main__":
    reset_stats()
    try:
        eval(
            tasks=crescendo_harmbench(),
            log_dir="logs/crescendo_tool_harmbench",
            max_connections=50,
        )
    finally:
        cleanup_all_sandboxes()
    print_final_stats()
