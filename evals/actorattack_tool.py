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
from evals.bridges.actorattack import INSTRUCTIONS, actorattack_bridge_tools
from evals.scorers.actorattack_stats import (
    actorattack_success_counter,
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
def actorattack_harmbench(max_rows: int | None = None):
    """Create actorattack harmbench evaluation task (tool-support version)."""
    goals = load_harmbench_goals(max_rows)
    dataset = build_harmbench_dataset(
        goals,
        INSTRUCTIONS,
        _CURRENT_DIR / "actorattack_tool_transcripts",
        supplement=tool_supplement_instructions_user("The max number of actor is 3."),
    )

    base_task = make_audit_task(
        dataset=dataset,
        extra_tools=[
            execute_code_in_sandbox(),
            execute_web_search(),
            *actorattack_bridge_tools(),
        ],
        scorer=actorattack_success_counter(),
        target_tools="synthetic",
    )

    return task_with(
        base_task,
        model_roles=default_model_roles(
            auditor_max_connections=30, target_max_connections=10
        ),
    )


if __name__ == "__main__":
    reset_stats()
    try:
        eval(
            tasks=actorattack_harmbench(),
            log_dir="logs/actorattack_tool_harmbench",
            max_connections=30,
        )
    finally:
        cleanup_all_sandboxes()
    print_final_stats()
