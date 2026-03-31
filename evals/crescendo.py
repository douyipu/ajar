from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
from inspect_ai import Task, task, task_with
from inspect_ai import eval

from ajar.tools import send_message, rollback_conversation, end_conversation, set_target_system_message
from ajar.tools.crescendo import (
    INSTRUCTIONS,
    initialize,
    generate_question,
    evaluate,
    get_status,
)
from ajar.solvers.auditor_agent import auditor_agent
from ajar.solvers.prompts import AUDITOR_SYSTEM_PROMPT_V2
from ajar.scorers.crescendo_stats import crescendo_success_counter, reset_stats, print_final_stats

from evals.common import (
    load_harmbench_goals,
    build_harmbench_dataset,
    base_supplement_instructions,
    default_model_roles,
    DEFAULT_AUDITOR_KWARGS,
)

dotenv.load_dotenv()

_CURRENT_DIR = Path(__file__).resolve().parent


@task
def crescendo_harmbench(max_rows: int | None = None):
    """
    Create crescendo harmbench evaluation task.

    Args:
        max_rows: Limit number of CSV rows to read for testing. If None, read all rows.
    """
    goals = load_harmbench_goals(max_rows)
    dataset = build_harmbench_dataset(
        goals,
        INSTRUCTIONS,
        _CURRENT_DIR / "crescendo_transcripts",
    )

    auditor_tools = [
        set_target_system_message(),
        send_message(),
        rollback_conversation(),
        end_conversation(),
        initialize(),
        generate_question(),
        evaluate(),
        get_status(),
    ]

    base_task = Task(
        dataset=dataset,
        solver=auditor_agent(
            **DEFAULT_AUDITOR_KWARGS,
            auditor_tools=auditor_tools,
            auditor_system_message=AUDITOR_SYSTEM_PROMPT_V2,
            supplement_instructions_user=base_supplement_instructions(),
        ),
        scorer=crescendo_success_counter(),
    )

    return task_with(
        base_task,
        model_roles=default_model_roles(auditor_max_connections=30),
    )


if __name__ == "__main__":
    reset_stats()
    eval(
        tasks=crescendo_harmbench(),
        log_dir="logs/crescendo_harmbench",
        max_connections=30,
    )
    print_final_stats()
