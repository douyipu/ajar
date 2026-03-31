from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
from inspect_ai import Task, task, task_with
from inspect_ai import eval

from ajar.tools import send_message, rollback_conversation, end_conversation, set_target_system_message
from ajar.tools.xteaming import (
    INSTRUCTIONS,
    initialize,
    generate_message,
    evaluate,
    get_status,
    generate_new_strategy,
)
from ajar.solvers.auditor_agent import auditor_agent
from ajar.solvers.prompts import AUDITOR_SYSTEM_PROMPT_V2
from ajar.scorers.xteaming_stats import xteaming_success_counter, reset_stats, print_final_stats

from evals.common import (
    load_harmbench_goals,
    build_harmbench_dataset,
    base_supplement_instructions,
    XTEAMING_INITIALIZE_PARAMS,
    default_model_roles,
    DEFAULT_AUDITOR_KWARGS,
)

dotenv.load_dotenv()

_CURRENT_DIR = Path(__file__).resolve().parent


@task
def xteaming_harmbench(max_rows: int | None = None):
    """
    Create X-Teaming harmbench evaluation task.

    Args:
        max_rows: Limit number of CSV rows to read for testing. If None, read all rows.
    """
    goals = load_harmbench_goals(max_rows)
    dataset = build_harmbench_dataset(
        goals,
        INSTRUCTIONS,
        _CURRENT_DIR / "xteaming_transcripts",
    )

    auditor_tools = [
        set_target_system_message(),
        send_message(),
        rollback_conversation(),
        end_conversation(),
        initialize(),
        generate_message(),
        evaluate(),
        get_status(),
        generate_new_strategy(),
    ]

    base_task = Task(
        dataset=dataset,
        solver=auditor_agent(
            **DEFAULT_AUDITOR_KWARGS,
            auditor_tools=auditor_tools,
            auditor_system_message=AUDITOR_SYSTEM_PROMPT_V2,
            supplement_instructions_user=base_supplement_instructions(
                trailing=XTEAMING_INITIALIZE_PARAMS.strip()
            ),
        ),
        scorer=xteaming_success_counter(),
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
