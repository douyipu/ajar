from pathlib import Path

import dotenv
from inspect_ai import Task, task, task_with
from inspect_ai import eval

from ajar.tools import (
    send_message,
    rollback_conversation,
    end_conversation,
    set_target_system_message,
    create_target_tool,
    send_tool_call_result,
    execute_code_in_sandbox,
    execute_web_search,
    cleanup_all_sandboxes,
)
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
    tool_supplement_instructions_user,
    SUPPLEMENT_INSTRUCTIONS_SYSTEM,
    XTEAMING_INITIALIZE_PARAMS,
    default_model_roles,
    DEFAULT_AUDITOR_KWARGS,
)

dotenv.load_dotenv()

_CURRENT_DIR = Path(__file__).resolve().parent


@task
def xteaming_harmbench(max_rows: int | None = None):
    """
    Create X-Teaming harmbench evaluation task (tool-support version).

    Args:
        max_rows: Limit number of CSV rows to read for testing. If None, read all rows.
    """
    goals = load_harmbench_goals(max_rows)
    dataset = build_harmbench_dataset(
        goals,
        INSTRUCTIONS,
        _CURRENT_DIR / "xteaming_tool_transcripts",
    )

    auditor_tools = [
        set_target_system_message(),
        send_message(),
        rollback_conversation(),
        end_conversation(),
        create_target_tool(),
        send_tool_call_result(),
        execute_code_in_sandbox(),
        execute_web_search(),
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
            supplement_instructions_user=tool_supplement_instructions_user(
                extra_suffix=XTEAMING_INITIALIZE_PARAMS.strip()
            ),
            supplement_instructions_system=SUPPLEMENT_INSTRUCTIONS_SYSTEM,
        ),
        scorer=xteaming_success_counter(),
    )

    return task_with(
        base_task,
        model_roles=default_model_roles(auditor_max_connections=20, target_max_connections=10),
    )


if __name__ == "__main__":
    reset_stats()
    try:
        eval(
            tasks=xteaming_harmbench(),
            log_dir="logs/xteaming_tool",
            max_connections=20,
        )
    finally:
        cleanup_all_sandboxes()
    print_final_stats()
