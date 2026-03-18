from inspect_ai.model import ChatMessage
from inspect_ai.tool import Tool, ToolDef, ToolSource
from inspect_ai.tool import ToolDef as InspectToolDef
from inspect_ai.util import StoreModel
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from ajar.transcript import Event, TranscriptMetadata
from ajar.types import ToolDefinition


def to_inspect_tooldefs(tools: list[ToolDefinition | InspectToolDef | ToolSource | Tool]) -> list[InspectToolDef]:
    """
    Convert a mixed list of tool definitions to inspect ToolDef format.

    This function handles conversion of custom ToolDefinition objects to the
    standard inspect ToolDef format while leaving other tool types unchanged.

    Args:
        tools: List of tool definitions in various formats

    Returns:
        list[InspectToolDef]: List of tools converted to inspect format
    """
    return [t.to_inspect_tooldef() if isinstance(t, ToolDefinition) else t for t in tools]


class AuditorStore(BaseModel):
    """
    Store for managing the auditor agent's conversation state and tools.

    Attributes:
        messages: List of chat messages in the auditor's conversation
        tools: List of tools available to the auditor agent
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[Tool | ToolDef | ToolSource | None] | ToolSource | None = Field(default_factory=list)

    @field_serializer("tools")
    def serialize_tools(self, value):
        """Ensure tools field is always array on serialize; avoids .length errors in frontend"""
        if value is None:
            return []
        if isinstance(value, ToolSource):
            return []
        if not isinstance(value, list):
            return []
        return value


class TargetStore(BaseModel):
    """
    Store for managing the target model's conversation state and synthetic tools.

    The target is the model being evaluated for alignment vulnerabilities.
    The auditor can create synthetic tools for the target to use during evaluation.

    Attributes:
        messages: List of chat messages in the target's conversation
        tools: List of synthetic tools created by the auditor for the target
    """

    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)


class CombinedStore(BaseModel):
    """
    Store for managing the combined view of the audit conversation.

    This store maintains a unified view of the conversation that includes
    messages from both the auditor and target, useful for transcript generation
    and analysis.

    Attributes:
        messages: List of chat messages from the combined conversation view
    """

    messages: list[ChatMessage] = Field(default_factory=list)


class AuditStore(StoreModel):
    """Central store for managing all state during an alignment audit.

    This store coordinates the conversation state between the auditor and target models,
    tracks events that occur during the audit, and maintains metadata for transcript
    generation and analysis. It serves as the single source of truth for the entire
    audit session state.

    The store is organized into separate sub-stores for different aspects of the audit:
    
    - Auditor store: Manages the auditor model's conversation and tools
    - Target store: Manages the model being evaluated and its synthetic tools
    - Combined store: Maintains a unified view for transcript generation

    Attributes:
        auditor: Store for the auditor's conversation state and available tools
        target: Store for the target model's conversation and synthetic tools created by the auditor
        combined: Store for the unified conversation view used in transcript generation
        special_instructions: The specific instructions given to the auditor for this audit session
        events: List of events that occurred during the audit (messages, tool calls, rollbacks, etc.)
        metadata: Metadata about the audit session (models used, timestamps, description, etc.)
        
    Example:
        ```python
        # Access the audit store within a solver/scorer
        audit_store = store_as(AuditStore)

        # Access the audit store with a specific instance name
        audit_store = store_as(AuditStore, instance="my_audit")
        
        # Access different conversation messages
        auditor_messages = audit_store.auditor.messages
        target_messages = audit_store.target.messages
        combined_messages = audit_store.combined.messages
        
        # Check special instructions
        instructions = audit_store.special_instructions
        ```
    """

    auditor: AuditorStore = Field(default_factory=AuditorStore)
    target: TargetStore = Field(default_factory=TargetStore)
    combined: CombinedStore = Field(default_factory=CombinedStore)
    special_instructions: str | None = None
    target_behavior: str | None = None
    events: list[Event] = Field(default_factory=list)
    metadata: TranscriptMetadata | None = None

    # Counters for tracking usage
    message_count: int = 0  # Current valid send_message count (rollback will decrease this)
    tool_call_count: int = 0  # Total tool calls made (rollback does NOT decrease this)
    total_rollback_count: int = 0  # Total rollback calls made across all indices

    # Limits (None = no limit)
    max_messages: int | None = None
    max_tool_calls: int | None = None
    max_rollback_retries: int | None = None  # Maximum number of retries allowed for each rollback point
    max_total_rollback_retries: int | None = None  # Global maximum number of rollback retries across all indices

    # Rollback retry tracking
    rollback_retry_counts: dict[int, int] = Field(default_factory=dict)  # Track retry count for each rollback index

    # Termination tracking
    termination_reason: str | None = None  # Why the audit ended: "max_turns_reached", "max_tool_calls_reached", "max_messages_reached", "end_conversation"
