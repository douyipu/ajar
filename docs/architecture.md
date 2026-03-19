# AJAR Architecture & Workflow

This document describes the architecture and workflow of **AJAR** (**A**daptive **J**ailbreak **A**rchitecture for **R**ed-teaming), a framework for evaluating multi-turn jailbreak behavior in agentic LLM settings.

Unlike prompt-only red-teaming pipelines, AJAR targets settings where the model can maintain dialogue state, react to intermediate feedback, and invoke external tools. The framework therefore evaluates not only unsafe text generation, but also whether a model can be induced to **plan, adapt, and attempt risky actions** over multiple turns.

## Why AJAR

Traditional red-teaming toolchains work well for single prompts, but they are less suitable for agent-style interactions:

1. **Stateful attacks need stateful orchestration.** Many jailbreaks rely on gradual context shaping rather than one-shot prompting.
2. **Tool-enabled models expand the attack surface.** Risk is no longer limited to harmful text; it may include tool misuse and action execution attempts.
3. **Failed branches affect future turns.** Once the visible context is polluted by weak attempts, later prompts may face stronger refusal behavior.

AJAR addresses these issues by exposing jailbreak methods as callable strategy services and placing an **Auditor Agent** in charge of planning, observation, rollback, and branch selection.

## Key Capabilities

- **Adaptive multi-turn control** through runtime planning rather than fixed scripts.
- **Rollback-aware search** to recover from refusal-heavy or low-value branches.
- **Strategy modularity** via MCP-exposed attack services such as Crescendo, ActorAttack, and X-Teaming.
- **Tool-aware evaluation** that captures both text outputs and action attempts.
- **Safe honeypot execution** that simulates risky operations without causing real-world side effects.

## System Architecture

![AJAR framework overview](images/AJAR_pipeline.svg)

AJAR is organized around three cooperating components.

### 1. Auditor Agent

The **Auditor Agent** is the control center of the framework. It maintains global conversation state, inspects target responses, tracks attack progress, and decides what to do next.

Its responsibilities include:

- maintaining dialogue state and branch history;
- deciding whether to continue, revise, or rollback the current path;
- invoking strategy tools exposed by the MCP Strategy Server;
- monitoring refusal, unsafe leakage, and tool-action attempts.

Rather than acting as a fixed script runner, the Auditor Agent orchestrates a search over possible attack trajectories.

### 2. MCP Strategy Server

The **MCP Strategy Server** exposes jailbreak methods as standardized tools. Instead of embedding each method as a monolithic script, AJAR packages them behind a common interface so the Auditor Agent can call them on demand.

Examples include:

- **Crescendo** for progressive multi-turn escalation,
- **ActorAttack** for role- and actor-based semantic indirection,
- **X-Teaming** for planner-style adaptive attack generation.

This keeps the framework extensible: new algorithms can be added without changing the Auditor Agent's control logic.

### 3. Target Environment

The **Target Environment** is a controlled sandbox containing the target model and its available tools. It serves two purposes:

- providing the interaction surface used during red-teaming;
- acting as a **honeypot** that captures tool calls for later evaluation.

This lets AJAR assess not only text generation, but also whether the target attempts operational actions such as file access, message sending, or command execution.

## Design Principles

### Protocol-driven orchestration

AJAR separates **attack planning** from **attack implementation**. The Auditor Agent does not embed the full logic of each jailbreak algorithm; it calls MCP-exposed tools for planning, message generation, optimization, and evaluation.

### Explicit state management

Conversation state is treated as a first-class object. AJAR tracks committed history, failed branches, rollback points, and progress signals such as dialogue depth and score trend.

### Safe action simulation

Potentially harmful tool calls are intercepted and replaced with controlled responses. This allows AJAR to observe whether a model would continue an unsafe action chain **without executing real-world harm**.

## Workflow Overview

AJAR runs in four stages: initialization, adaptive attack execution, state monitoring, and tool interception.

## 1. Initialization

The initialization phase constructs the attack surface before the main loop begins.

![AJAR initialization sequence diagram](images/initialize_UML.drawio.svg)

Typical setup actions include:

- **Target configuration:** define the goal, attempt limits, and stopping conditions;
- **System prompt setup:** inject the desired role or system context;
- **Synthetic tool creation:** register mock or controlled tools so the target perceives a realistic tool-enabled environment.

Initialization is therefore not just parameter loading; it shapes the target model's starting environment.

## 2. Adaptive Jailbreak Loop

The adaptive jailbreak loop is the core execution cycle of AJAR.

![AJAR jailbreak attack loop sequence diagram](images/jailbreak_UML.drawio.svg)

At a high level, the loop follows this pattern:

1. **Generate or revise an attack plan.**
2. **Produce the next message.**
3. **Send it to the target.**
4. **Evaluate the outcome.**
5. **Adapt through continuation, revision, or rollback.**

Unlike a static red-teaming script, AJAR does **not** assume a fixed linear attack trajectory. It continuously reacts to the target's behavior.

### Rollback as a first-class mechanism

Rollback is one of AJAR's defining features. When a branch triggers a strong refusal or contaminates the visible context, the Auditor Agent can restore an earlier state and continue from a cleaner point.

This matters because many multi-turn failures are caused not just by a weak next prompt, but by earlier attempts pushing the target into a more defensive mode.

## 3. State Monitoring

During execution, the Auditor Agent can inspect the global runtime state through a status query.

![Attack status retrieval flow](images/get_status.drawio.svg)

A typical `get_status` query returns:

- current conversation depth;
- attempted branches and rollback count;
- latest evaluation results;
- recent score trend.

This makes AJAR easier to debug, analyze, and compare across strategies.

## 4. Tool Honeypot and Action Interception

Tool handling is where AJAR extends beyond text-only red-teaming.

![Tool call interception and simulation mechanism](images/get_tools.drawio.svg)

When the target model calls a tool, AJAR routes the request through a policy layer:

- **Benign operations** may be executed normally to preserve realism.
- **Sensitive or unsafe operations** are intercepted and replaced with simulated success or controlled responses.

This achieves two goals at once:

1. preventing real-world side effects during evaluation;
2. revealing whether the model is willing to continue an unsafe action chain once it believes earlier steps succeeded.

## Summary

AJAR reframes jailbreak red-teaming as a **stateful, adaptive, tool-aware search process** rather than a one-shot prompt attack. The Auditor Agent provides global control, the MCP Strategy Server supplies reusable attack capabilities, and the Target Environment offers a safe but realistic execution surface.

Together, these components make AJAR suitable for evaluating LLM safety boundaries in agentic settings where both language and actions matter.
