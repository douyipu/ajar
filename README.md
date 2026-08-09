# ajar

AJAR: Adaptive Jailbreak Architecture for Red-teaming

AJAR is an adaptive jailbreak framework for LLM red-teaming, built on [inspect_petri](https://github.com/meridianlabs-ai/inspect_petri) (Petri 3) and extended with the Model Context Protocol (MCP). It supports state rollback, tool simulation, and adaptive planning for multi-turn security evaluation in agentic scenarios.

See **[Architecture & Workflow](docs/architecture.md)** for detailed design and diagrams.

## Setup

```bash
uv sync --extra dev
```

## Run eval

```bash
uv run evals/crescendo.py
```

Jailbreak algorithms live under [`servers/`](servers) (Crescendo, ActorAttack, X-Teaming) as MCP 2.0 handlers. Evals use an Inspect `store_as` bridge ([`evals/bridges/`](evals/bridges)): the auditor only sees plain tools without a `state` argument; the bridge injects/persists state around each handler call. Orchestration uses stock `inspect_petri`.

Because Petri now exposes [`Controller.resolve_id()`](https://github.com/meridianlabs-ai/inspect_petri/pull/123) ([PR #123](https://github.com/meridianlabs-ai/inspect_petri/pull/123)), bridge `evaluate` takes a short message id from `resume` (e.g. `evaluate(message_id="M8")`) and looks up the exact Target reply in Petri's conversation state instead of asking the auditor to copy the full response into tool arguments. That cuts copy/truncation errors and saves auditor output tokens; `resume` still shows the Target reply in context as usual.

## Paper artifact

The code corresponding to the AJAR paper ([arXiv:2601.10971](https://arxiv.org/abs/2601.10971), last revised 19 Mar 2026, **v2**) is preserved on the branch [`release/ajar-arxiv-2601.10971-v2`](https://github.com/douyipu/ajar/tree/release/ajar-arxiv-2601.10971-v2).

```bash
git checkout release/ajar-arxiv-2601.10971-v2
```

`main` continues active development (including the current MCP 2.0 / Petri 3 refactor) and may diverge from the paper artifact.

## Responsible use

Ajar is designed for **authorized security research and defensive analysis** in the field of large language model security and red-teaming. The software probes for concerning behaviors, which may involve harmful or sensitive content.

**Important considerations:**
- Use only in compliance with all applicable laws and platform terms of service.
- Model providers may block accounts that generate excessive harmful requests—review their policies before use.
- Obtain proper authorization before conducting any red-teaming or security testing.

## Disclaimer

This software is provided "AS IS", without warranty of any kind. Due to the nature of this project (LLM security and red-teaming), it is intended solely for authorized security research and defensive analysis. The authors and copyright holders shall not be liable for any claim, damages, legal consequences, or other liability arising from the use of this software. **Users are strictly responsible for complying with all applicable laws and platform terms of service.**

## Citation

If you use Ajar in your research, please cite:

```bibtex
@misc{dou2026ajaradaptivejailbreakarchitecture,
      title={AJAR: Adaptive Jailbreak Architecture for Red-teaming}, 
      author={Yipu Dou and Wang Yang},
      year={2026},
      eprint={2601.10971},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2601.10971}, 
}
```

## Acknowledgments

This project depends on [inspect_petri](https://github.com/meridianlabs-ai/inspect_petri) (Petri 3, git `main`) and [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) (0.3.252+, with MCP 2.0 support). The message-id evaluate path above relies on Petri [PR #123](https://github.com/meridianlabs-ai/inspect_petri/pull/123) (`Controller.resolve_id`).

## License

MIT License — see [LICENSE](LICENSE) for details.
