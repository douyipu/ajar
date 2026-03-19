# ajar

AJAR: Adaptive Jailbreak Architecture for Red-teaming

AJAR is an adaptive jailbreak framework for LLM red-teaming, built on [Petri](https://github.com/safety-research/petri) and extended with the Model Context Protocol (MCP). It supports state rollback, tool simulation, and adaptive planning for multi-turn security evaluation in agentic scenarios.

See **[Architecture & Workflow](docs/architecture.md)** for detailed design and diagrams.

## Setup

```bash
uv sync
```

## Run eval

```bash
uv run evals/crescendo.py
```

Jailbreak algorithms are implemented as [tools](src/ajar/tools) and MCP servers. The tools support parallel evaluation natively; MCP servers do not. We recommend using the [tools](src/ajar/tools) for evaluation.

Supported algorithms: Crescendo, ActorAttack, X-Teaming.

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

This project is developed based on [Petri](https://github.com/safety-research/petri) v2.0.0.

## License

MIT License — see [LICENSE](LICENSE) for details.