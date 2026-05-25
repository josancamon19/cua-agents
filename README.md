# CUA Agents

Reusable computer-use agent adapters for Cifrato/Harbor desktop trials.

This repo contains:

- `anthropic_cua.py`: Anthropic computer-use adapter.
- `openai_cua.py`: OpenAI Responses computer adapter.
- `gemini_cua.py`: Gemini computer-use adapter.
- `harbor.py`: Cifrato Harbor `BaseAgent` wrapper used by `accounting-env`.
- `prompt.txt`: shared Windows desktop operator prompt.

The Harbor wrapper expects to run inside `accounting-env`, where the `harbor`
and `env.harbor` utilities provide trajectories, verifier execution, and cost
accounting.
