"""LLM-agent motor design loop.

An agent (LLM code-writer or parametric heuristic) proposes a constructed
motor, the existing differentiable solver scores it, and the score is fed
back.  Each iteration writes a viewer-compatible checkpoint, so
``motor-web`` shows the agent's designs evolving live.
"""

from organic_motor.agent.loop import AgentLoop, RunResult, run_loop
from organic_motor.agent.sandbox import BuildSpec, execute_agent_code, make_namespace

__all__ = [
    "AgentLoop",
    "RunResult",
    "run_loop",
    "BuildSpec",
    "execute_agent_code",
    "make_namespace",
]
