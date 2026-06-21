"""AUTO planner package (Flask-backend only).

Self-contained port of the rl-branch multi-aircraft planner + GMM policy, with
no dependency on the training packages (rl_bc / rl_multiple / heuristics_*).
Heavy deps (torch) load only when the planner is started, so importing this
package — or the base simulator / Pyodide build — stays cheap.

This whole package is backend-only: it cannot run in the Pyodide (browser)
deployment and is excluded from the GitHub Pages build manifest.
"""

from __future__ import annotations

_PLANNER = None


def get_planner(airport: str = 'test', runway: str = '27', plan_steps: int = 400):
    """Return the process-wide AutoPlanner singleton (created on first call)."""
    global _PLANNER
    if _PLANNER is None:
        from auto_plan.planner import AutoPlanner
        _PLANNER = AutoPlanner(airport=airport, runway=runway, plan_steps=plan_steps)
    return _PLANNER


def reset_planner_singleton():
    global _PLANNER
    _PLANNER = None
