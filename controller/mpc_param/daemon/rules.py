"""rules.py — Phase 2: rule yaml loader, safe-eval engine, priority + conflict merge.

Loads rules/<target>.yaml. Evaluates per-lap. Returns aggregated key updates
(after step_bound, clamp, and same-key multiplicative merge).

Safe eval: uses eval() with empty __builtins__ and a small whitelist of helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Priority order (low number = high priority).
# ### HJ : 두 가지 hierarchy 동시 지원.
# (a) Legacy 6-level (upenn_mpc / upenn_mpcc): safety → path → stability → speed
#     → smoothness → terminal
# (b) New 4-level (mpcc, control-engineering 표준): safety → stability → speed
#     → smoothness  (path/terminal 흡수)
# yaml 마다 자기 hierarchy 만 사용. 두 set 다 등록해 둬서 어떤 yaml 든 동작.
PRIORITY_LEVELS = {
    # legacy 6-level
    "L0_safety":     0,
    "L1_path":       1,
    "L2_stability":  2,
    "L3_speed":      3,
    "L4_smoothness": 4,
    "L5_terminal":   5,
    # new 4-level (mpcc.yaml)
    "L1_stability":  1,
    "L2_speed":      2,
    "L3_smoothness": 3,
}

SEVERITY_RANK = {"HIGH": 3, "MED": 2, "LOW": 1}

_SAFE_BUILTINS = {
    "any": any, "all": all, "sum": sum, "min": min, "max": max,
    "abs": abs, "len": len, "round": round,
    "True": True, "False": False, "None": None,
    "math": math,
}


@dataclass
class Action:
    key: str
    op: str            # "*" or "+"
    factor: float
    clamp: Tuple[float, float]


@dataclass
class Rule:
    name: str
    priority: str      # e.g. "L1_path"
    severity: str
    when_all: List[str] = field(default_factory=list)
    when_any: List[str] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    reason: str = ""

    @property
    def level(self) -> int:
        return PRIORITY_LEVELS.get(self.priority, 99)


@dataclass
class Policy:
    step_bound_per_key_per_lap: float = 0.20
    suppression: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class FiredRule:
    rule: Rule
    triggered_clauses: List[str]


@dataclass
class KeyUpdate:
    key: str
    new_value: float
    old_value: float
    multiplier_pre_bound: float
    multiplier_applied: float       # after step bound
    contributing_rules: List[str]
    clamped: bool


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
def load_ruleset(path: str) -> Tuple[Policy, List[Rule]]:
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}
    p = doc.get("policy", {})
    policy = Policy(
        step_bound_per_key_per_lap=float(p.get("step_bound_per_key_per_lap", 0.20)),
        suppression={k: list(v) for k, v in (p.get("suppression") or {}).items()},
    )
    rules: List[Rule] = []
    for r in doc.get("rules", []):
        when = r.get("when", {}) or {}
        actions = [
            Action(
                key=str(a["key"]),
                op=str(a.get("op", "*")),
                factor=float(a["factor"]),
                clamp=tuple(a.get("clamp", [-1e18, 1e18])),
            )
            for a in r.get("actions", [])
        ]
        rules.append(Rule(
            name=str(r["name"]),
            priority=str(r.get("priority", "L1_path")),
            severity=str(r.get("severity", "MED")).upper(),
            when_all=list(when.get("all", [])),
            when_any=list(when.get("any", [])),
            actions=actions,
            reason=str(r.get("reason", "")),
        ))
    return policy, rules


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------
def _safe_eval(expr: str, context: Dict[str, Any]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}, **_SAFE_BUILTINS}, context))
    except Exception as e:
        # Caller logs; return False (rule doesn't fire on broken expr).
        raise RuntimeError(f"eval fail in '{expr}': {e}") from e


def _rule_fires(rule: Rule, ctx: Dict[str, Any]) -> Tuple[bool, List[str]]:
    triggered: List[str] = []
    if rule.when_all:
        for expr in rule.when_all:
            if not _safe_eval(expr, ctx):
                return False, []
            triggered.append(expr)
    if rule.when_any:
        any_hit = False
        for expr in rule.when_any:
            if _safe_eval(expr, ctx):
                triggered.append(expr)
                any_hit = True
        if not any_hit:
            return False, []
    if not rule.when_all and not rule.when_any:
        return False, []
    return True, triggered


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def evaluate(metrics: dict,
             policy: Policy,
             rules: List[Rule],
             current_yaml_values: Dict[str, float],
             forbidden_keys: Optional[set] = None) -> Tuple[List[FiredRule], List[KeyUpdate], dict]:
    """Run rules against metrics. Return (fired, updates, debug).

    `current_yaml_values` : {key: current value} for keys that may be touched.
    `forbidden_keys`      : reject any action targeting these keys (defense).
    """
    forbidden_keys = forbidden_keys or set()
    ctx = {
        "lap":     metrics.get("lap", {}),
        "sectors": metrics.get("sectors", []),
        "corners": metrics.get("corners", []),
    }

    fired: List[FiredRule] = []
    eval_errors: List[Tuple[str, str]] = []

    # 1. Find which rules fire.
    for rule in rules:
        try:
            ok, clauses = _rule_fires(rule, ctx)
        except RuntimeError as e:
            eval_errors.append((rule.name, str(e)))
            continue
        if ok:
            fired.append(FiredRule(rule=rule, triggered_clauses=clauses))

    # 2. Suppression: graph-based. If any rule at level X fires AND
    #    suppression[X] contains level Y, all rules at level Y are dropped.
    suppressed: List[FiredRule] = []
    if fired and policy.suppression:
        firing_levels = {f.rule.priority for f in fired}
        suppressed_levels: set = set()
        for active_lvl in firing_levels:
            for victim in policy.suppression.get(active_lvl, []):
                suppressed_levels.add(victim)
        kept, dropped = [], []
        for f in fired:
            (dropped if f.rule.priority in suppressed_levels else kept).append(f)
        suppressed = dropped
        fired = kept

    # 3. Aggregate per key (multiplicative).
    per_key_mult: Dict[str, float] = {}
    per_key_origins: Dict[str, List[str]] = {}
    rejected_actions: List[Tuple[str, str, str]] = []   # (rule, key, why)

    for f in fired:
        for a in f.rule.actions:
            if a.key in forbidden_keys:
                rejected_actions.append((f.rule.name, a.key, "forbidden"))
                continue
            if a.key not in current_yaml_values:
                rejected_actions.append((f.rule.name, a.key, "key_not_in_yaml"))
                continue
            if a.op == "*":
                per_key_mult[a.key] = per_key_mult.get(a.key, 1.0) * a.factor
                per_key_origins.setdefault(a.key, []).append(f.rule.name)
            elif a.op == "+":
                # additive in log domain isn't great; keep as raw delta but uncommon.
                # Convert to multiplicative around current value to share path.
                cur = current_yaml_values[a.key]
                if cur != 0:
                    per_key_mult[a.key] = per_key_mult.get(a.key, 1.0) * (1.0 + a.factor / cur)
                    per_key_origins.setdefault(a.key, []).append(f.rule.name)
                else:
                    rejected_actions.append((f.rule.name, a.key, "additive_on_zero"))

    # 4. Apply step bound, clamp.
    bound = policy.step_bound_per_key_per_lap
    lo_mult, hi_mult = 1.0 - bound, 1.0 + bound
    updates: List[KeyUpdate] = []
    # Build per-key clamp from the *first* action mentioning the key (rules
    # should agree on clamp; we take the most restrictive intersection).
    key_clamps: Dict[str, Tuple[float, float]] = {}
    for f in fired:
        for a in f.rule.actions:
            if a.key in per_key_mult:
                lo_old, hi_old = key_clamps.get(a.key, (-1e18, 1e18))
                key_clamps[a.key] = (max(lo_old, a.clamp[0]), min(hi_old, a.clamp[1]))

    for key, mult in per_key_mult.items():
        applied = max(lo_mult, min(hi_mult, mult))
        cur = float(current_yaml_values[key])
        new_val = cur * applied
        lo, hi = key_clamps.get(key, (-1e18, 1e18))
        clamped_val = max(lo, min(hi, new_val))
        clamped = clamped_val != new_val
        updates.append(KeyUpdate(
            key=key,
            new_value=clamped_val,
            old_value=cur,
            multiplier_pre_bound=mult,
            multiplier_applied=applied,
            contributing_rules=per_key_origins[key],
            clamped=clamped,
        ))

    debug = {
        "fired_count": len(fired),
        "suppressed_count": len(suppressed),
        "suppressed_rules": [f.rule.name for f in suppressed],
        "eval_errors": eval_errors,
        "rejected_actions": rejected_actions,
        "top_level": (min(f.rule.level for f in fired) if fired else None),
    }
    return fired, updates, debug


def updates_to_diff(updates: List[KeyUpdate]) -> Dict[str, dict]:
    """Compact dict for csv trail / git commit message."""
    return {
        u.key: {
            "old": round(u.old_value, 6),
            "new": round(u.new_value, 6),
            "mult": round(u.multiplier_applied, 4),
            "rules": u.contributing_rules,
            "clamped": u.clamped,
        }
        for u in updates
    }
