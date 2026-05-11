#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter-scan-first full analysis for the viability-preserving expansion model.

Core proposition:
    Biological cognition consists in expanding controllable futures while maintaining viability.

This script runs the analysis in two stages.

Stage 1: Parameter exploration
    One-factor and threshold sensitivity analyses are run before the final full analysis.
    The purpose is not to choose the best parameter setting, but to check whether the
    qualitative result is stable across prespecified settings.

Stage 2: Full analysis
    The prespecified default setting is then run at larger N_env.

Main model:
    joint_viability_expansion

Viability-constrained expansion family:
    viability_constrained_endpoint
    joint_viability_expansion

Single-principle / incomplete comparison models:
    random_admissible
    homeostasis_only
    uncertainty_minimizing
    novelty_seeking
    unconstrained_controllability

Primary outcome:
    joint_viable_expansion_score

Default pilot / scan command:
    python3 viability_expansion_paramscan_full.py \
      --outdir ~/Desktop/ve_paramscan_full \
      --scan-envs 50 \
      --full-envs 1000 \
      --workers 4 \
      --scan-permutations 1000 \
      --full-permutations 10000

Fast smoke test:
    python3 viability_expansion_paramscan_full.py \
      --outdir ~/Desktop/ve_paramscan_smoke \
      --scan-envs 2 \
      --full-envs 2 \
      --T 10 \
      --workers 1 \
      --scan-permutations 100 \
      --full-permutations 100 \
      --bootstraps 100 \
      --screen-batch-size 16 \
      --max-candidates 5000
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

NEUTRAL = 0
RESOURCE = 1
HAZARD = 2
OBSTACLE = 3

ACTIONS: Tuple[str, ...] = ("up", "down", "left", "right", "stay")
ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}

MODEL_NAMES: Tuple[str, ...] = (
    "random_admissible",
    "homeostasis_only",
    "uncertainty_minimizing",
    "novelty_seeking",
    "unconstrained_controllability",
    "viability_constrained_endpoint",
    "joint_viability_expansion",
)

MAIN_MODEL = "joint_viability_expansion"
FAMILY_MODELS: Tuple[str, ...] = ("viability_constrained_endpoint", "joint_viability_expansion")
INCOMPLETE_MODELS: Tuple[str, ...] = (
    "random_admissible",
    "homeostasis_only",
    "uncertainty_minimizing",
    "novelty_seeking",
    "unconstrained_controllability",
)


# =============================================================================
# Configuration and structures
# =============================================================================

@dataclass(frozen=True)
class Config:
    N: int = 15
    T: int = 120

    H_pi: int = 4
    H_c: int = 4
    H_ref: int = 8
    max_reference_states: int = 32

    h0: float = 0.60
    hmin: float = 0.25

    resource_min: int = 2
    resource_max: int = 8
    hazard_min: int = 20
    hazard_max: int = 70
    obstacle_min: int = 5
    obstacle_max: int = 45

    move_cost_min: float = 0.020
    move_cost_max: float = 0.060
    resource_gain_min: float = 0.05
    resource_gain_max: float = 0.25
    hazard_damage_min: float = 0.35
    hazard_damage_max: float = 0.90

    alpha: float = 1.0
    eta: float = 0.1

    threshold_rho_omega: float = 0.20
    threshold_rho_delta: float = 0.20
    threshold_kappa: float = 0.05

    n_env: int = 1000
    base_seed: int = 12345

    n_permutations: int = 10000
    n_bootstrap: int = 10000

    max_candidates: int = 300000
    screen_batch_size: int = 128


@dataclass(frozen=True)
class EnvParams:
    n_resources: int
    n_hazards: int
    n_obstacles: int
    move_cost: float
    resource_gain: float
    hazard_damage: float


Position = Tuple[int, int]
Policy = Tuple[str, ...]


@dataclass
class Environment:
    grid: np.ndarray
    start: Position
    params: EnvParams
    source_seed: int


@dataclass(frozen=True)
class SimState:
    pos: Position
    h: float
    visited: FrozenSet[Position]


@dataclass(frozen=True)
class PolicySimulation:
    policy: Policy
    endpoint_pos: Position
    endpoint_h: float
    min_h: float
    visited_after: FrozenSet[Position]
    uncertainty_exposure: float
    movement_cost: float


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: Policy
    sim: PolicySimulation
    admissible: bool
    c_now: float
    c_future: float
    c_unrestricted_future: float
    delta_c: float
    uncertainty_exposure: float
    movement_cost: float


# =============================================================================
# Utility
# =============================================================================

def all_policies(H: int) -> List[Policy]:
    return list(itertools.product(ACTIONS, repeat=H))


def action_index(policy: Policy) -> int:
    action_to_i = {a: i for i, a in enumerate(ACTIONS)}
    idx = 0
    base = len(ACTIONS)
    for action in policy:
        idx = idx * base + action_to_i[action]
    return idx


def in_bounds(pos: Position, N: int) -> bool:
    return 0 <= pos[0] < N and 0 <= pos[1] < N


def neighbors4(pos: Position, N: int) -> List[Position]:
    out: List[Position] = []
    for action in ("up", "down", "left", "right"):
        dr, dc = ACTION_DELTAS[action]
        q = (pos[0] + dr, pos[1] + dc)
        if in_bounds(q, N):
            out.append(q)
    return out


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def stable_label_seed(label: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % 100000


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        prev = min(prev, ranked[i] * n / rank)
        adjusted[order[i]] = min(prev, 1.0)

    return adjusted


# =============================================================================
# Environment generation
# =============================================================================

def draw_env_params(rng: random.Random, cfg: Config) -> EnvParams:
    return EnvParams(
        n_resources=rng.randint(cfg.resource_min, cfg.resource_max),
        n_hazards=rng.randint(cfg.hazard_min, cfg.hazard_max),
        n_obstacles=rng.randint(cfg.obstacle_min, cfg.obstacle_max),
        move_cost=rng.uniform(cfg.move_cost_min, cfg.move_cost_max),
        resource_gain=rng.uniform(cfg.resource_gain_min, cfg.resource_gain_max),
        hazard_damage=rng.uniform(cfg.hazard_damage_min, cfg.hazard_damage_max),
    )


def movement_cost(action: str, env: Environment) -> float:
    return 0.0 if action == "stay" else env.params.move_cost


def passable(pos: Position, env: Environment, cfg: Config) -> bool:
    if not in_bounds(pos, cfg.N):
        return False
    return int(env.grid[pos]) != OBSTACLE


def move_position(pos: Position, action: str, env: Environment, cfg: Config) -> Position:
    dr, dc = ACTION_DELTAS[action]
    q = (pos[0] + dr, pos[1] + dc)
    if not passable(q, env, cfg):
        return pos
    return q


def update_viability(h: float, action: str, next_pos: Position, env: Environment) -> float:
    cell = int(env.grid[next_pos])
    r = env.params.resource_gain if cell == RESOURCE else 0.0
    d = env.params.hazard_damage if cell == HAZARD else 0.0
    return clip01(h - movement_cost(action, env) + r - d)


def viability_preserving_path_to_resource_exists(env: Environment, cfg: Config) -> bool:
    start = env.start
    queue: List[Tuple[Position, float]] = [(start, cfg.h0)]
    seen: Set[Tuple[Position, int]] = {(start, int(round(cfg.h0 * 1000)))}

    while queue:
        pos, h = queue.pop(0)

        if int(env.grid[pos]) == RESOURCE and h >= cfg.hmin:
            return True

        for action in ACTIONS:
            q = move_position(pos, action, env, cfg)
            h_next = update_viability(h, action, q, env)

            if h_next < cfg.hmin:
                continue

            key = (q, int(round(h_next * 1000)))
            if key not in seen:
                seen.add(key)
                queue.append((q, h_next))

    return False


def generate_environment(source_seed: int, cfg: Config) -> Environment:
    rng = random.Random(cfg.base_seed + source_seed)
    N = cfg.N
    start = (N // 2, N // 2)
    all_cells = [(r, c) for r in range(N) for c in range(N)]

    for _attempt in range(50000):
        params = draw_env_params(rng, cfg)
        grid = np.full((N, N), NEUTRAL, dtype=np.int8)

        forbidden = {start}
        start_adjacent = set(neighbors4(start, N))

        hazard_candidates = [p for p in all_cells if p not in forbidden and p not in start_adjacent]
        hazards = set(rng.sample(hazard_candidates, min(params.n_hazards, len(hazard_candidates))))
        for p in hazards:
            grid[p] = HAZARD

        remaining = [p for p in all_cells if p not in forbidden and p not in hazards]
        obstacles = set(rng.sample(remaining, min(params.n_obstacles, len(remaining))))
        for p in obstacles:
            grid[p] = OBSTACLE

        remaining = [p for p in remaining if p not in obstacles]
        resources = set(rng.sample(remaining, min(params.n_resources, len(remaining))))
        for p in resources:
            grid[p] = RESOURCE

        grid[start] = NEUTRAL

        env = Environment(grid=grid, start=start, params=params, source_seed=source_seed)

        if viability_preserving_path_to_resource_exists(env, cfg):
            return env

    raise RuntimeError(f"Failed to generate a valid environment for source_seed={source_seed}")


# =============================================================================
# Controllability and policy evaluation
# =============================================================================

def reachable_endpoint_positions(
    state: SimState,
    env: Environment,
    cfg: Config,
    viability_filter: bool,
    cache: Dict[Tuple, Set[Position]],
) -> Set[Position]:
    key = (state.pos, round(state.h, 6), cfg.H_c, viability_filter)
    if key in cache:
        return cache[key]

    frontier: Set[Tuple[Position, float]] = {(state.pos, round(state.h, 6))}

    for _ in range(cfg.H_c):
        new_frontier: Set[Tuple[Position, float]] = set()

        for pos, h_round in frontier:
            h = float(h_round)
            for action in ACTIONS:
                q = move_position(pos, action, env, cfg)
                h_next = update_viability(h, action, q, env)

                if viability_filter and h_next < cfg.hmin:
                    continue

                new_frontier.add((q, round(h_next, 6)))

        frontier = new_frontier
        if not frontier:
            break

    endpoints = {pos for pos, _h in frontier}
    cache[key] = endpoints
    return endpoints


def c_value(
    state: SimState,
    env: Environment,
    cfg: Config,
    viability_filter: bool,
    cache: Dict[Tuple, Set[Position]],
) -> float:
    endpoints = reachable_endpoint_positions(state, env, cfg, viability_filter, cache)
    return math.log(1.0 + len(endpoints))


def simulate_policy(state: SimState, env: Environment, cfg: Config, policy: Policy) -> PolicySimulation:
    pos = state.pos
    h = state.h
    min_h = h
    visited = set(state.visited)
    uncertainty_exposure = 0.0
    cost = 0.0

    for action in policy:
        q = move_position(pos, action, env, cfg)
        if q not in visited:
            uncertainty_exposure += 1.0

        h = update_viability(h, action, q, env)
        cost += movement_cost(action, env)

        pos = q
        visited.add(pos)
        min_h = min(min_h, h)

    return PolicySimulation(
        policy=policy,
        endpoint_pos=pos,
        endpoint_h=h,
        min_h=min_h,
        visited_after=frozenset(visited),
        uncertainty_exposure=uncertainty_exposure,
        movement_cost=cost,
    )


def evaluate_policy(
    state: SimState,
    env: Environment,
    cfg: Config,
    policy: Policy,
    cache: Dict[Tuple, Set[Position]],
    c_now: float,
) -> PolicyEvaluation:
    sim = simulate_policy(state, env, cfg, policy)
    endpoint_state = SimState(sim.endpoint_pos, sim.endpoint_h, sim.visited_after)

    admissible = sim.min_h >= cfg.hmin
    c_future = c_value(endpoint_state, env, cfg, True, cache)
    c_unrestricted_future = c_value(endpoint_state, env, cfg, False, cache)
    delta_c = c_future - c_now

    return PolicyEvaluation(
        policy=policy,
        sim=sim,
        admissible=admissible,
        c_now=c_now,
        c_future=c_future,
        c_unrestricted_future=c_unrestricted_future,
        delta_c=delta_c,
        uncertainty_exposure=sim.uncertainty_exposure,
        movement_cost=sim.movement_cost,
    )


def evaluate_policy_space(
    state: SimState,
    env: Environment,
    cfg: Config,
    policies: Sequence[Policy],
    cache: Dict[Tuple, Set[Position]],
) -> List[PolicyEvaluation]:
    c_now = c_value(state, env, cfg, True, cache)
    return [evaluate_policy(state, env, cfg, pi, cache, c_now) for pi in policies]


# =============================================================================
# Critical-regime screening
# =============================================================================

def collect_reference_states(env: Environment, cfg: Config) -> List[SimState]:
    start_state = SimState(env.start, cfg.h0, frozenset({env.start}))
    queue: List[Tuple[SimState, int]] = [(start_state, 0)]
    seen: Set[Tuple[Position, int]] = {(start_state.pos, int(round(start_state.h * 1000)))}

    out: List[SimState] = [start_state]

    while queue and len(out) < cfg.max_reference_states:
        state, depth = queue.pop(0)
        if depth >= cfg.H_ref:
            continue

        for action in ACTIONS:
            q = move_position(state.pos, action, env, cfg)
            h_next = update_viability(state.h, action, q, env)

            if h_next < cfg.hmin:
                continue

            visited = set(state.visited)
            visited.add(q)
            next_state = SimState(q, h_next, frozenset(visited))

            key = (q, int(round(h_next * 1000)))
            if key not in seen:
                seen.add(key)
                out.append(next_state)
                queue.append((next_state, depth + 1))

                if len(out) >= cfg.max_reference_states:
                    break

    return out[:cfg.max_reference_states]


def diagnostics_for_state(state: SimState, env: Environment, cfg: Config, policies: Sequence[Policy]) -> Dict:
    cache: Dict[Tuple, Set[Position]] = {}
    evaluations = evaluate_policy_space(state, env, cfg, policies, cache)

    n_all = len(evaluations)
    admissible = [ev for ev in evaluations if ev.admissible]
    inadmissible = [ev for ev in evaluations if not ev.admissible]
    n_adm = len(admissible)

    rho_omega = 1.0 - (n_adm / n_all)

    if n_adm == 0:
        rho_delta = 0.0
        max_viable_c = 0.0
    else:
        rho_delta = sum(1 for ev in admissible if ev.delta_c > 0.0) / n_adm
        max_viable_c = max(ev.c_future for ev in admissible)

    if inadmissible:
        max_inadmissible_unrestricted_c = max(ev.c_unrestricted_future for ev in inadmissible)
    else:
        max_inadmissible_unrestricted_c = max_viable_c

    cmax = math.log(1.0 + cfg.N * cfg.N)
    kappa = max(0.0, max_inadmissible_unrestricted_c - max_viable_c) / cmax

    return {
        "rho_omega": float(rho_omega),
        "rho_delta": float(rho_delta),
        "kappa": float(kappa),
        "n_policies": n_all,
        "n_admissible": n_adm,
        "n_inadmissible": n_all - n_adm,
    }


def environment_diagnostics(env: Environment, cfg: Config) -> Dict:
    policies = all_policies(cfg.H_pi)
    reference_states = collect_reference_states(env, cfg)

    state_rows = [diagnostics_for_state(q, env, cfg, policies) for q in reference_states]

    rho_omega = float(np.mean([r["rho_omega"] for r in state_rows]))
    rho_delta = float(np.mean([r["rho_delta"] for r in state_rows]))
    kappa = float(np.mean([r["kappa"] for r in state_rows]))

    critical = (
        rho_omega >= cfg.threshold_rho_omega
        and rho_delta >= cfg.threshold_rho_delta
        and kappa >= cfg.threshold_kappa
    )

    p = env.params

    return {
        "source_seed": env.source_seed,
        "critical_regime": bool(critical),
        "rho_omega": rho_omega,
        "rho_delta": rho_delta,
        "kappa": kappa,
        "n_reference_states": len(reference_states),
        "mean_n_admissible": float(np.mean([r["n_admissible"] for r in state_rows])),
        "mean_n_inadmissible": float(np.mean([r["n_inadmissible"] for r in state_rows])),
        "n_resources": p.n_resources,
        "n_hazards": p.n_hazards,
        "n_obstacles": p.n_obstacles,
        "move_cost": p.move_cost,
        "resource_gain": p.resource_gain,
        "hazard_damage": p.hazard_damage,
    }


def screen_candidate(args: Tuple[int, Config]) -> Tuple[Environment, Dict]:
    source_seed, cfg = args
    env = generate_environment(source_seed, cfg)
    diag = environment_diagnostics(env, cfg)
    return env, diag


def collect_environments(cfg: Config, workers: int, critical_only: bool) -> Tuple[List[Environment], pd.DataFrame, pd.DataFrame]:
    accepted_envs: List[Environment] = []
    accepted_rows: List[Dict] = []
    screening_rows: List[Dict] = []

    next_seed = 0

    print(
        f"[screen] target={cfg.n_env}, critical_only={critical_only}, "
        f"thresholds: rho_omega>={cfg.threshold_rho_omega}, "
        f"rho_delta>={cfg.threshold_rho_delta}, kappa>={cfg.threshold_kappa}"
    )

    while len(accepted_envs) < cfg.n_env and next_seed < cfg.max_candidates:
        batch = list(range(next_seed, min(next_seed + cfg.screen_batch_size, cfg.max_candidates)))
        next_seed += len(batch)
        tasks = [(seed, cfg) for seed in batch]

        if workers <= 1:
            outputs = [screen_candidate(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(screen_candidate, task) for task in tasks]
                outputs = [f.result() for f in as_completed(futures)]

        for env, diag in outputs:
            screening_rows.append(dict(diag))
            accept = bool(diag["critical_regime"]) if critical_only else True

            if accept and len(accepted_envs) < cfg.n_env:
                env_id = len(accepted_envs)
                row = dict(diag)
                row["env_id"] = env_id
                accepted_envs.append(env)
                accepted_rows.append(row)

        critical_rate = float(np.mean([r["critical_regime"] for r in screening_rows])) if screening_rows else 0.0
        print(
            f"[screen] checked={len(screening_rows)}, accepted={len(accepted_envs)}, "
            f"critical_rate={critical_rate:.4f}"
        )

    if len(accepted_envs) < cfg.n_env:
        raise RuntimeError(
            f"Accepted only {len(accepted_envs)} environments after screening "
            f"{len(screening_rows)} candidates. Relax thresholds or increase --max-candidates."
        )

    accepted = pd.DataFrame(accepted_rows).sort_values("env_id").reset_index(drop=True)
    screening = pd.DataFrame(screening_rows).reset_index(drop=True)

    return accepted_envs, accepted, screening


# =============================================================================
# Model selection
# =============================================================================

def admissible_or_fallback(evaluations: Sequence[PolicyEvaluation]) -> List[PolicyEvaluation]:
    adm = [ev for ev in evaluations if ev.admissible]
    if adm:
        return adm

    max_min_h = max(ev.sim.min_h for ev in evaluations)
    return [ev for ev in evaluations if ev.sim.min_h == max_min_h]


def select_policy(model: str, evaluations: Sequence[PolicyEvaluation], cfg: Config, rng: random.Random) -> PolicyEvaluation:
    if model == "unconstrained_controllability":
        candidates = list(evaluations)
    else:
        candidates = admissible_or_fallback(evaluations)

    if model == "random_admissible":
        return rng.choice(candidates)

    if model == "homeostasis_only":
        return max(candidates, key=lambda ev: (ev.sim.endpoint_h, -ev.movement_cost, -action_index(ev.policy)))

    if model == "uncertainty_minimizing":
        return max(candidates, key=lambda ev: (-ev.uncertainty_exposure, ev.sim.endpoint_h, -ev.movement_cost, -action_index(ev.policy)))

    if model == "novelty_seeking":
        return max(candidates, key=lambda ev: (ev.uncertainty_exposure, ev.sim.endpoint_h, -ev.movement_cost, -action_index(ev.policy)))

    if model == "unconstrained_controllability":
        return max(candidates, key=lambda ev: (ev.c_unrestricted_future, -ev.movement_cost, -action_index(ev.policy)))

    if model == "viability_constrained_endpoint":
        return max(candidates, key=lambda ev: (ev.c_future, ev.sim.endpoint_h, -ev.movement_cost, -action_index(ev.policy)))

    if model == "joint_viability_expansion":
        return max(
            candidates,
            key=lambda ev: (
                cfg.alpha * ev.delta_c - cfg.eta * ev.movement_cost,
                ev.sim.endpoint_h,
                -ev.movement_cost,
                -action_index(ev.policy),
            ),
        )

    raise ValueError(f"Unknown model: {model}")


# =============================================================================
# Simulation
# =============================================================================

def run_model_on_environment(env_id: int, env: Environment, cfg: Config, model: str, policies: Sequence[Policy]) -> Dict:
    rng = random.Random(cfg.base_seed + env.source_seed * 1009 + MODEL_NAMES.index(model))

    start_state = SimState(env.start, cfg.h0, frozenset({env.start}))
    state = start_state

    c_cache: Dict[Tuple, Set[Position]] = {}
    c0 = c_value(start_state, env, cfg, True, c_cache)

    alive = True
    violation_count = 0

    sve_series: List[float] = []
    auc_surv_series: List[float] = []
    auc_series: List[float] = []
    h_series: List[float] = [cfg.h0]

    cumulative_movement_cost = 0.0
    cumulative_uncertainty_exposure = 0.0

    for _t in range(cfg.T):
        evals = evaluate_policy_space(state, env, cfg, policies, c_cache)
        chosen = select_policy(model, evals, cfg, rng)

        first_action = chosen.policy[0]
        next_pos = move_position(state.pos, first_action, env, cfg)
        next_h = update_viability(state.h, first_action, next_pos, env)

        visited = set(state.visited)
        uncertainty_step = 0.0
        if next_pos not in visited:
            uncertainty_step = 1.0
        visited.add(next_pos)

        state = SimState(next_pos, next_h, frozenset(visited))

        if next_h < cfg.hmin:
            alive = False
            violation_count += 1

        c_after = c_value(state, env, cfg, True, c_cache)
        survival_indicator = 1.0 if alive else 0.0

        sve_series.append(survival_indicator * max(0.0, c_after - c0))
        auc_surv_series.append(survival_indicator * c_after)
        auc_series.append(c_after)

        cumulative_movement_cost += movement_cost(first_action, env)
        cumulative_uncertainty_exposure += uncertainty_step
        h_series.append(next_h)

    final_c = c_value(state, env, cfg, True, c_cache)
    terminal_delta = final_c - c0

    return {
        "env_id": env_id,
        "source_seed": env.source_seed,
        "model": model,
        "joint_viable_expansion_score": float(np.mean(sve_series)),
        "survival_adjusted_auc_c_omega": float(np.mean(auc_surv_series)),
        "auc_c_omega": float(np.mean(auc_series)),
        "validity": int(violation_count == 0),
        "terminal_delta_c_omega": float(terminal_delta),
        "final_c_omega": float(final_c),
        "initial_c_omega": float(c0),
        "cumulative_viability_violations": int(violation_count),
        "cumulative_movement_cost": float(cumulative_movement_cost),
        "total_explored_cells": int(len(state.visited)),
        "cumulative_uncertainty_exposure": float(cumulative_uncertainty_exposure),
        "mean_internal_viability": float(np.mean(h_series)),
        "final_internal_viability": float(state.h),
    }


def run_one_environment(args: Tuple[int, Environment, Config]) -> List[Dict]:
    env_id, env, cfg = args
    policies = all_policies(cfg.H_pi)
    return [run_model_on_environment(env_id, env, cfg, model, policies) for model in MODEL_NAMES]


def run_all_models(envs: Sequence[Environment], cfg: Config, workers: int) -> pd.DataFrame:
    tasks = [(env_id, env, cfg) for env_id, env in enumerate(envs)]
    rows: List[Dict] = []

    if workers <= 1:
        for task in tasks:
            rows.extend(run_one_environment(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_one_environment, task) for task in tasks]
            for fut in as_completed(futures):
                rows.extend(fut.result())

    return pd.DataFrame(rows).sort_values(["env_id", "model"]).reset_index(drop=True)


def merge_diagnostics(results: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "env_id",
        "critical_regime",
        "rho_omega",
        "rho_delta",
        "kappa",
        "n_reference_states",
        "n_resources",
        "n_hazards",
        "n_obstacles",
        "move_cost",
        "resource_gain",
        "hazard_damage",
    ]
    return results.merge(diagnostics[cols], on="env_id", how="left")


# =============================================================================
# Summaries and tests
# =============================================================================

PRIMARY_OUTCOME = "joint_viable_expansion_score"
SECONDARY_OUTCOMES: Tuple[str, ...] = (
    "survival_adjusted_auc_c_omega",
    "auc_c_omega",
    "validity",
    "terminal_delta_c_omega",
    "final_c_omega",
    "cumulative_viability_violations",
    "cumulative_movement_cost",
    "total_explored_cells",
    "cumulative_uncertainty_exposure",
    "mean_internal_viability",
    "final_internal_viability",
)


def summary_by_model(results: pd.DataFrame) -> pd.DataFrame:
    outcomes = (PRIMARY_OUTCOME,) + SECONDARY_OUTCOMES
    rows: List[Dict] = []

    for model, g in results.groupby("model"):
        row: Dict = {"model": model, "n": int(len(g))}
        for outcome in outcomes:
            vals = g[outcome].to_numpy(dtype=float)
            row[f"{outcome}_mean"] = float(np.mean(vals))
            row[f"{outcome}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{outcome}_median"] = float(np.median(vals))
            row[f"{outcome}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        rows.append(row)

    return pd.DataFrame(rows)


def family_summary(results: pd.DataFrame) -> Dict:
    mean_by_model = results.groupby("model")[PRIMARY_OUTCOME].mean().to_dict()

    family_mean = float(np.mean([mean_by_model[m] for m in FAMILY_MODELS if m in mean_by_model]))
    incomplete_mean = float(np.mean([mean_by_model[m] for m in INCOMPLETE_MODELS if m in mean_by_model]))
    joint_mean = float(mean_by_model.get(MAIN_MODEL, np.nan))
    endpoint_mean = float(mean_by_model.get("viability_constrained_endpoint", np.nan))

    return {
        "family_mean_psi": family_mean,
        "incomplete_mean_psi": incomplete_mean,
        "family_minus_incomplete_mean_psi": family_mean - incomplete_mean,
        "joint_mean_psi": joint_mean,
        "endpoint_mean_psi": endpoint_mean,
        "joint_minus_endpoint_mean_psi": joint_mean - endpoint_mean,
    }


def paired_permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    obs = float(np.mean(diff))

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diff)), replace=True)
    null = np.mean(signs * diff[None, :], axis=1)

    p_one = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    p_two = float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))

    return obs, p_one, p_two


def bootstrap_ci_mean_difference(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(diff)

    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(diff[idx]))

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pairwise_tests(results: pd.DataFrame, cfg: Config, outcome: str, analysis_label: str) -> pd.DataFrame:
    baselines = [m for m in MODEL_NAMES if m != MAIN_MODEL]
    pivot = results.pivot(index="env_id", columns="model", values=outcome)

    rng = np.random.default_rng(cfg.base_seed + 20260511 + stable_label_seed(outcome + analysis_label))

    rows: List[Dict] = []
    for baseline in baselines:
        paired = pivot[[MAIN_MODEL, baseline]].dropna()
        if len(paired) < 2:
            continue

        x = paired[MAIN_MODEL].to_numpy(dtype=float)
        y = paired[baseline].to_numpy(dtype=float)

        diff, p_one, p_two = paired_permutation_test(x, y, cfg.n_permutations, rng)
        lo, hi = bootstrap_ci_mean_difference(x, y, cfg.n_bootstrap, rng)

        rows.append({
            "analysis": analysis_label,
            "outcome": outcome,
            "comparison": f"{MAIN_MODEL}_minus_{baseline}",
            "n_env": int(len(paired)),
            "paired_mean_difference": diff,
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "p_one_sided_main_greater": p_one,
            "p_two_sided": p_two,
        })

    return pd.DataFrame(rows)


def family_vs_incomplete_tests(results: pd.DataFrame, cfg: Config, analysis_label: str) -> pd.DataFrame:
    """
    Environment-level family score is the mean primary score of the two family models.
    Environment-level incomplete score is the mean primary score of the five incomplete models.
    """
    pivot = results.pivot(index="env_id", columns="model", values=PRIMARY_OUTCOME)
    needed = list(FAMILY_MODELS) + list(INCOMPLETE_MODELS)
    paired = pivot[needed].dropna()

    if len(paired) < 2:
        return pd.DataFrame()

    family = paired[list(FAMILY_MODELS)].mean(axis=1).to_numpy(dtype=float)
    incomplete = paired[list(INCOMPLETE_MODELS)].mean(axis=1).to_numpy(dtype=float)

    rng = np.random.default_rng(cfg.base_seed + 606 + stable_label_seed(analysis_label))
    diff, p_one, p_two = paired_permutation_test(family, incomplete, cfg.n_permutations, rng)
    lo, hi = bootstrap_ci_mean_difference(family, incomplete, cfg.n_bootstrap, rng)

    return pd.DataFrame([{
        "analysis": analysis_label,
        "comparison": "viability_constrained_expansion_family_minus_incomplete_models",
        "n_env": int(len(paired)),
        "paired_mean_difference": diff,
        "bootstrap95_low": lo,
        "bootstrap95_high": hi,
        "p_one_sided_family_greater": p_one,
        "p_two_sided": p_two,
    }])


def run_statistical_tests(results: pd.DataFrame, cfg: Config, analysis_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pairwise_tests(results, cfg, PRIMARY_OUTCOME, analysis_label)
    family = family_vs_incomplete_tests(results, cfg, analysis_label)

    secondary_parts = [pairwise_tests(results, cfg, outcome, analysis_label) for outcome in SECONDARY_OUTCOMES]
    secondary = pd.concat(secondary_parts, ignore_index=True) if secondary_parts else pd.DataFrame()

    if len(secondary):
        secondary["p_two_sided_fdr"] = bh_fdr(secondary["p_two_sided"].to_numpy())

    return primary, secondary, family


def pareto_summary(results: pd.DataFrame) -> pd.DataFrame:
    tab = results.groupby("model").agg(
        joint_viable_expansion_score=(PRIMARY_OUTCOME, "mean"),
        validity=("validity", "mean"),
        cumulative_movement_cost=("cumulative_movement_cost", "mean"),
    ).reset_index()

    rows: List[Dict] = []
    for _, row in tab.iterrows():
        dominated = False
        dominated_by = ""

        for _, other in tab.iterrows():
            if other["model"] == row["model"]:
                continue

            at_least_as_good = (
                other["joint_viable_expansion_score"] >= row["joint_viable_expansion_score"]
                and other["validity"] >= row["validity"]
                and other["cumulative_movement_cost"] <= row["cumulative_movement_cost"]
            )
            strictly_better = (
                other["joint_viable_expansion_score"] > row["joint_viable_expansion_score"]
                or other["validity"] > row["validity"]
                or other["cumulative_movement_cost"] < row["cumulative_movement_cost"]
            )

            if at_least_as_good and strictly_better:
                dominated = True
                dominated_by = str(other["model"])
                break

        out = row.to_dict()
        out["pareto_dominated"] = bool(dominated)
        out["dominated_by"] = dominated_by
        rows.append(out)

    return pd.DataFrame(rows)


# =============================================================================
# Parameter scan
# =============================================================================

def scan_configurations(base: Config, n_env: int, n_perm: int, n_boot: int) -> List[Tuple[str, Config]]:
    """
    Prespecified one-factor-at-a-time parameter scan plus weight grid and threshold grid.
    The default setting is included once.
    """
    configs: List[Tuple[str, Config]] = []

    def add(label: str, cfg: Config) -> None:
        cfg = replace(cfg, n_env=n_env, n_permutations=n_perm, n_bootstrap=n_boot)
        labels = [x[0] for x in configs]
        if label not in labels:
            configs.append((label, cfg))

    add("default", base)

    for H_pi in (3, 4, 5, 6):
        add(f"Hpi_{H_pi}", replace(base, H_pi=H_pi))

    for H_c in (3, 4, 5, 6):
        add(f"Hc_{H_c}", replace(base, H_c=H_c))

    for hmin in (0.20, 0.25, 0.30):
        add(f"hmin_{hmin:g}", replace(base, hmin=hmin))

    for alpha in (0.5, 1.0, 2.0):
        for eta in (0.05, 0.1, 0.2):
            add(f"alpha_{alpha:g}_eta_{eta:g}", replace(base, alpha=alpha, eta=eta))

    for rho_omega in (0.10, 0.20, 0.30):
        for rho_delta in (0.10, 0.20, 0.30):
            for kappa in (0.025, 0.05, 0.10):
                add(
                    f"thr_ro_{rho_omega:g}_rd_{rho_delta:g}_k_{kappa:g}",
                    replace(
                        base,
                        threshold_rho_omega=rho_omega,
                        threshold_rho_delta=rho_delta,
                        threshold_kappa=kappa,
                    ),
                )

    return configs


def run_one_analysis(
    label: str,
    cfg: Config,
    outdir: Path,
    workers: int,
    critical_only: bool,
    save_per_environment: bool,
) -> Dict:
    setting_dir = outdir / label
    setting_dir.mkdir(parents=True, exist_ok=True)

    envs, diagnostics, screening = collect_environments(cfg, workers, critical_only=critical_only)
    results = run_all_models(envs, cfg, workers)
    results = merge_diagnostics(results, diagnostics)

    summary = summary_by_model(results)
    primary, secondary, family_tests = run_statistical_tests(results, cfg, label)
    pareto = pareto_summary(results)

    diagnostics.to_csv(setting_dir / "accepted_environment_diagnostics.csv", index=False)
    screening.to_csv(setting_dir / "screening_log.csv", index=False)
    summary.to_csv(setting_dir / "summary_by_model.csv", index=False)
    primary.to_csv(setting_dir / "primary_pairwise_tests.csv", index=False)
    secondary.to_csv(setting_dir / "secondary_pairwise_tests_fdr.csv", index=False)
    family_tests.to_csv(setting_dir / "family_vs_incomplete_tests.csv", index=False)
    pareto.to_csv(setting_dir / "pareto_summary.csv", index=False)

    if save_per_environment:
        results.to_csv(setting_dir / "per_environment_results.csv", index=False)

    fs = family_summary(results)

    config_row = cfg.__dict__.copy()
    config_row["analysis_label"] = label
    config_row["critical_only"] = critical_only
    pd.DataFrame([config_row]).to_csv(setting_dir / "config_used.csv", index=False)

    out = {
        "analysis_label": label,
        "n_env": cfg.n_env,
        "T": cfg.T,
        "H_pi": cfg.H_pi,
        "H_c": cfg.H_c,
        "H_ref": cfg.H_ref,
        "h0": cfg.h0,
        "hmin": cfg.hmin,
        "alpha": cfg.alpha,
        "eta": cfg.eta,
        "threshold_rho_omega": cfg.threshold_rho_omega,
        "threshold_rho_delta": cfg.threshold_rho_delta,
        "threshold_kappa": cfg.threshold_kappa,
        "accepted_rho_omega_mean": float(diagnostics["rho_omega"].mean()),
        "accepted_rho_delta_mean": float(diagnostics["rho_delta"].mean()),
        "accepted_kappa_mean": float(diagnostics["kappa"].mean()),
    }
    out.update(fs)

    if len(family_tests):
        out["family_vs_incomplete_difference"] = float(family_tests["paired_mean_difference"].iloc[0])
        out["family_vs_incomplete_ci_low"] = float(family_tests["bootstrap95_low"].iloc[0])
        out["family_vs_incomplete_ci_high"] = float(family_tests["bootstrap95_high"].iloc[0])
        out["family_vs_incomplete_p_one_sided"] = float(family_tests["p_one_sided_family_greater"].iloc[0])
    else:
        out["family_vs_incomplete_difference"] = np.nan
        out["family_vs_incomplete_ci_low"] = np.nan
        out["family_vs_incomplete_ci_high"] = np.nan
        out["family_vs_incomplete_p_one_sided"] = np.nan

    for model in MODEL_NAMES:
        vals = results.loc[results["model"] == model, PRIMARY_OUTCOME]
        out[f"{model}_psi_mean"] = float(vals.mean())

    return out


def run_parameter_scan(
    base_cfg: Config,
    outdir: Path,
    workers: int,
    critical_only: bool,
    scan_envs: int,
    scan_permutations: int,
    scan_bootstraps: int,
    save_per_environment: bool,
) -> pd.DataFrame:
    scan_dir = outdir / "parameter_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    settings = scan_configurations(base_cfg, scan_envs, scan_permutations, scan_bootstraps)
    rows: List[Dict] = []

    for i, (label, cfg) in enumerate(settings, start=1):
        print(f"\n[parameter_scan] {i}/{len(settings)}: {label}")
        row = run_one_analysis(
            label=label,
            cfg=cfg,
            outdir=scan_dir,
            workers=workers,
            critical_only=critical_only,
            save_per_environment=save_per_environment,
        )
        rows.append(row)

        pd.DataFrame(rows).to_csv(scan_dir / "parameter_scan_summary_incremental.csv", index=False)

    scan_summary = pd.DataFrame(rows)
    scan_summary.to_csv(scan_dir / "parameter_scan_summary.csv", index=False)

    return scan_summary


# =============================================================================
# Figures
# =============================================================================

def save_scan_figure(scan_summary: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "parameter_scan" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    x = np.arange(len(scan_summary))
    plt.bar(x, scan_summary["family_minus_incomplete_mean_psi"].to_numpy(dtype=float))
    plt.xticks(x, scan_summary["analysis_label"].tolist(), rotation=90, fontsize=6)
    plt.ylabel("Family - incomplete models, mean Ψ")
    plt.title("Parameter scan: viability-constrained expansion family advantage")
    plt.tight_layout()
    plt.savefig(figdir / "parameter_scan_family_advantage.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(x, scan_summary["joint_minus_endpoint_mean_psi"].to_numpy(dtype=float))
    plt.xticks(x, scan_summary["analysis_label"].tolist(), rotation=90, fontsize=6)
    plt.ylabel("Joint - endpoint, mean Ψ")
    plt.title("Parameter scan: joint model versus endpoint family member")
    plt.tight_layout()
    plt.savefig(figdir / "parameter_scan_joint_vs_endpoint.png", dpi=300)
    plt.close()


def save_full_figures(results: pd.DataFrame, diagnostics: pd.DataFrame, pareto: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "full_analysis" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    order = list(MODEL_NAMES)

    def bar_plot(outcome: str, ylabel: str, title: str, filename: str) -> None:
        means = [results.loc[results["model"] == m, outcome].mean() for m in order]
        sems = [
            results.loc[results["model"] == m, outcome].std(ddof=1) / math.sqrt(max(1, (results["model"] == m).sum()))
            for m in order
        ]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(order)), means, yerr=sems, capsize=3)
        plt.xticks(range(len(order)), order, rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=300)
        plt.close()

    bar_plot(PRIMARY_OUTCOME, "Joint viable-expansion score Ψ", "Primary outcome", "fig1_joint_viable_expansion_score.png")
    bar_plot("survival_adjusted_auc_c_omega", "Survival-adjusted AUC-CΩ", "Survival-adjusted viable controllability", "fig2_survival_adjusted_auc_c_omega.png")
    bar_plot("validity", "Validity probability", "Validity probability", "fig3_validity_probability.png")
    bar_plot("cumulative_viability_violations", "Cumulative viability violations", "Viability violations", "fig4_viability_violations.png")
    bar_plot("terminal_delta_c_omega", "Terminal ΔCΩ", "Terminal viable-controllability gain", "fig5_terminal_delta_c_omega.png")
    bar_plot("cumulative_movement_cost", "Cumulative movement cost", "Movement cost", "fig6_movement_cost.png")
    bar_plot("cumulative_uncertainty_exposure", "Cumulative uncertainty exposure", "Uncertainty exposure", "fig7_uncertainty_exposure.png")

    for col, title, fname in [
        ("rho_omega", "Viability-risk index ρΩ", "fig8_rho_omega_distribution.png"),
        ("rho_delta", "Expansion-opportunity index ρΔ", "fig9_rho_delta_distribution.png"),
        ("kappa", "Viability-control conflict index κ", "fig10_kappa_distribution.png"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.hist(diagnostics[col].dropna().to_numpy(dtype=float), bins=20)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figdir / fname, dpi=300)
        plt.close()

    plt.figure(figsize=(7, 5))
    for _, row in pareto.iterrows():
        plt.scatter(row["cumulative_movement_cost"], row["joint_viable_expansion_score"])
        plt.text(row["cumulative_movement_cost"], row["joint_viable_expansion_score"], str(row["model"]), fontsize=8)
    plt.xlabel("Cumulative movement cost")
    plt.ylabel("Joint viable-expansion score Ψ")
    plt.title("Pareto plane")
    plt.tight_layout()
    plt.savefig(figdir / "fig11_pareto_plane.png", dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter-scan-first full analysis for viability-preserving expansion.")

    p.add_argument("--outdir", required=True, type=str)

    p.add_argument("--scan-envs", default=50, type=int)
    p.add_argument("--full-envs", default=1000, type=int)
    p.add_argument("--T", default=120, type=int)

    p.add_argument("--Hpi", "--H-pi", dest="Hpi", default=4, type=int)
    p.add_argument("--Hc", "--H-c", dest="Hc", default=4, type=int)
    p.add_argument("--Href", "--H-ref", dest="Href", default=8, type=int)
    p.add_argument("--max-reference-states", default=32, type=int)

    p.add_argument("--h0", default=0.60, type=float)
    p.add_argument("--hmin", "--h-min", dest="hmin", default=0.25, type=float)

    p.add_argument("--alpha", default=1.0, type=float)
    p.add_argument("--eta", default=0.1, type=float)

    p.add_argument("--threshold-rho-omega", default=0.20, type=float)
    p.add_argument("--threshold-rho-delta", default=0.20, type=float)
    p.add_argument("--threshold-kappa", default=0.05, type=float)

    p.add_argument("--workers", default=1, type=int)
    p.add_argument("--base-seed", default=12345, type=int)

    p.add_argument("--scan-permutations", default=1000, type=int)
    p.add_argument("--full-permutations", default=10000, type=int)
    p.add_argument("--bootstraps", default=1000, type=int)
    p.add_argument("--full-bootstraps", default=10000, type=int)

    p.add_argument("--max-candidates", default=300000, type=int)
    p.add_argument("--screen-batch-size", default=128, type=int)

    p.add_argument("--all-environments", action="store_true")
    p.add_argument("--skip-scan", action="store_true")
    p.add_argument("--skip-full", action="store_true")
    p.add_argument("--save-scan-per-environment", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    critical_only = not args.all_environments

    base_cfg = Config(
        T=args.T,
        H_pi=args.Hpi,
        H_c=args.Hc,
        H_ref=args.Href,
        max_reference_states=args.max_reference_states,
        h0=args.h0,
        hmin=args.hmin,
        alpha=args.alpha,
        eta=args.eta,
        threshold_rho_omega=args.threshold_rho_omega,
        threshold_rho_delta=args.threshold_rho_delta,
        threshold_kappa=args.threshold_kappa,
        base_seed=args.base_seed,
        max_candidates=args.max_candidates,
        screen_batch_size=args.screen_batch_size,
    )

    print("[analysis] parameter-scan-first full analysis")
    print(f"[analysis] outdir={outdir}")
    print(f"[analysis] critical_only={critical_only}")

    if not args.skip_scan:
        scan_cfg = replace(base_cfg, n_permutations=args.scan_permutations, n_bootstrap=args.bootstraps)
        scan_summary = run_parameter_scan(
            base_cfg=scan_cfg,
            outdir=outdir,
            workers=args.workers,
            critical_only=critical_only,
            scan_envs=args.scan_envs,
            scan_permutations=args.scan_permutations,
            scan_bootstraps=args.bootstraps,
            save_per_environment=args.save_scan_per_environment,
        )
        save_scan_figure(scan_summary, outdir)
    else:
        print("[analysis] scan skipped")

    if not args.skip_full:
        print("\n[full_analysis] starting prespecified default full analysis")
        full_dir = outdir / "full_analysis"
        full_cfg = replace(
            base_cfg,
            n_env=args.full_envs,
            n_permutations=args.full_permutations,
            n_bootstrap=args.full_bootstraps,
        )

        row = run_one_analysis(
            label="default_full",
            cfg=full_cfg,
            outdir=outdir / "full_analysis",
            workers=args.workers,
            critical_only=critical_only,
            save_per_environment=True,
        )

        results = pd.read_csv(full_dir / "default_full" / "per_environment_results.csv")
        diagnostics = pd.read_csv(full_dir / "default_full" / "accepted_environment_diagnostics.csv")
        pareto = pd.read_csv(full_dir / "default_full" / "pareto_summary.csv")
        save_full_figures(results, diagnostics, pareto, outdir)

        pd.DataFrame([row]).to_csv(full_dir / "full_analysis_summary_row.csv", index=False)
        print("[full_analysis] complete")
    else:
        print("[analysis] full analysis skipped")

    print("[analysis] done")
    print(f"[analysis] output: {outdir}")


if __name__ == "__main__":
    main()
