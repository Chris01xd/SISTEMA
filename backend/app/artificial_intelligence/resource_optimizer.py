"""Resource reallocation optimization: GA (primary), RL (sequential policy),
and a small MILP baseline for validation. Rationale in docs/ARCHITECTURE.md
section 6.4. Human-in-the-loop by design: this module returns *recommendations*,
never auto-applies changes to the live hospital state.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from deap import algorithms, base, creator, tools

try:
    import pulp
except ImportError:
    pulp = None


@dataclass
class AllocationCandidate:
    """A candidate reallocation: how many overflow beds to open per ward,
    and how many on-call staff to recall, per specialty."""

    overflow_beds_opened: int
    on_call_staff_recalled: int
    expected_wait_reduction_minutes: float
    staffing_overtime_cost: float


def genetic_optimize(
    max_overflow_beds: int,
    max_recallable_staff: int,
    current_waiting: int,
    current_saturation: float,
    population_size: int = 40,
    generations: int = 30,
    seed: int | None = 42,
) -> AllocationCandidate:
    """Multi-objective GA: minimize (expected wait) and (overtime cost).

    Chosen over gradient methods because the decision variables are discrete
    (bed counts, staff headcount) and the objective is non-differentiable and
    multi-modal (see docs/ARCHITECTURE.md section 6.4).
    """
    random.seed(seed)

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_beds", random.randint, 0, max_overflow_beds)
    toolbox.register("attr_staff", random.randint, 0, max_recallable_staff)
    toolbox.register(
        "individual", tools.initCycle, creator.Individual, (toolbox.attr_beds, toolbox.attr_staff), n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        beds_opened, staff_recalled = ind
        capacity_relief = beds_opened * 1.0 + staff_recalled * 0.6
        expected_wait = max(0.0, current_waiting * (1.0 - current_saturation) - capacity_relief) * 4.0
        overtime_cost = staff_recalled * 150.0 + beds_opened * 40.0
        return expected_wait, overtime_cost

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=max(max_overflow_beds, max_recallable_staff), indpb=0.3)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(
        pop, toolbox, mu=population_size, lambda_=population_size,
        cxpb=0.5, mutpb=0.3, ngen=generations, verbose=False,
    )
    best = tools.selBest(pop, 1)[0]
    wait_reduction, cost = evaluate(best)
    return AllocationCandidate(
        overflow_beds_opened=best[0],
        on_call_staff_recalled=best[1],
        expected_wait_reduction_minutes=wait_reduction,
        staffing_overtime_cost=cost,
    )


class QLearningAssignmentPolicy:
    """Tabular Q-learning policy for sequential bed/OR assignment.

    State: discretized (queue_length_bucket, saturation_bucket).
    Action: {PRIORITIZE_CRITICAL, PRIORITIZE_FIFO, HOLD_FOR_OR}.
    Reward: -wait_time_incurred - penalty_per_death.

    Runs *in place of* the ABM's greedy baseline when
    Simulation.assignment_strategy == "rl" (see app/simulation/abm/model.py).
    This is intentionally a small, auditable tabular policy rather than a deep
    RL agent: assignment-policy decisions in a clinical context must remain
    inspectable (dump the Q-table, verify no policy state prefers letting a
    critical patient wait over a minor one).
    """

    ACTIONS = ["PRIORITIZE_CRITICAL", "PRIORITIZE_FIFO", "HOLD_FOR_OR"]

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple, dict[str, float]] = {}

    def _state_key(self, queue_length: int, saturation_index: float) -> tuple:
        q_bucket = min(queue_length // 5, 10)
        s_bucket = int(saturation_index * 10)
        return (q_bucket, s_bucket)

    def select_action(self, queue_length: int, saturation_index: float) -> str:
        key = self._state_key(queue_length, saturation_index)
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.ACTIONS}
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        return max(self.q_table[key], key=self.q_table[key].get)

    def update(self, state_key: tuple, action: str, reward: float, next_state_key: tuple) -> None:
        self.q_table.setdefault(state_key, {a: 0.0 for a in self.ACTIONS})
        self.q_table.setdefault(next_state_key, {a: 0.0 for a in self.ACTIONS})
        best_next = max(self.q_table[next_state_key].values())
        td_target = reward + self.gamma * best_next
        self.q_table[state_key][action] += self.alpha * (td_target - self.q_table[state_key][action])


def milp_validate_assignment(patients: list[dict], beds: list[dict]) -> dict:
    """Small-scale exact MILP assignment (<= ~50 patients) used only to validate
    the GA/RL/greedy heuristics against a provable optimum (see architecture doc
    section 6.4). patients: [{"id", "priority"}]; beds: [{"id", "compatible_ids"}].
    """
    if pulp is None:
        raise RuntimeError("pulp is not installed; MILP validation is optional/offline-only")

    prob = pulp.LpProblem("bed_assignment_validation", pulp.LpMaximize)
    x = {
        (p["id"], b["id"]): pulp.LpVariable(f"x_{p['id']}_{b['id']}", cat="Binary")
        for p in patients
        for b in beds
        if p["id"] in b.get("compatible_ids", [p["id"]])
    }
    prob += pulp.lpSum(x[(p["id"], b["id"])] * p["priority"] for (p_id, b_id), _ in x.items() for p in patients if p["id"] == p_id for b in beds if b["id"] == b_id)
    for p in patients:
        vars_for_p = [v for (pid, _bid), v in x.items() if pid == p["id"]]
        if vars_for_p:
            prob += pulp.lpSum(vars_for_p) <= 1
    for b in beds:
        vars_for_b = [v for (_pid, bid), v in x.items() if bid == b["id"]]
        if vars_for_b:
            prob += pulp.lpSum(vars_for_b) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    assignment = {pid: bid for (pid, bid), v in x.items() if v.value() == 1}
    return {"status": pulp.LpStatus[prob.status], "assignment": assignment}
