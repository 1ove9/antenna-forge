# ============================================================
# REFERENCE
#   仿造来源：pymoo @ https://github.com/anyoptimization/pymoo
#   对标文件：pymoo/algorithms/moo/nsga2.py
#   对标类/函数：NSGA2, NonDominatedSorting, CrowdingDistance
#   关键设计点：
#     - 非支配排序（O(M N²)）
#     - 拥挤度距离（边界解赋无穷大）
#     - 锦标赛选择（支配关系优先，其次拥挤度）
#     - 模拟二进交叉（SBX）+ 多项式变异
#     - 精英策略（父代+子代合并排序）
#   YAF 的差异化改造：
#     - 纯 NumPy 实现（零 pymoo 依赖）
#     - 天线双目标专用（增益最大化 + VSWR 最小化）
#     - --demo：二维测试函数（ZDT1）
#     - 简化的实数编码（边界 [0,1]^n）
# ============================================================

"""NSGA-II — Non-dominated Sorting Genetic Algorithm II.

Multi-objective optimization for antenna design (gain vs VSWR,
bandwidth vs size, etc.).

Usage:
    python -m yaf_ai.optimization.nsga --demo
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np


class NSGA2:
    """NSGA-II algorithm for multi-objective optimization.

    Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II", IEEE TEC, 2002.
    """

    def __init__(
        self,
        n_vars: int,
        n_obj: int,
        pop_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float | None = None,  # defaults to 1/n_vars
        eta_cross: float = 20.0,
        eta_mut: float = 20.0,
        bounds: np.ndarray | None = None,
    ) -> None:
        """Initialize NSGA-II.

        Args:
            n_vars: Number of decision variables.
            n_obj: Number of objectives.
            pop_size: Population size (even number).
            n_generations: Number of generations.
            crossover_prob: SBX crossover probability.
            mutation_prob: Polynomial mutation probability (default: 1/n_vars).
            eta_cross: SBX distribution index.
            eta_mut: Mutation distribution index.
            bounds: (n_vars, 2) lower/upper bounds, default [0, 1].
        """
        self.n_vars = n_vars
        self.n_obj = n_obj
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob if mutation_prob is not None else 1.0 / n_vars
        self.eta_cross = eta_cross
        self.eta_mut = eta_mut

        if bounds is None:
            self.bounds = np.column_stack([np.zeros(n_vars), np.ones(n_vars)])
        else:
            self.bounds = bounds

    def _initialize_population(self) -> np.ndarray:
        """Random initialization within bounds."""
        pop = np.random.random((self.pop_size, self.n_vars))
        for j in range(self.n_vars):
            pop[:, j] = (
                pop[:, j] * (self.bounds[j, 1] - self.bounds[j, 0])
                + self.bounds[j, 0]
            )
        return pop

    def _evaluate(
        self, population: np.ndarray, objective_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Evaluate objective function for population.

        Args:
            population: (pop_size, n_vars).
            objective_fn: function(x) -> (n_obj,) array.

        Returns:
            Objective values (pop_size, n_obj).
        """
        obj = np.zeros((len(population), self.n_obj))
        for i, ind in enumerate(population):
            obj[i] = objective_fn(ind)
        return obj

    def _non_dominated_sort(self, obj: np.ndarray) -> list[np.ndarray]:
        """Non-dominated sorting (O(M N²)).

        Args:
            obj: (N, n_obj) objective values.

        Returns:
            List of fronts, each an index array.
        """
        n = len(obj)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = []

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # p dominates q?
                if np.all(obj[p] <= obj[q]) and np.any(obj[p] < obj[q]):
                    dominated_solutions[p].append(q)
                elif np.all(obj[q] <= obj[p]) and np.any(obj[q] < obj[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front: list[int] = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            if len(next_front) > 0:
                fronts.append(next_front)

        return [np.array(f) for f in fronts if len(f) > 0]

    def _crowding_distance(self, obj: np.ndarray, front: np.ndarray) -> np.ndarray:
        """Compute crowding distance for a front.

        Args:
            obj: Full objective array.
            front: Index array for this front.

        Returns:
            Crowding distance array (size of front).
        """
        n_f = len(front)
        distances = np.zeros(n_f)

        if n_f <= 2:
            distances[:] = np.inf
            return distances

        for m in range(self.n_obj):
            f_obj = obj[front, m]
            order = np.argsort(f_obj)
            sorted_front = front[order]

            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            f_min = f_obj[order[0]]
            f_max = f_obj[order[-1]]
            if f_max - f_min < 1e-12:
                continue

            for i in range(1, n_f - 1):
                distances[order[i]] += (
                    (f_obj[order[i + 1]] - f_obj[order[i - 1]])
                    / (f_max - f_min)
                )

        return distances

    def _tournament_selection(
        self, population: np.ndarray, obj: np.ndarray, fronts: list[np.ndarray],
        crowding: list[np.ndarray],
    ) -> np.ndarray:
        """Binary tournament selection.

        Args:
            population: Full combined population.
            obj: Objective values.
            fronts: List of front index arrays.
            crowding: List of crowding distance arrays.

        Returns:
            Selected population (pop_size, n_vars).
        """
        n_total = len(population)
        rank = np.zeros(n_total, dtype=int)
        crowd = np.zeros(n_total)

        for i, (f, c) in enumerate(zip(fronts, crowding)):
            rank[f] = i
            crowd[f] = c

        selected = np.zeros((self.pop_size, self.n_vars))

        for i in range(self.pop_size):
            a, b = np.random.choice(n_total, 2, replace=False)
            if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] > crowd[b]):
                selected[i] = population[a]
            else:
                selected[i] = population[b]

        return selected

    def _sbx_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX).

        Args:
            parent1, parent2: Two parent vectors.

        Returns:
            Two offspring vectors.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if np.random.random() > self.crossover_prob:
            return child1, child2

        for j in range(self.n_vars):
            if np.random.random() > 0.5:
                if abs(parent1[j] - parent2[j]) > 1e-14:
                    y1 = min(parent1[j], parent2[j])
                    y2 = max(parent1[j], parent2[j])

                    bl = self.bounds[j, 0]
                    bu = self.bounds[j, 1]

                    rand = np.random.random()
                    beta = 1.0 + 2.0 * (y1 - bl) / (y2 - y1 + 1e-12)
                    alpha = 2.0 - beta ** (-(self.eta_cross + 1))

                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (self.eta_cross + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (
                            1.0 / (self.eta_cross + 1)
                        )

                    c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                    c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                    child1[j] = np.clip(c1, bl, bu)
                    child2[j] = np.clip(c2, bl, bu)

        return child1, child2

    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation.

        Args:
            individual: Individual to mutate.

        Returns:
            Mutated individual.
        """
        mutant = individual.copy()

        for j in range(self.n_vars):
            if np.random.random() <= self.mutation_prob:
                bl = self.bounds[j, 0]
                bu = self.bounds[j, 1]
                delta = min(individual[j] - bl, bu - individual[j]) / (bu - bl + 1e-12)

                rand = np.random.random()
                if rand <= 0.5:
                    delta_q = (
                        (2 * rand + (1 - 2 * rand) * (1 - delta) ** (self.eta_mut + 1))
                        ** (1.0 / (self.eta_mut + 1))
                        - 1
                    )
                else:
                    delta_q = (
                        1
                        - (
                            2 * (1 - rand)
                            + 2 * (rand - 0.5) * (1 - delta) ** (self.eta_mut + 1)
                        )
                        ** (1.0 / (self.eta_mut + 1))
                    )

                mutant[j] = np.clip(
                    individual[j] + delta_q * (bu - bl), bl, bu
                )

        return mutant

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run NSGA-II optimization.

        Args:
            objective_fn: function(x) -> (n_obj,) objective vector.
            verbose: Print progress.

        Returns:
            (pareto_front_vars, pareto_front_obj, history_obj).
            history_obj is the objective values of all final-front individuals.
        """
        # Initialize
        parent_pop = self._initialize_population()
        parent_obj = self._evaluate(parent_pop, objective_fn)

        for gen in range(self.n_generations):
            # Non-dominated sorting of parents
            fronts = self._non_dominated_sort(parent_obj)
            crowding = [self._crowding_distance(parent_obj, f) for f in fronts]

            # Tournament selection
            selected = self._tournament_selection(
                parent_pop, parent_obj, fronts, crowding
            )

            # Crossover
            offspring = np.zeros_like(selected)
            for i in range(0, self.pop_size, 2):
                c1, c2 = self._sbx_crossover(selected[i], selected[i + 1])
                offspring[i] = c1
                offspring[i + 1] = c2

            # Mutation
            for i in range(self.pop_size):
                offspring[i] = self._polynomial_mutation(offspring[i])

            # Evaluate offspring
            offspring_obj = self._evaluate(offspring, objective_fn)

            # Merge parents and offspring
            combined_pop = np.vstack([parent_pop, offspring])
            combined_obj = np.vstack([parent_obj, offspring_obj])

            # Non-dominated sort of combined
            combined_fronts = self._non_dominated_sort(combined_obj)
            combined_crowding = [
                self._crowding_distance(combined_obj, f) for f in combined_fronts
            ]

            # Select next generation
            next_pop = np.zeros((self.pop_size, self.n_vars))
            next_obj = np.zeros((self.pop_size, self.n_obj))
            count = 0

            for f_idx, front in enumerate(combined_fronts):
                if count + len(front) <= self.pop_size:
                    next_pop[count : count + len(front)] = combined_pop[front]
                    next_obj[count : count + len(front)] = combined_obj[front]
                    count += len(front)
                else:
                    remaining = self.pop_size - count
                    cd = combined_crowding[f_idx]
                    order = np.argsort(cd)[::-1]
                    selected_front = front[order[:remaining]]
                    next_pop[count:] = combined_pop[selected_front]
                    next_obj[count:] = combined_obj[selected_front]
                    break

            parent_pop = next_pop
            parent_obj = next_obj

            if verbose and gen % 10 == 0:
                fronts_final = self._non_dominated_sort(parent_obj)
                print(
                    f"  Gen {gen:3d}: |Front1|={len(fronts_final[0])}"
                )

        # Final Pareto front
        final_fronts = self._non_dominated_sort(parent_obj)
        pareto_idx = final_fronts[0]

        return parent_pop[pareto_idx], parent_obj[pareto_idx], parent_obj


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 test problem.

    Args:
        x: Decision vector in [0, 1]^n.

    Returns:
        (f1, f2) objectives.
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])


def demo() -> None:
    """Quick demo: NSGA-II on ZDT1."""
    print("=" * 50)
    print("  NSGA-II Multi-Objective Optimization Demo (ZDT1)")
    print("=" * 50)

    nsga = NSGA2(
        n_vars=30,
        n_obj=2,
        pop_size=50,
        n_generations=30,
    )

    print(f"Variables: {nsga.n_vars}, Population: {nsga.pop_size}")
    print(f"Generations: {nsga.n_generations}")

    pareto_vars, pareto_obj, _ = nsga.optimize(zdt1, verbose=True)

    print(f"\n  Pareto front size: {len(pareto_obj)}")
    print(f"  Objective ranges:")
    print(f"    f1: [{pareto_obj[:, 0].min():.4f}, {pareto_obj[:, 0].max():.4f}]")
    print(f"    f2: [{pareto_obj[:, 1].min():.4f}, {pareto_obj[:, 1].max():.4f}]")

    # Hypervolume estimate (reference point [2, 2])
    hv = np.mean((2 - pareto_obj[:, 0]) * (2 - pareto_obj[:, 1]))
    print(f"  Hypervolume estimate: {hv:.4f}")

    print("  Demo complete.\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="NSGA-II Optimization")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--n-vars", type=int, default=30)
    parser.add_argument("--n-generations", type=int, default=30)

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        demo()
    else:
        nsga = NSGA2(
            n_vars=args.n_vars,
            n_obj=2,
            n_generations=args.n_generations,
        )
        nsga.optimize(zdt1)


if __name__ == "__main__":
    main()
