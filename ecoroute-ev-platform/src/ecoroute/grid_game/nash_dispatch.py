"""
nash_dispatch.py
=================
Multi-Agent Grid Demand Matching (Game Theory Optimization).

When multiple independent fleets (e.g. two competing delivery operators)
share the same public charging infrastructure, a single-fleet optimizer
(`optimizer.grid_dispatcher`) is the wrong model -- each fleet is a
self-interested agent that will greedily grab the cheapest slots, causing
convoy effects and price spikes at peak hours.

This module implements a simplified simultaneous-move congestion game with
dynamic, load-responsive pricing, and iterates a best-response / fictitious-
play procedure toward an approximate Nash equilibrium: at equilibrium, no
fleet can unilaterally re-time or re-route a charging request to reduce its
own cost, given the others' current requests.

This is intentionally a *tractable, explainable* approximation (best-
response dynamics with a congestion-pricing feedback loop) rather than a
full general-sum equilibrium solver -- appropriate for an operational
dispatch system that must return an answer in milliseconds, not a
game-theory research artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ChargeRequest:
    fleet_id: str
    vehicle_id: str
    station_id: str
    preferred_slot: int  # discretized time slot index, e.g. 15-min buckets
    flexibility: int = 2  # +/- how many slots the vehicle can shift
    energy_needed_kwh: float = 30.0


@dataclass
class StationSlotState:
    base_price_per_kwh: float
    capacity_kw: float
    congestion_sensitivity: float = 0.08  # price rise per kWh of excess demand


@dataclass
class SettledRequest:
    fleet_id: str
    vehicle_id: str
    station_id: str
    assigned_slot: int
    price_per_kwh: float
    cost: float


@dataclass
class EquilibriumResult:
    settled: List[SettledRequest]
    iterations: int
    converged: bool
    slot_prices: Dict[Tuple[str, int], float] = field(default_factory=dict)


class MultiFleetNashDispatcher:
    """
    Runs iterative best-response dynamics: on each round, every fleet's
    vehicles independently pick the (station, slot) within their
    flexibility window that minimizes their own cost given the *current*
    load-responsive prices; prices are then updated from the resulting
    aggregate demand. Repeats until assignments stop changing (an
    approximate pure-strategy Nash equilibrium) or a max-iteration cap.
    """

    def __init__(self, station_states: Dict[str, StationSlotState], max_iterations: int = 25):
        self.station_states = station_states
        self.max_iterations = max_iterations

    def _slot_price(self, station_id: str, slot: int, demand_kwh: Dict[Tuple[str, int], float]) -> float:
        state = self.station_states[station_id]
        excess = max(0.0, demand_kwh.get((station_id, slot), 0.0) - state.capacity_kw)
        return state.base_price_per_kwh * (1.0 + state.congestion_sensitivity * excess)

    def solve(self, requests: List[ChargeRequest]) -> EquilibriumResult:
        """
        Sequential (asynchronous) best-response: agents move one at a time,
        immediately updating the shared demand/price state before the next
        agent responds. This is the standard convergence-safe scheme for
        congestion games (a potential game), avoiding the herding
        oscillation that synchronous simultaneous best-response can cause
        when many agents chase the same momentarily-cheap slot at once.
        """
        current: List[Tuple[str, int]] = [(r.station_id, r.preferred_slot) for r in requests]
        demand_kwh: Dict[Tuple[str, int], float] = {}
        for req, key in zip(requests, current):
            demand_kwh[key] = demand_kwh.get(key, 0.0) + req.energy_needed_kwh

        converged = False
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            changed = False

            for idx, req in enumerate(requests):
                old_key = current[idx]
                demand_kwh[old_key] = demand_kwh.get(old_key, 0.0) - req.energy_needed_kwh

                best_choice = old_key
                best_cost = self._slot_price(*old_key, demand_kwh) * req.energy_needed_kwh

                for delta in range(-req.flexibility, req.flexibility + 1):
                    candidate = (req.station_id, req.preferred_slot + delta)
                    price = self._slot_price(candidate[0], candidate[1], demand_kwh)
                    shift_penalty = 0.02 * abs(delta) * req.energy_needed_kwh
                    cost = price * req.energy_needed_kwh + shift_penalty
                    if cost < best_cost - 1e-9:
                        best_cost = cost
                        best_choice = candidate

                demand_kwh[best_choice] = demand_kwh.get(best_choice, 0.0) + req.energy_needed_kwh
                if best_choice != old_key:
                    changed = True
                current[idx] = best_choice

            if not changed:
                converged = True
                break

        settled = []
        slot_prices: Dict[Tuple[str, int], float] = {}
        for req, (st, slot) in zip(requests, current):
            price = self._slot_price(st, slot, demand_kwh)
            slot_prices[(st, slot)] = price
            settled.append(
                SettledRequest(
                    fleet_id=req.fleet_id,
                    vehicle_id=req.vehicle_id,
                    station_id=st,
                    assigned_slot=slot,
                    price_per_kwh=round(price, 4),
                    cost=round(price * req.energy_needed_kwh, 4),
                )
            )

        return EquilibriumResult(settled=settled, iterations=iterations, converged=converged, slot_prices=slot_prices)
