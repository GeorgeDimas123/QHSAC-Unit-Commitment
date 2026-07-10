from dataclasses import dataclass
import math

@dataclass
class RewardWeights:
    w_c: float = 1.0
    w_shed: float = 15.0
    w_curt: float = 2.0
    w_res: float = 6.0
    w_flow: float = 6.0
    w_ramp: float = 1.0
    w_mut: float = 2.0
    w_sw: float = 0.5
    w_time: float = 0.2
    eta: float = 0.5     # potential shaping
    rho: float = 1.0     # potential scale
    hard_penalty: float = 20.0  # episode-killing penalty

class RewardShaper:
    def __init__(self, J_star_total, T, median_solve_sec=0.25, gamma=0.99, w: RewardWeights=RewardWeights()):
        self.w = w
        self.gamma = gamma
        self.J_bar = max(1e-6, J_star_total / max(1, T))  # per-period cost scale
        self.median_solve_sec = max(1e-3, median_solve_sec)
        self.prev_state = None
        self.prev_commit = None
        self.prev_p = None
        self.prev_phi = 0.0

    def _ed_merit_order_cost(self, net_load_MW, commit_mask, var_costs, pmax):
        """
        Very fast merit-order ED for one step (no network, no binaries).
        var_costs: [G] $/MWh, commit_mask: [G] {0/1}, pmax: [G] MW
        Returns approximate $ cost for serving net_load_MW.
        """
        stack = sorted([(var_costs[g], pmax[g] if commit_mask[g] else 0.0, g) for g in range(len(var_costs))])
        remaining = max(0.0, net_load_MW)
        cost = 0.0
        for c, cap, _ in stack:
            take = min(cap, remaining)
            cost += c * take
            remaining -= take
            if remaining <= 1e-9:
                break
        # if remaining > 0, we couldn't serve all load: treat as very expensive
        if remaining > 1e-6:
            cost += 10_000.0 * remaining   # proxy VOLL
        return cost

    def potential(self, state, commit_mask, forecasts, gen_var_costs, pmax):
        """
        Phi(s) = -rho * (ED cost for next-period net load) / J_bar
        state/forecasts must provide next-step net load; fall back to current if needed.
        """
        net_load = float(forecasts.get("next_net_load_MW", forecasts.get("net_load_MW", 0.0)))
        ed_cost = self._ed_merit_order_cost(net_load, commit_mask, gen_var_costs, pmax)
        return -self.w.rho * (ed_cost / self.J_bar)

    def step_reward(self, t, uc, state, forecasts, gen_var_costs, pmax, gamma=None):
        """
        uc dict needs:
          costs: {gen, no_load, startup, shutdown} in $
          shed_MW_sum, curt_MW_sum
          reserve_req_up, reserve_req_dn, reserve_sup_up, reserve_sup_dn
          flows: list of MW line flows; Fmax: list
          p_t: list of unit outputs (MW)
          u_t: list of commitments {0/1}
          feasible: bool
          solve_sec: float
          Pd_sum_MW: float total demand this period
        """
        g = self.w
        gamma = self.gamma if gamma is None else gamma

        # Early termination on infeasibility
        if not uc.get("feasible", True):
            return -g.hard_penalty, True  # reward, done

        # Normalizers
        Pd_sum = max(1e-6, uc.get("Pd_sum_MW", 0.0))
        J_bar = self.J_bar

        # Dense cost
        C = uc["costs"]
        c_t = (C["gen"] + C["no_load"] + C["startup"] + C["shutdown"]) / J_bar

        # Penalties
        p_shed = uc.get("shed_MW_sum", 0.0) / Pd_sum
        curt_denom = max(1e-6, uc.get("Pw_avail_MW_sum", 0.0))
        p_curt = uc.get("curt_MW_sum", 0.0) / curt_denom

        req_up = uc.get("reserve_req_up", 0.0); sup_up = uc.get("reserve_sup_up", 0.0)
        req_dn = uc.get("reserve_req_dn", 0.0); sup_dn = uc.get("reserve_sup_dn", 0.0)
        p_res = (max(0.0, req_up - sup_up) + max(0.0, req_dn - sup_dn)) / max(1e-6, (req_up + req_dn))

        flows = uc.get("flows", [])
        Fmax  = uc.get("Fmax",  [])
        p_flow = 0.0
        for f, cap in zip(flows, Fmax):
            if cap > 0:
                p_flow += max(0.0, abs(f) - cap) / cap
        p_flow = p_flow**2  # quadratic barrier

        # Ramping and min up/down
        p_t = uc.get("p_t", [])
        u_t = uc.get("u_t", [])
        p_ramp = 0.0; p_mut = 0.0; p_sw = 0.0
        if self.prev_p is not None and self.prev_commit is not None and len(self.prev_p)==len(p_t):
            RU = uc.get("RU", [1.0]*len(p_t))
            RD = uc.get("RD", [1.0]*len(p_t))
            for i,(p_now,p_prev,ru,rd) in enumerate(zip(p_t, self.prev_p, RU, RD)):
                if ru>0: p_ramp += max(0.0, (p_now - p_prev) - ru)/ru
                if rd>0: p_ramp += max(0.0, (p_prev - p_now) - rd)/rd
            p_ramp /= max(1, len(p_t))
            # switch & min time (environment can flag explicit violations)
            p_sw = sum(1 for a,b in zip(u_t, self.prev_commit) if a!=b)/max(1, len(u_t))
        p_mut = uc.get("min_time_viol_frac", 0.0)

        # Solve time
        solve_sec = max(0.0, uc.get("solve_sec", 0.0))
        p_time = math.log(1.0 + solve_sec / self.median_solve_sec)

        # Potential-based shaping (policy invariant)
        phi_next = self.potential(state, u_t, forecasts, gen_var_costs, pmax)
        F_shape = gamma*phi_next - self.prev_phi

        # Aggregate
        dense = (
            g.w_c * c_t +
            g.w_shed * p_shed + g.w_curt * p_curt + g.w_res * p_res +
            g.w_flow * p_flow + g.w_ramp * p_ramp + g.w_mut * p_mut +
            g.w_sw * p_sw + g.w_time * p_time
        )
        r_t = -dense + g.eta * F_shape

        # Bookkeeping
        self.prev_phi = phi_next
        self.prev_commit = list(u_t)
        self.prev_p = list(p_t)

        # Optional early stop if any critical penalty is large
        done = (p_shed > 0.01) or (p_flow > 0.5)  # tune for your case
        return r_t, done