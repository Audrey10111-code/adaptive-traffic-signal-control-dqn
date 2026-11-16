"""DQN pipeline for a simplified four-way traffic signal control problem.

This is an educational, queue-based simulator (not a SUMO or real-road model).
It implements:
- A four-approach intersection with two non-conflicting phases (NS and EW)
- Four actions consistent with the project proposal
- Minimum-green and maximum-green action masks
- A DQN with experience replay and a target network
- Fixed-time and longest-queue-first baseline controllers
- Evaluation metrics and CSV/PNG/model outputs

Run in Google Colab or locally:
    pip install -r requirements.txt
    python dqn_traffic.py --episodes 80 --steps 240 --eval-episodes 10

For a quick test:
    python dqn_traffic.py --episodes 5 --steps 60 --eval-episodes 2
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


# ------------------------------- Configuration -------------------------------
SEED = 42
N_ACTIONS = 4
STATE_SIZE = 10  # 4 queues + 2 phase flags + 1 phase-age + 3 time-period flags
STEP_SECONDS = 5
MIN_GREEN_SECONDS = 10
MAX_GREEN_SECONDS = 60
MAX_QUEUE_FEATURE = 40.0
SERVICE_PER_APPROACH_PER_STEP = 2.0

# Action meanings (same as the project proposal):
# 0 = keep the current phase for one normal 5-second interval
# 1 = switch to north-south green (only if safe and currently on EW)
# 2 = switch to east-west green (only if safe and currently on NS)
# 3 = extend the current green by one additional 5-second interval
# In this simulator, action 3 holds the current phase for 10 seconds total.


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class TrafficIntersection:
    """Simplified stochastic queue simulator for one four-way intersection.

    Phase 0 serves North and South. Phase 1 serves East and West.
    One episode contains three representative demand periods (off-peak,
    morning peak, evening peak). It is a short demonstration cycle, not a
    calibrated 24-hour traffic model.
    """

    def __init__(self, steps_per_episode: int = 240, seed: int | None = None):
        self.steps_per_episode = steps_per_episode
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.queues = np.zeros(4, dtype=np.float32)  # N, S, E, W
        self.phase = 0  # start with NS green
        self.phase_elapsed = 0
        self.decision_steps = 0
        self.simulated_seconds = 0
        self.total_wait_vehicle_seconds = 0.0
        self.total_served = 0
        self.switches = 0
        self.max_queue_observed = 0.0
        self.served_by_approach = np.zeros(4, dtype=np.float32)
        self.arrived_by_approach = np.zeros(4, dtype=np.float32)
        return self._get_state()

    def _period_index(self) -> int:
        # Divide the representative episode into off-peak, morning peak,
        # and evening peak traffic demand periods.
        fraction = self.decision_steps / max(1, self.steps_per_episode)
        if fraction < 1.0 / 3.0:
            return 0
        if fraction < 2.0 / 3.0:
            return 1
        return 2

    def _arrival_rates(self) -> np.ndarray:
        # Expected arrivals per 5-second internal interval. These are
        # illustrative synthetic rates and should be calibrated if real data
        # or SUMO counts become available.
        rates = np.array([
            [0.45, 0.45, 0.45, 0.45],  # off-peak
            [1.10, 1.25, 0.65, 0.70],  # morning: heavier N/S demand
            [0.65, 0.70, 1.15, 1.20],  # evening: heavier E/W demand
        ], dtype=np.float32)
        return rates[self._period_index()]

    def _get_state(self) -> np.ndarray:
        queue_features = np.clip(self.queues, 0, MAX_QUEUE_FEATURE) / MAX_QUEUE_FEATURE
        phase_features = np.array([1.0, 0.0] if self.phase == 0 else [0.0, 1.0], dtype=np.float32)
        phase_age = np.array([min(self.phase_elapsed, MAX_GREEN_SECONDS) / MAX_GREEN_SECONDS], dtype=np.float32)
        time_features = np.zeros(3, dtype=np.float32)
        time_features[self._period_index()] = 1.0
        state = np.concatenate([queue_features, phase_features, phase_age, time_features]).astype(np.float32)
        assert state.shape == (STATE_SIZE,)
        return state

    def valid_action_mask(self) -> np.ndarray:
        """Return legal actions; the agent cannot violate phase safety rules."""
        # Do not permit a phase to exceed the configured maximum green time.
        can_keep = self.phase_elapsed + STEP_SECONDS <= MAX_GREEN_SECONDS
        can_extend = self.phase_elapsed + (2 * STEP_SECONDS) <= MAX_GREEN_SECONDS
        mask = np.array([can_keep, False, False, can_extend], dtype=bool)
        can_switch = self.phase_elapsed >= MIN_GREEN_SECONDS
        if can_switch and self.phase == 1:
            mask[1] = True  # switch to NS
        if can_switch and self.phase == 0:
            mask[2] = True  # switch to EW
        # At the maximum green time, a switch is required.
        # Ensure at least one action remains legal.
        return mask

    def _one_internal_interval(self, green_phase: int | None) -> Tuple[float, float, float]:
        """Advance 5 seconds; None means yellow/all-red clearance (no service).

        Returns waiting cost (vehicle-seconds), vehicles served, longest queue.
        """
        arrivals = self.rng.poisson(self._arrival_rates()).astype(np.float32)
        self.queues += arrivals
        self.arrived_by_approach += arrivals

        # Approximate waiting accumulated over this 5-second interval.
        waiting_cost = float(np.sum(self.queues) * STEP_SECONDS)
        served = 0.0
        if green_phase is not None:
            active = (0, 1) if green_phase == 0 else (2, 3)
            for idx in active:
                departure = min(float(self.queues[idx]), SERVICE_PER_APPROACH_PER_STEP)
                self.queues[idx] -= departure
                self.served_by_approach[idx] += departure
                served += departure

        longest_queue = float(np.max(self.queues))
        self.max_queue_observed = max(self.max_queue_observed, longest_queue)
        self.total_wait_vehicle_seconds += waiting_cost
        self.total_served += served
        self.simulated_seconds += STEP_SECONDS
        return waiting_cost, served, longest_queue

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        mask = self.valid_action_mask()
        if action < 0 or action >= N_ACTIONS or not mask[action]:
            raise ValueError(f"Invalid/unsafe action {action}; valid mask is {mask.tolist()}")

        waiting_cost = 0.0
        switched = 0

        if action in (1, 2):
            new_phase = 0 if action == 1 else 1
            if new_phase == self.phase:
                raise ValueError("Switch action must select the opposite phase.")
            switched = 1
            self.switches += 1
            # One 5-second yellow/all-red clearance interval. No approach is served.
            w, _, _ = self._one_internal_interval(green_phase=None)
            waiting_cost += w
            self.phase = new_phase
            self.phase_elapsed = 0
            # Then one 5-second interval of the newly selected green phase.
            w, _, _ = self._one_internal_interval(green_phase=self.phase)
            waiting_cost += w
            self.phase_elapsed += STEP_SECONDS
        elif action == 3:
            # Hold the current phase for two 5-second intervals (10 seconds total).
            for _ in range(2):
                w, _, _ = self._one_internal_interval(green_phase=self.phase)
                waiting_cost += w
                self.phase_elapsed += STEP_SECONDS
        else:  # action 0: keep the current phase for one 5-second interval
            w, _, _ = self._one_internal_interval(green_phase=self.phase)
            waiting_cost += w
            self.phase_elapsed += STEP_SECONDS

        self.decision_steps += 1
        done = self.decision_steps >= self.steps_per_episode

        # Reward follows the proposal: penalise accumulated waiting, the
        # longest queue, and unnecessary phase switching. Waiting is divided
        # by 10 only to keep numerical magnitudes manageable during training.
        rho1 = 0.5
        rho2 = 2.0
        reward = -(waiting_cost / 10.0) - rho1 * float(np.max(self.queues)) - rho2 * switched

        info = {
            "waiting_vehicle_seconds": waiting_cost,
            "longest_queue": float(np.max(self.queues)),
            "vehicles_served": float(self.total_served),
            "switches": float(self.switches),
            "simulated_seconds": float(self.simulated_seconds),
        }
        return self._get_state(), float(reward), done, info


# --------------------------------- DQN model ----------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size: int = STATE_SIZE, n_actions: int = N_ACTIONS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, learning_rate: float = 1e-3, gamma: float = 0.95,
                 batch_size: int = 64, buffer_size: int = 20_000,
                 target_update: int = 200, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    @staticmethod
    def _masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
        safe_q = np.where(mask, q_values, -1e9)
        return int(np.argmax(safe_q))

    def choose_action(self, state: np.ndarray, valid_mask: np.ndarray, epsilon: float) -> int:
        valid_actions = np.flatnonzero(valid_mask)
        if len(valid_actions) == 0:
            raise RuntimeError("No legal actions available; check the safety mask.")
        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(state_t).squeeze(0).cpu().numpy()
        return self._masked_argmax(q_values, valid_mask)

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)

        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool, device=self.device)

        current_q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_all = self.target(next_states)
            next_q_all = next_q_all.masked_fill(~next_masks, -1e9)
            max_next_q = next_q_all.max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


# ---------------------------- Baseline controllers ----------------------------
def fixed_time_policy(env: TrafficIntersection, state: np.ndarray) -> int:
    """Alternate NS/EW every 30 seconds, respecting minimum-green masking."""
    mask = env.valid_action_mask()
    target_phase = 1 - env.phase if env.phase_elapsed >= 30 else env.phase
    if target_phase == env.phase:
        return 0
    target_action = 2 if target_phase == 1 else 1
    return target_action if mask[target_action] else 0


def longest_queue_policy(env: TrafficIntersection, state: np.ndarray) -> int:
    """Serve the busier pair of approaches, subject to safety and max green."""
    mask = env.valid_action_mask()
    ns_total = float(env.queues[0] + env.queues[1])
    ew_total = float(env.queues[2] + env.queues[3])
    target_phase = 0 if ns_total >= ew_total else 1
    if target_phase != env.phase:
        target_action = 1 if target_phase == 0 else 2
        if mask[target_action]:
            return target_action
        return 0
    # If this phase is still busier, extend only where the extension is legal.
    if mask[3]:
        return 3
    # At maximum green, switch to the other phase to prevent starvation.
    if mask[0]:
        return 0
    return 2 if env.phase == 0 else 1


def evaluate_policy(policy_name: str, policy_fn, episodes: int, steps: int,
                    seeds: List[int], agent: DQNAgent | None = None) -> List[Dict[str, float | str]]:
    results: List[Dict[str, float | str]] = []
    for ep in range(episodes):
        env = TrafficIntersection(steps_per_episode=steps, seed=seeds[ep])
        state = env.reset(seed=seeds[ep])
        episode_reward = 0.0
        invalid_actions = 0
        done = False
        while not done:
            if agent is not None:
                action = agent.choose_action(state, env.valid_action_mask(), epsilon=0.0)
            else:
                action = policy_fn(env, state)
            # Safety check: an invalid action is counted and replaced by keep.
            if not env.valid_action_mask()[action]:
                invalid_actions += 1
                legal_actions = np.flatnonzero(env.valid_action_mask())
                action = int(legal_actions[0])
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        results.append({
            "policy": policy_name,
            "episode": ep + 1,
            "return": episode_reward,
            "total_wait_vehicle_seconds": env.total_wait_vehicle_seconds,
            "final_mean_queue_per_approach": float(np.mean(env.queues)),
            "minimum_approach_service_ratio": float(np.min(
                env.served_by_approach / np.maximum(env.arrived_by_approach, 1.0)
            )),
            "maximum_queue_observed": env.max_queue_observed,
            "vehicles_served": env.total_served,
            "phase_switches": env.switches,
            "invalid_actions": invalid_actions,
            "safety_violations": 0,
        })
    return results


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate DQN traffic signal controller")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default="traffic_rl_outputs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent()
    training_rows: List[Dict[str, float]] = []
    total_training_steps = args.episodes * args.steps
    global_step = 0
    epsilon_start, epsilon_end = 1.0, 0.05
    epsilon_decay_steps = max(1, int(total_training_steps * 0.8))

    print(f"Training DQN for {args.episodes} episodes; device={agent.device}")
    for episode in range(args.episodes):
        env = TrafficIntersection(steps_per_episode=args.steps, seed=args.seed + episode)
        state = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        losses: List[float] = []
        done = False

        while not done:
            fraction = min(1.0, global_step / epsilon_decay_steps)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            mask = env.valid_action_mask()
            action = agent.choose_action(state, mask, epsilon)
            next_state, reward, done, info = env.step(action)
            next_mask = env.valid_action_mask()
            agent.memory.add(Transition(state.copy(), action, reward,
                                        next_state.copy(), done, next_mask.copy()))
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_reward += reward
            global_step += 1

        training_rows.append({
            "episode": episode + 1,
            "episode_return": episode_reward,
            "epsilon": epsilon,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "total_wait_vehicle_seconds": env.total_wait_vehicle_seconds,
            "vehicles_served": env.total_served,
            "phase_switches": env.switches,
        })
        if (episode + 1) % max(1, args.episodes // 10) == 0:
            print(f"Episode {episode + 1:>4}/{args.episodes} | return={episode_reward:>9.2f} | epsilon={epsilon:.3f}")

    write_csv(output_dir / "training_log.csv", training_rows)
    torch.save(agent.online.state_dict(), output_dir / "dqn_traffic_model.pt")

    # Evaluate all controllers on the same random seeds for a fair comparison.
    evaluation_seeds = [args.seed + 10_000 + i for i in range(args.eval_episodes)]
    evaluation_rows: List[Dict] = []
    evaluation_rows += evaluate_policy("DQN", None, args.eval_episodes, args.steps, evaluation_seeds, agent=agent)
    evaluation_rows += evaluate_policy("Fixed-time", fixed_time_policy, args.eval_episodes, args.steps, evaluation_seeds)
    evaluation_rows += evaluate_policy("Longest-queue-first", longest_queue_policy, args.eval_episodes, args.steps, evaluation_seeds)
    write_csv(output_dir / "evaluation_results.csv", evaluation_rows)

    summary_rows: List[Dict[str, float | str]] = []
    metrics = ["return", "total_wait_vehicle_seconds", "vehicles_served",
               "maximum_queue_observed", "phase_switches",
               "minimum_approach_service_ratio", "invalid_actions", "safety_violations"]
    for policy_name in ("DQN", "Fixed-time", "Longest-queue-first"):
        subset = [r for r in evaluation_rows if r["policy"] == policy_name]
        row: Dict[str, float | str] = {"policy": policy_name, "episodes": len(subset)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in subset], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    write_csv(output_dir / "evaluation_summary.csv", summary_rows)

    print("\nEvaluation summary (mean across test episodes):")
    for policy in ("DQN", "Fixed-time", "Longest-queue-first"):
        subset = [r for r in evaluation_rows if r["policy"] == policy]
        wait = np.mean([float(r["total_wait_vehicle_seconds"]) for r in subset])
        served = np.mean([float(r["vehicles_served"]) for r in subset])
        maxq = np.mean([float(r["maximum_queue_observed"]) for r in subset])
        switches = np.mean([float(r["phase_switches"]) for r in subset])
        fairness = np.mean([float(r["minimum_approach_service_ratio"]) for r in subset])
        print(f"{policy:>20}: mean wait={wait:.1f} vehicle-s, served={served:.1f}, max queue={maxq:.1f}, switches={switches:.1f}, min service ratio={fairness:.2f}")

    # Plot the training curve, if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt
        returns = [r["episode_return"] for r in training_rows]
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(returns) + 1), returns, alpha=0.35, label="Episode return")
        ma = moving_average(returns, window=min(10, max(2, len(returns))))
        start = len(returns) - len(ma) + 1
        plt.plot(range(start, len(returns) + 1), ma, label="Moving average")
        plt.xlabel("Training episode")
        plt.ylabel("Cumulative reward")
        plt.title("DQN Training Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.png", dpi=160)
        plt.close()
    except ImportError:
        print("matplotlib not installed; skipped training_curve.png")

    print(f"\nFiles saved in: {output_dir.resolve()}")
    print("Important: use the generated CSV/plot values in your report; do not invent or copy example results.")


if __name__ == "__main__":
    main()
