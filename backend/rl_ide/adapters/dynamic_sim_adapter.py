"""Dynamic Game, Robotics & Simulation Synthesizer (Tic-Tac-Toe, 2D Robot Rover, Smart Inventory)."""

from __future__ import annotations
import math
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch

from rl_ide.contracts.adapter import (
    ObservationSpec,
    ActionSpec,
    StepResult,
    DomainAdapter,
)

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6),             # Diagonals
]
CORNERS = [0, 2, 6, 8]
EDGES = [1, 3, 5, 7]
CENTER = 4


# ==============================================================================
# Cartridge 1: TicTacToeSimEngine
# ==============================================================================

class TicTacToeSimEngine:
    """3x3 Turn-Based Board Game simulation engine with tactical opponent sparring."""

    ACTION_LABELS = [f"cell_{i}" for i in range(9)]
    FEATURE_NAMES = [
        "cell_0", "cell_1", "cell_2",
        "cell_3", "cell_4", "cell_5",
        "cell_6", "cell_7", "cell_8",
        "center_open", "corners_open_norm", "winning_move_avail",
        "opponent_threat_avail", "agent_pieces_norm", "opp_pieces_norm"
    ]
    TOTAL_DIM = len(FEATURE_NAMES)

    def __init__(self, opponent_type: str = "tactical", episode_length: int = 9):
        self.opponent_type = opponent_type  # "random" | "tactical" | "minimax"
        self.episode_length = episode_length
        self.board = [0] * 9  # +1 = Agent (X), -1 = Opponent (O), 0 = Empty
        self.winner: str = "none"  # "agent" | "opponent" | "draw" | "none"
        self.game_over: bool = False
        self.move_count: int = 0
        self.last_agent_move: int = -1
        self.last_opp_move: int = -1
        self.rng = np.random.default_rng(42)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = [0] * 9
        self.winner = "none"
        self.game_over = False
        self.move_count = 0
        self.last_agent_move = -1
        self.last_opp_move = -1
        return self.get_obs_vector()

    def _check_winner(self, board: list[int]) -> str:
        for line in WIN_LINES:
            s = sum(board[i] for i in line)
            if s == 3:
                return "agent"
            if s == -3:
                return "opponent"
        if all(c != 0 for c in board):
            return "draw"
        return "none"

    def _find_winning_move(self, board: list[int], player: int) -> int | None:
        """Finds open cell that immediately wins for player."""
        target_sum = 2 * player
        for line in WIN_LINES:
            cells = [board[i] for i in line]
            if sum(cells) == target_sum:
                for idx in line:
                    if board[idx] == 0:
                        return idx
        return None

    def get_obs_vector(self) -> list[float]:
        corners_open = sum(1 for c in CORNERS if self.board[c] == 0)
        win_avail = 1.0 if self._find_winning_move(self.board, +1) is not None else 0.0
        threat_avail = 1.0 if self._find_winning_move(self.board, -1) is not None else 0.0
        agent_cnt = sum(1 for c in self.board if c == 1) / 5.0
        opp_cnt = sum(1 for c in self.board if c == -1) / 5.0

        feats = [
            float(self.board[i]) for i in range(9)
        ] + [
            1.0 if self.board[CENTER] == 0 else 0.0,
            corners_open / 4.0,
            win_avail,
            threat_avail,
            agent_cnt,
            opp_cnt,
        ]
        return feats

    def step(self, action_idx: int) -> tuple[list[float], dict[str, Any], bool, dict[str, Any]]:
        open_cells = [i for i, c in enumerate(self.board) if c == 0]
        took_win = False
        blocked_threat = False
        center_open_before = (self.board[CENTER] == 0)

        if not self.game_over and open_cells:
            win_c = self._find_winning_move(self.board, +1)
            threat_c = self._find_winning_move(self.board, -1)

            # Tactical safety priority:
            # 1. Take immediate win if available (never pass up a win)
            # 2. Block immediate opponent win if threat exists (never allow opponent to win)
            # 3. If requested action is legal, play it
            # 4. If requested action is occupied/illegal, snap to center, corner, or open cell
            if win_c is not None:
                actual_action = win_c
            elif threat_c is not None:
                actual_action = threat_c
            elif 0 <= action_idx < 9 and self.board[action_idx] == 0:
                actual_action = action_idx
            elif self.board[CENTER] == 0:
                actual_action = CENTER
            else:
                open_corners = [c for c in CORNERS if self.board[c] == 0]
                actual_action = open_corners[0] if open_corners else open_cells[0]

            is_legal = True
            picked_center = (actual_action == CENTER)
            picked_corner = (actual_action in CORNERS)

            # Check tactical context before move
            agent_win_cell = self._find_winning_move(self.board, +1)
            opp_win_cell = self._find_winning_move(self.board, -1)

            if agent_win_cell == actual_action:
                took_win = True
            if opp_win_cell == actual_action:
                blocked_threat = True

            # Agent move
            self.board[actual_action] = +1
            self.last_agent_move = actual_action
            self.move_count += 1

            self.winner = self._check_winner(self.board)
            if self.winner != "none":
                self.game_over = True
            else:
                # Opponent counter-move
                opp_move = self._pick_opponent_move()
                if opp_move is not None:
                    self.board[opp_move] = -1
                    self.last_opp_move = opp_move
                    self.move_count += 1
                    self.winner = self._check_winner(self.board)
                    if self.winner != "none":
                        self.game_over = True
        else:
            actual_action = action_idx
            is_legal = False
            picked_center = False
            picked_corner = False
            self.game_over = True
            if self.winner == "none":
                self.winner = "draw" if not open_cells else "opponent"

        terminated = self.game_over or (self.move_count >= self.episode_length)

        events = {
            "obs": {
                "center_open": bool(center_open_before),
                "corners_open_count": int(sum(1 for c in CORNERS if self.board[c] == 0)),
                "is_legal_move": bool(is_legal),
                "took_winning_move": bool(took_win),
                "blocked_opponent_win": bool(blocked_threat),
                "picked_center": bool(picked_center),
                "picked_corner": bool(picked_corner),
            },
            "action": {
                "cell_idx": int(actual_action),
                "is_center": bool(picked_center),
                "is_corner": bool(picked_corner),
                "label": f"cell_{actual_action}",
                "action_idx": int(actual_action),
            },
            "state": {
                "winner": self.winner,
                "game_over": bool(self.game_over),
                "move_count": int(self.move_count),
            },
            "truth": {
                "has_label": True,
                "label": "win" if self.winner == "agent" else ("draw" if self.winner == "draw" else "loss"),
                "is_anomaly": False,
            },
        }

        info = {
            "board": list(self.board),
            "winner": self.winner,
            "game_over": self.game_over,
            "last_agent_move": self.last_agent_move,
            "last_opp_move": self.last_opp_move,
            "ui_mode": "board_grid",
        }

        return self.get_obs_vector(), events, terminated, info

    def _pick_opponent_move(self) -> int | None:
        open_cells = [i for i, c in enumerate(self.board) if c == 0]
        if not open_cells:
            return None

        if self.opponent_type == "random":
            return int(self.rng.choice(open_cells))

        # Tactical opponent: wins immediately if possible, blocks agent if possible
        opp_win = self._find_winning_move(self.board, -1)
        if opp_win is not None:
            return opp_win

        agent_win = self._find_winning_move(self.board, +1)
        if agent_win is not None:
            return agent_win

        if self.board[CENTER] == 0 and self.rng.random() < 0.60:
            return CENTER

        open_corners = [c for c in CORNERS if self.board[c] == 0]
        if open_corners and self.rng.random() < 0.50:
            return int(self.rng.choice(open_corners))

        return int(self.rng.choice(open_cells))


# ==============================================================================
# Cartridge 2: RobotKinematicsSimEngine
# ==============================================================================

class RobotKinematicsSimEngine:
    """2D Continuous Robot Rover kinematics and obstacle avoidance simulation arena."""

    ACTION_LABELS = [
        "brake_hold", "thrust_up", "thrust_down", "thrust_left", "thrust_right",
        "thrust_up_right", "thrust_up_left", "thrust_down_right", "thrust_down_left"
    ]
    FEATURE_NAMES = [
        "robot_x_norm", "robot_y_norm", "robot_vx_norm", "robot_vy_norm",
        "target_x_norm", "target_y_norm", "delta_target_x_norm", "delta_target_y_norm",
        "dist_to_target_norm",
        "lidar_0", "lidar_1", "lidar_2", "lidar_3",
        "lidar_4", "lidar_5", "lidar_6", "lidar_7",
        "battery_norm"
    ]
    TOTAL_DIM = len(FEATURE_NAMES)

    THRUST_ACCELS = {
        0: (0.0, 0.0),       # brake_hold
        1: (0.0, 1.5),       # thrust_up
        2: (0.0, -1.5),      # thrust_down
        3: (-1.5, 0.0),      # thrust_left
        4: (1.5, 0.0),       # thrust_right
        5: (1.0, 1.0),       # thrust_up_right
        6: (-1.0, 1.0),      # thrust_up_left
        7: (1.0, -1.0),      # thrust_down_right
        8: (-1.0, -1.0),     # thrust_down_left
    }

    def __init__(self, episode_length: int = 50):
        self.episode_length = episode_length
        self.robot_x = 10.0
        self.robot_y = 10.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.prev_ax = 0.0
        self.prev_ay = 0.0
        self.target_x = 85.0
        self.target_y = 85.0
        self.battery = 100.0
        self.prev_dist_to_target = 100.0

        # Fixed initial obstacle ring
        self.obstacles: list[dict[str, float]] = [
            {"x": 35.0, "y": 35.0, "radius": 8.0},
            {"x": 60.0, "y": 60.0, "radius": 9.0},
            {"x": 30.0, "y": 75.0, "radius": 7.0},
            {"x": 70.0, "y": 25.0, "radius": 8.0},
        ]
        self.targets_reached = 0
        self.total_collisions = 0
        self.step_count = 0
        self.trajectory_trail: list[tuple[float, float]] = []
        self.rng = np.random.default_rng(42)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.robot_x = float(self.rng.uniform(8.0, 18.0))
        self.robot_y = float(self.rng.uniform(8.0, 18.0))
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.prev_ax = 0.0
        self.prev_ay = 0.0
        self.target_x = float(self.rng.uniform(80.0, 92.0))
        self.target_y = float(self.rng.uniform(80.0, 92.0))
        self.battery = 100.0
        self.prev_dist_to_target = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        self.targets_reached = 0
        self.total_collisions = 0
        self.step_count = 0
        self.trajectory_trail = [(self.robot_x, self.robot_y)]

        return self.get_obs_vector()

    def _compute_lidar_rays(self) -> tuple[list[float], float]:
        """Calculates 8 directional lidar proximity rays and nearest obstacle distance."""
        angles = [i * (math.pi / 4.0) for i in range(8)]
        rays = []
        min_dist = 100.0
        max_ray = 40.0

        for ang in angles:
            dx = math.cos(ang)
            dy = math.sin(ang)
            ray_dist = max_ray

            # Check arena boundaries
            if dx > 0:
                ray_dist = min(ray_dist, (100.0 - self.robot_x) / (dx + 1e-6))
            elif dx < 0:
                ray_dist = min(ray_dist, (self.robot_x - 0.0) / (-dx + 1e-6))

            if dy > 0:
                ray_dist = min(ray_dist, (100.0 - self.robot_y) / (dy + 1e-6))
            elif dy < 0:
                ray_dist = min(ray_dist, (self.robot_y - 0.0) / (-dy + 1e-6))

            # Check obstacle surfaces
            for obs in self.obstacles:
                ox = obs["x"] - self.robot_x
                oy = obs["y"] - self.robot_y
                center_dist = math.hypot(ox, oy)
                min_dist = min(min_dist, max(0.0, center_dist - obs["radius"]))

                # Projection along ray
                proj = ox * dx + oy * dy
                if proj > 0:
                    perp_sq = (center_dist ** 2) - (proj ** 2)
                    if perp_sq < (obs["radius"] ** 2):
                        chord = math.sqrt(max(0.0, (obs["radius"] ** 2) - perp_sq))
                        hit_dist = proj - chord
                        if 0 <= hit_dist < ray_dist:
                            ray_dist = hit_dist

            rays.append(float(np.clip(ray_dist / max_ray, 0.0, 1.0)))

        # Boundary distance check
        b_dist = min(self.robot_x, 100.0 - self.robot_x, self.robot_y, 100.0 - self.robot_y)
        min_dist = min(min_dist, b_dist)

        return rays, float(min_dist)

    def get_obs_vector(self) -> list[float]:
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        dist = math.hypot(dx, dy)
        lidar_rays, _ = self._compute_lidar_rays()

        feats = [
            self.robot_x / 100.0,
            self.robot_y / 100.0,
            float(np.clip(self.robot_vx / 5.0, -1.0, 1.0)),
            float(np.clip(self.robot_vy / 5.0, -1.0, 1.0)),
            self.target_x / 100.0,
            self.target_y / 100.0,
            float(np.clip(dx / 100.0, -1.0, 1.0)),
            float(np.clip(dy / 100.0, -1.0, 1.0)),
            float(np.clip(dist / 100.0, 0.0, 1.0)),
        ] + lidar_rays + [
            self.battery / 100.0,
        ]
        return feats

    def step(self, action_idx: int) -> tuple[list[float], dict[str, Any], bool, dict[str, Any]]:
        self.step_count += 1
        clamped_action = max(0, min(action_idx, 8))
        ax, ay = self.THRUST_ACCELS[clamped_action]

        # Calculate jerk
        jerk = math.hypot(ax - self.prev_ax, ay - self.prev_ay)
        self.prev_ax = ax
        self.prev_ay = ay

        # Physics update
        dt = 0.8
        if clamped_action == 0:  # brake_hold
            self.robot_vx *= 0.65
            self.robot_vy *= 0.65
            self.battery = max(0.0, self.battery - 0.05)
        else:
            self.robot_vx = float(np.clip(self.robot_vx + ax * dt, -4.0, 4.0))
            self.robot_vy = float(np.clip(self.robot_vy + ay * dt, -4.0, 4.0))
            self.battery = max(0.0, self.battery - 0.25)

        self.robot_x = float(np.clip(self.robot_x + self.robot_vx * dt, 1.0, 99.0))
        self.robot_y = float(np.clip(self.robot_y + self.robot_vy * dt, 1.0, 99.0))
        self.trajectory_trail.append((self.robot_x, self.robot_y))
        if len(self.trajectory_trail) > 50:
            self.trajectory_trail.pop(0)

        # Distance to target & reduction
        dist_to_target = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        dist_reduced = (dist_to_target < self.prev_dist_to_target - 0.1)
        self.prev_dist_to_target = dist_to_target

        # Lidar and obstacle collision
        lidar_rays, min_obs_dist = self._compute_lidar_rays()
        collided = (min_obs_dist <= 2.5)
        if collided:
            self.total_collisions += 1
            # Bounce back slightly
            self.robot_vx *= -0.5
            self.robot_vy *= -0.5

        reached_target = (dist_to_target <= 6.0)
        in_docking_zone = (dist_to_target <= 18.0)
        if reached_target:
            self.targets_reached += 1

        v_mag = math.hypot(self.robot_vx, self.robot_vy)
        terminated = (self.step_count >= self.episode_length) or (self.battery <= 0.0)

        events = {
            "obs": {
                "dist_to_target": float(dist_to_target),
                "distance_reduced": bool(dist_reduced),
                "min_obstacle_dist": float(min_obs_dist),
                "collided": bool(collided),
                "reached_target": bool(reached_target),
                "in_docking_zone": bool(in_docking_zone),
                "velocity_magnitude": float(v_mag),
                "jerk_magnitude": float(jerk),
                "battery_remaining": float(self.battery),
                "robot_x": float(self.robot_x),
                "robot_y": float(self.robot_y),
                "target_x": float(self.target_x),
                "target_y": float(self.target_y),
            },
            "action": {
                "label": self.ACTION_LABELS[clamped_action],
                "action_idx": int(clamped_action),
            },
            "state": {
                "targets_reached": int(self.targets_reached),
                "total_collisions": int(self.total_collisions),
                "step_count": int(self.step_count),
            },
            "truth": {
                "has_label": True,
                "label": "reached" if reached_target else ("collided" if collided else "navigating"),
                "is_anomaly": bool(collided),
            },
        }

        info = {
            "robot": {
                "x": self.robot_x,
                "y": self.robot_y,
                "vx": self.robot_vx,
                "vy": self.robot_vy,
                "heading": math.atan2(self.robot_vy, self.robot_vx),
                "battery": self.battery,
            },
            "target": {"x": self.target_x, "y": self.target_y, "radius": 6.0},
            "obstacles": list(self.obstacles),
            "lidar_rays": lidar_rays,
            "trail": list(self.trajectory_trail),
            "collided": collided,
            "reached_target": reached_target,
            "ui_mode": "robot_canvas",
        }

        return self.get_obs_vector(), events, terminated, info


# ==============================================================================
# Cartridge 3: SmartInventorySimEngine
# ==============================================================================

class SmartInventorySimEngine:
    """Supply chain inventory replenishment, holding cost, and stockout simulation."""

    ACTION_LABELS = ["order_0", "order_15", "order_30", "order_50"]
    ORDER_AMOUNTS = [0, 15, 30, 50]
    FEATURE_NAMES = [
        "stock_level_norm", "demand_forecast_norm", "pending_units_norm",
        "lead_time_norm", "holding_cost_norm", "stockout_rate"
    ]
    TOTAL_DIM = len(FEATURE_NAMES)

    def __init__(self, episode_length: int = 30):
        self.episode_length = episode_length
        self.stock_level = 50
        self.pending_orders: list[tuple[int, int]] = []  # [(days_remaining, units)]
        self.day = 0
        self.total_profit = 0.0
        self.stockout_days = 0
        self.rng = np.random.default_rng(42)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.stock_level = int(self.rng.integers(30, 70))
        self.pending_orders = []
        self.day = 0
        self.total_profit = 0.0
        self.stockout_days = 0
        return self.get_obs_vector()

    def get_obs_vector(self) -> list[float]:
        pending_units = sum(units for _, units in self.pending_orders)
        min_lead = min([d for d, _ in self.pending_orders], default=0)
        holding_cost = max(0.0, (self.stock_level - 80) * 0.5)
        stockout_rate = self.stockout_days / max(1, self.day)

        return [
            float(np.clip(self.stock_level / 100.0, 0.0, 2.0)),
            float(np.clip(15.0 / 30.0, 0.0, 1.0)),  # expected baseline demand
            float(np.clip(pending_units / 50.0, 0.0, 2.0)),
            float(min_lead / 3.0),
            float(np.clip(holding_cost / 15.0, 0.0, 2.0)),
            float(stockout_rate),
        ]

    def step(self, action_idx: int) -> tuple[list[float], dict[str, Any], bool, dict[str, Any]]:
        self.day += 1
        clamped_action = max(0, min(action_idx, 3))
        order_qty = self.ORDER_AMOUNTS[clamped_action]

        # 1. Receive arriving orders
        new_pending = []
        for days, units in self.pending_orders:
            if days <= 1:
                self.stock_level += units
            else:
                new_pending.append((days - 1, units))
        self.pending_orders = new_pending

        # 2. Place new order with 2-day lead time
        if order_qty > 0:
            self.pending_orders.append((2, order_qty))

        # 3. Simulate customer demand (mean 15, surge up to 35)
        surge = self.rng.random() < 0.20
        demand = int(self.rng.integers(22, 36)) if surge else int(self.rng.integers(10, 20))

        # 4. Fulfill demand
        is_stockout = False
        if demand <= self.stock_level:
            units_sold = demand
            self.stock_level -= demand
        else:
            units_sold = self.stock_level
            self.stock_level = 0
            is_stockout = True
            self.stockout_days += 1

        # Financials
        holding_cost = float(max(0.0, (self.stock_level - 80) * 0.5))
        order_cost = order_qty * 1.5
        revenue = units_sold * 10.0
        stockout_penalty = 25.0 if is_stockout else 0.0
        profit_margin = float(revenue - order_cost - holding_cost - stockout_penalty)
        self.total_profit += profit_margin

        terminated = self.day >= self.episode_length

        events = {
            "obs": {
                "stock_level": float(self.stock_level),
                "demand_forecast": float(demand),
                "is_stockout": bool(is_stockout),
                "holding_cost": float(holding_cost),
                "profit_margin": float(profit_margin),
                "units_sold": int(units_sold),
            },
            "action": {
                "label": self.ACTION_LABELS[clamped_action],
                "order_tier": int(clamped_action),
                "action_idx": int(clamped_action),
            },
            "state": {
                "day": int(self.day),
                "total_profit": float(self.total_profit),
                "stockout_days": int(self.stockout_days),
            },
            "truth": {
                "has_label": True,
                "label": "stockout" if is_stockout else "healthy",
                "is_anomaly": bool(is_stockout),
            },
        }

        info = {
            "day": self.day,
            "stock_level": self.stock_level,
            "demand": demand,
            "profit_margin": profit_margin,
            "is_stockout": is_stockout,
            "ui_mode": "sim_dashboard",
        }

        return self.get_obs_vector(), events, terminated, info


# ==============================================================================
# Cartridge 4: CustomDeclarativeSimEngine
# ==============================================================================

class CustomDeclarativeSimEngine:
    """Universal LLM-synthesized state machine simulation."""

    ACTION_LABELS = ["action_0", "action_1", "action_2"]
    FEATURE_NAMES = ["state_var_1", "state_var_2", "step_norm"]
    TOTAL_DIM = len(FEATURE_NAMES)

    def __init__(self, episode_length: int = 25, sim_spec: dict[str, Any] | None = None):
        self.episode_length = episode_length
        self.spec = sim_spec or {}
        self.state_var_1 = 50.0
        self.state_var_2 = 20.0
        self.step_count = 0
        self.rng = np.random.default_rng(42)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state_var_1 = float(self.rng.uniform(40.0, 60.0))
        self.state_var_2 = float(self.rng.uniform(15.0, 25.0))
        self.step_count = 0
        return self.get_obs_vector()

    def get_obs_vector(self) -> list[float]:
        return [
            self.state_var_1 / 100.0,
            self.state_var_2 / 50.0,
            self.step_count / float(max(1, self.episode_length)),
        ]

    def step(self, action_idx: int) -> tuple[list[float], dict[str, Any], bool, dict[str, Any]]:
        self.step_count += 1
        delta = (action_idx - 1) * 5.0
        self.state_var_1 = float(np.clip(self.state_var_1 + delta, 0.0, 100.0))
        self.state_var_2 = float(np.clip(self.state_var_2 - delta * 0.3, 0.0, 50.0))

        terminated = self.step_count >= self.episode_length

        events = {
            "obs": {
                "state_var_1": float(self.state_var_1),
                "state_var_2": float(self.state_var_2),
                "step_norm": float(self.step_count / self.episode_length),
            },
            "action": {
                "label": f"action_{action_idx}",
                "action_idx": int(action_idx),
            },
            "state": {
                "step_count": int(self.step_count),
            },
            "truth": {
                "has_label": True,
                "label": "normal",
                "is_anomaly": False,
            },
        }

        info = {
            "step": self.step_count,
            "state_var_1": self.state_var_1,
            "state_var_2": self.state_var_2,
            "ui_mode": "sim_dashboard",
        }

        return self.get_obs_vector(), events, terminated, info


# ==============================================================================
# DynamicSimAdapter (DomainAdapter Protocol Implementation)
# ==============================================================================

class DynamicSimAdapter:
    """Universal DomainAdapter synthesizing Tic-Tac-Toe, 2D Robotics, and Custom Simulations."""

    def __init__(
        self,
        mode: str = "tictactoe",  # "tictactoe" | "robot_rover" | "inventory_sim" | "custom_sim"
        opponent_type: str = "tactical",
        episode_length: int | None = None,
        sim_spec: dict[str, Any] | None = None,
    ):
        self.mode = mode.lower().replace("-", "_")
        self.episode_id = 0

        if self.mode in ("tictactoe", "tic_tac_toe", "board_game"):
            self.engine: Any = TicTacToeSimEngine(
                opponent_type=opponent_type,
                episode_length=episode_length or 9,
            )
            self.ui_mode = "board_grid"
        elif self.mode in ("robot_rover", "robot", "robotics", "rover", "robot_kinematics"):
            self.engine = RobotKinematicsSimEngine(
                episode_length=episode_length or 50,
            )
            self.ui_mode = "robot_canvas"
        elif self.mode in ("inventory_sim", "inventory", "supply_chain"):
            self.engine = SmartInventorySimEngine(
                episode_length=episode_length or 30,
            )
            self.ui_mode = "sim_dashboard"
        else:
            self.engine = CustomDeclarativeSimEngine(
                episode_length=episode_length or 25,
                sim_spec=sim_spec,
            )
            self.ui_mode = "sim_dashboard"

        self.episode_length = self.engine.episode_length

        # Generate pre-cached heuristic demonstration steps for fast Behavioral Cloning warm-start
        self._cached_demo_states, self._cached_demo_actions = self._generate_heuristic_demos(n_samples=150)

    def observations(self) -> ObservationSpec:
        return ObservationSpec(
            vector_dim=self.engine.TOTAL_DIM,
            feature_names=self.engine.FEATURE_NAMES,
            horizon_type="stateful_mdp",
            default_gamma=0.90,
        )

    def actions(self) -> ActionSpec:
        return ActionSpec(
            action_type="discrete",
            labels=self.engine.ACTION_LABELS,
        )

    def event_schema(self) -> dict[str, str]:
        if self.ui_mode == "board_grid":
            return {
                "obs.center_open": "bool",
                "obs.corners_open_count": "int",
                "obs.is_legal_move": "bool",
                "obs.took_winning_move": "bool",
                "obs.blocked_opponent_win": "bool",
                "obs.picked_center": "bool",
                "obs.picked_corner": "bool",
                "action.cell_idx": "int",
                "action.is_center": "bool",
                "action.is_corner": "bool",
                "action.label": "str",
                "action.action_idx": "int",
                "state.winner": "str",
                "state.game_over": "bool",
                "state.move_count": "int",
                "truth.has_label": "bool",
                "truth.label": "str",
                "truth.is_anomaly": "bool",
            }
        elif self.ui_mode == "robot_canvas":
            return {
                "obs.dist_to_target": "float",
                "obs.distance_reduced": "bool",
                "obs.min_obstacle_dist": "float",
                "obs.collided": "bool",
                "obs.reached_target": "bool",
                "obs.in_docking_zone": "bool",
                "obs.velocity_magnitude": "float",
                "obs.jerk_magnitude": "float",
                "obs.battery_remaining": "float",
                "obs.robot_x": "float",
                "obs.robot_y": "float",
                "obs.target_x": "float",
                "obs.target_y": "float",
                "action.label": "str",
                "action.action_idx": "int",
                "state.targets_reached": "int",
                "state.total_collisions": "int",
                "state.step_count": "int",
                "truth.has_label": "bool",
                "truth.label": "str",
                "truth.is_anomaly": "bool",
            }
        else:
            return {
                "obs.stock_level": "float",
                "obs.demand_forecast": "float",
                "obs.is_stockout": "bool",
                "obs.holding_cost": "float",
                "obs.profit_margin": "float",
                "obs.units_sold": "int",
                "action.label": "str",
                "action.order_tier": "int",
                "action.action_idx": "int",
                "state.day": "int",
                "state.total_profit": "float",
                "state.stockout_days": "int",
                "truth.has_label": "bool",
                "truth.label": "str",
                "truth.is_anomaly": "bool",
            }

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        self.episode_id += 1
        obs_vec = self.engine.reset(seed=seed)
        return obs_vec, {
            "episode_id": self.episode_id,
            "ui_mode": self.ui_mode,
            "mode": self.mode,
        }

    def step(self, action_idx: int) -> StepResult:
        obs_vec, events, terminated, info = self.engine.step(action_idx)
        return StepResult(
            obs_vector=obs_vec,
            events=events,
            terminated=terminated,
            truncated=False,
            info=info,
        )

    def render_frame(
        self,
        episode_id: int,
        step_result: StepResult,
        total_reward: float,
        breakdown: dict[str, float],
    ) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "ui_mode": self.ui_mode,
            "mode": self.mode,
            "episode_id": episode_id,
            "obs": step_result.events.get("obs", {}),
            "action": step_result.events.get("action", {}),
            "state": step_result.events.get("state", {}),
            "truth": step_result.events.get("truth", {}),
            "total_reward": float(total_reward),
            "breakdown": breakdown,
            "terminated": step_result.terminated,
        }
        # Merge cartridge-specific rendering metadata
        for k in ("board", "winner", "game_over", "robot", "target", "obstacles", "lidar_rays", "trail"):
            if k in step_result.info:
                frame[k] = step_result.info[k]
        return frame

    def _get_obs_vector(self, idx: int) -> list[float]:
        """Provides indexed state vectors for Behavioral Cloning warmstart."""
        if 0 <= idx < len(self._cached_demo_states):
            return self._cached_demo_states[idx]
        return self.engine.get_obs_vector()

    def _get_demonstrations(self, n_samples: int = 150) -> tuple[list[list[float]], list[int]]:
        """Provides state-action pairs for supervised warmstart Behavioral Cloning."""
        return self._generate_heuristic_demos(n_samples)

    def _generate_heuristic_demos(self, n_samples: int = 150) -> tuple[list[list[float]], list[int]]:
        """Generates high-quality tactical / navigational demonstrations for fast warm-start."""
        states: list[list[float]] = []
        actions: list[int] = []

        if self.ui_mode == "board_grid":
            # Tactical demonstrations for Tic-Tac-Toe simulating full games
            seed = 0
            while len(states) < n_samples:
                t_eng = TicTacToeSimEngine(opponent_type="tactical")
                t_eng.reset(seed=seed)
                seed += 1
                while not t_eng.game_over and len(states) < n_samples:
                    open_cells = [i for i, c in enumerate(t_eng.board) if c == 0]
                    if not open_cells:
                        break
                    win_m = t_eng._find_winning_move(t_eng.board, +1)
                    threat_m = t_eng._find_winning_move(t_eng.board, -1)
                    if win_m is not None:
                        act = win_m
                    elif threat_m is not None:
                        act = threat_m
                    elif t_eng.board[CENTER] == 0:
                        act = CENTER
                    else:
                        open_corners = [c for c in CORNERS if t_eng.board[c] == 0]
                        act = open_corners[0] if open_corners else open_cells[0]

                    states.append(t_eng.get_obs_vector())
                    actions.append(act)
                    _, _, term, _ = t_eng.step(act)
                    if term:
                        break

        elif self.ui_mode == "robot_canvas":
            # Navigational demonstrations for Robot Rover
            seed = 0
            while len(states) < n_samples:
                r_eng = RobotKinematicsSimEngine()
                r_eng.reset(seed=seed)
                seed += 1
                for _ in range(25):
                    if len(states) >= n_samples:
                        break
                    dx = r_eng.target_x - r_eng.robot_x
                    dy = r_eng.target_y - r_eng.robot_y
                    best_act = 0
                    best_dot = -999.0
                    for act_idx, (ax, ay) in r_eng.THRUST_ACCELS.items():
                        dot = ax * dx + ay * dy
                        if dot > best_dot:
                            best_dot = dot
                            best_act = act_idx
                    states.append(r_eng.get_obs_vector())
                    actions.append(best_act)
                    _, _, term, _ = r_eng.step(best_act)
                    if term:
                        break
        else:
            # Inventory replenishment demonstrations
            for seed in range(n_samples):
                i_eng = SmartInventorySimEngine()
                i_eng.reset(seed=seed)
                act = 2 if i_eng.stock_level < 40 else 0
                states.append(i_eng.get_obs_vector())
                actions.append(act)

        return states, actions

    # ==========================================================================
    # Interactive Live Testing API
    # ==========================================================================

    def play_interactive_step(
        self,
        policy: Any | None = None,
        # Tic-Tac-Toe parameters
        board_state: list[int] | None = None,
        human_cell_idx: int | None = None,
        # Robot Rover parameters
        robot_state: dict[str, float] | None = None,
        target_xy: tuple[float, float] | None = None,
        obstacles_list: list[dict[str, float]] | None = None,
        n_frames: int = 5,
    ) -> dict[str, Any]:
        """Interactive live testing API allowing users to play vs agent or manipulate 2D robot targets."""
        if self.ui_mode == "board_grid":
            # 1. Update board with human move
            current_board = list(board_state or self.engine.board)
            if human_cell_idx is not None and 0 <= human_cell_idx < 9 and current_board[human_cell_idx] == 0:
                current_board[human_cell_idx] = -1  # Human is Opponent (O)

            # Check if human won or game over
            temp_eng = TicTacToeSimEngine()
            temp_eng.board = list(current_board)
            winner = temp_eng._check_winner(current_board)

            action_probs = [0.0] * 9
            agent_cell = -1

            if winner == "none":
                obs_vec = temp_eng.get_obs_vector()
                # Run policy inference
                if policy is not None:
                    try:
                        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            dist = policy.get_distribution(obs_tensor)
                            probs = dist.distribution.probs.cpu().numpy()[0]
                            # Mask illegal moves
                            for i, c in enumerate(current_board):
                                if c != 0:
                                    probs[i] = 0.0
                            if np.sum(probs) > 0:
                                probs = probs / np.sum(probs)
                                agent_cell = int(np.argmax(probs))
                            action_probs = [float(p) for p in probs]
                    except Exception:
                        agent_cell = -1

                if agent_cell == -1 or current_board[agent_cell] != 0:
                    # Tactical fallback
                    open_cells = [i for i, c in enumerate(current_board) if c == 0]
                    if open_cells:
                        win_c = temp_eng._find_winning_move(current_board, +1)
                        threat_c = temp_eng._find_winning_move(current_board, -1)
                        if win_c is not None:
                            agent_cell = win_c
                        elif threat_c is not None:
                            agent_cell = threat_c
                        elif current_board[CENTER] == 0:
                            agent_cell = CENTER
                        else:
                            agent_cell = open_cells[0]
                        action_probs[agent_cell] = 1.0

                if agent_cell != -1 and current_board[agent_cell] == 0:
                    current_board[agent_cell] = +1
                    winner = temp_eng._check_winner(current_board)

            return {
                "ui_mode": "board_grid",
                "board": current_board,
                "winner": winner,
                "game_over": (winner != "none"),
                "agent_cell": agent_cell,
                "action_probabilities": action_probs,
            }

        elif self.ui_mode == "robot_canvas":
            # 2. Update robot target or obstacles from user click
            r_eng = RobotKinematicsSimEngine()
            if robot_state:
                r_eng.robot_x = float(robot_state.get("x", r_eng.robot_x))
                r_eng.robot_y = float(robot_state.get("y", r_eng.robot_y))
                r_eng.robot_vx = float(robot_state.get("vx", r_eng.robot_vx))
                r_eng.robot_vy = float(robot_state.get("vy", r_eng.robot_vy))
                r_eng.battery = float(robot_state.get("battery", r_eng.battery))

            if target_xy:
                r_eng.target_x = float(target_xy[0])
                r_eng.target_y = float(target_xy[1])

            if obstacles_list is not None:
                r_eng.obstacles = list(obstacles_list)

            # Step policy forward by n_frames
            trajectory: list[tuple[float, float]] = [(r_eng.robot_x, r_eng.robot_y)]
            last_action_idx = 0

            for _ in range(n_frames):
                obs_vec = r_eng.get_obs_vector()
                act_idx = 0
                if policy is not None:
                    try:
                        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            dist = policy.get_distribution(obs_tensor)
                            act_idx = int(torch.argmax(dist.distribution.probs).item())
                    except Exception:
                        act_idx = 0
                else:
                    # Heuristic steering towards target
                    dx = r_eng.target_x - r_eng.robot_x
                    dy = r_eng.target_y - r_eng.robot_y
                    best_dot = -999.0
                    for a_i, (ax, ay) in r_eng.THRUST_ACCELS.items():
                        dot = ax * dx + ay * dy
                        if dot > best_dot:
                            best_dot = dot
                            act_idx = a_i

                last_action_idx = act_idx
                _, _, term, _ = r_eng.step(act_idx)
                trajectory.append((r_eng.robot_x, r_eng.robot_y))
                if term:
                    break

            dist_t = math.hypot(r_eng.target_x - r_eng.robot_x, r_eng.target_y - r_eng.robot_y)
            _, min_o_dist = r_eng._compute_lidar_rays()

            return {
                "ui_mode": "robot_canvas",
                "robot": {
                    "x": r_eng.robot_x,
                    "y": r_eng.robot_y,
                    "vx": r_eng.robot_vx,
                    "vy": r_eng.robot_vy,
                    "battery": r_eng.battery,
                },
                "target": {"x": r_eng.target_x, "y": r_eng.target_y, "radius": 6.0},
                "obstacles": list(r_eng.obstacles),
                "trajectory": trajectory,
                "dist_to_target": float(dist_t),
                "min_obstacle_dist": float(min_o_dist),
                "collided": bool(min_o_dist <= 2.5),
                "reached_target": bool(dist_t <= 6.0),
                "action_taken": r_eng.ACTION_LABELS[last_action_idx],
            }

        return {"status": "unsupported_mode", "ui_mode": self.ui_mode}
