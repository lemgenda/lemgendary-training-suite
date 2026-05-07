import os
import torch
import math
import numpy as np

class SmartTrainingGovernor:
    """
    2026 Universal Autonomous Optimization Engine.
    
    v10.0 Manifold Anchor:
    - State-Loop Detection (Prevents circular 'Escalate-Recoil' cycles)
    - Failure Path Memory (Blacklists breaking points)
    - Surgical State-Loop Penalties (LR/Temp anchoring for failed states)
    - Numerical Shakeups (Forcing manifold diversification)
    """
    def __init__(self, model_info, stabilizers=None):
        self.model_info = model_info
        opt = model_info.get("optimization", {})
        self.enabled = opt.get("enabled", True)
        
        # --- Curriculum Ladder ---
        self.res_ladder = opt.get("res_ladder", [224, 384, 512, 640])
        self.target_effective_batch = opt.get("target_effective_batch", 24)
        
        # --- Persistent State ---
        self.current_fraction = opt.get("initial_fraction", 0.1)
        self.current_batch = 16 if model_info.get("batch_size") == "auto" else int(model_info.get("batch_size"))
        self.current_acc = 1
        
        raw_size = model_info.get("input_size", 224)
        self.current_res = raw_size[1] if isinstance(raw_size, list) else raw_size
        if self.current_res not in self.res_ladder:
            self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
            
        self.stab = stabilizers or {}
        self.task_type = model_info.get("dataset_type", "quality")
        if isinstance(self.task_type, list): self.task_type = self.task_type[0]
        self.min_temp = 0.5 if self.task_type == "quality" else 0.1
        self.current_temp = self.stab.get("softmax_temp", self.min_temp)
        self.current_clamp = self.stab.get("logit_clamp", 15.0)
        
        # --- Surgical Memory (v10.0) ---
        self.history = [] # Last 5 epochs [quality, loss]
        self.failure_log = {} # {(res, round(frac,2)): failure_count}
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.best_quality = 0.0
        self.stabilization_epochs = 0
        self.lr_multiplier = 1.0
        self.last_action_epoch = 0
        self.epoch_count = 0
        
    def get_phase(self):
        res_idx = self.res_ladder.index(self.current_res)
        if res_idx == 0 and self.current_fraction < 0.5: return "FOUNDATION"
        if self.current_fraction < 0.95: return "EXPANSION"
        if res_idx < len(self.res_ladder) - 1: return "DEEPENING"
        return "REFINEMENT"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None, current_loss=None):
        if not self.enabled: return False, False, False, False, False, False, ""
        self.epoch_count += 1
        
        # 1. Update Memory
        self.history.append((current_quality, current_loss))
        if len(self.history) > 5: self.history.pop(0)
        self.best_quality = max(self.best_quality, current_quality)
        
        # 2. Guard: Stabilization
        if self.stabilization_epochs > 0:
            self.stabilization_epochs -= 1
            self.prev_quality = current_quality
            if current_loss: self.prev_loss = current_loss
            return False, False, False, False, False, False, "📡 Anchoring Manifold..."

        f_changed = r_changed = lr_changed = t_changed = c_changed = b_changed = False
        self.lr_multiplier = 1.0
        msg_parts = []
        phase = self.get_phase()

        # 3. Diagnosis
        delta_q = current_quality - self.prev_quality
        is_turbulent = False
        if len(self.history) >= 3:
            q_values = [h[0] for h in self.history[-3:]]
            deltas = [q_values[i] - q_values[i-1] for i in range(1, len(q_values))]
            if all(deltas[i] * deltas[i-1] < 0 for i in range(1, len(deltas))): is_turbulent = True

        is_flat = abs(delta_q) < 0.0005 and len(self.history) >= 2
        is_regressing = delta_q < -0.01 

        # 4. LOOP DETECTION (v10.0)
        current_state = (self.current_res, round(self.current_fraction, 2))
        failures = self.failure_log.get(current_state, 0)
        
        # If we just hit a regression or turbulence in a known shaky state, it counts as a failure
        if is_regressing or is_turbulent:
            self.failure_log[current_state] = failures + 1
            msg_parts.append(f"⚠️ FAILURE LOGGED: State {current_state} (Count: {self.failure_log[current_state]})")
            
            # --- EMERGENCY RECOIL (Surgical v10.0) ---
            if self.failure_log[current_state] >= 2:
                # We are stuck in a loop. Trigger Numerical Shakeup.
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                t_changed = True
                msg_parts.append("⛓️ LOOP DETECTED: Forcing Numerical Shakeup to diversify manifold")
            
            old_frac = self.current_fraction
            self.current_fraction = max(0.15, self.current_fraction - 0.15)
            f_changed = True
            self.lr_multiplier = 0.7 
            lr_changed = True
            self.stabilization_epochs = 3
            msg_parts.append(f"RECOIL: Strategic retreat to {self.current_fraction*100:.0f}%")

        # --- PROPULSION: FLATLINE/STAGNATION ---
        elif is_flat or (current_quality > 0.94 and delta_q < 0.001):
            # Check if NEXT state is a failure state
            next_frac = min(1.0, self.current_fraction + 0.2)
            next_state = (self.current_res, round(next_frac, 2))
            
            if self.failure_log.get(next_state, 0) > 0:
                # Approach with caution: Lower LR before entering known failure zone
                self.lr_multiplier = 0.5
                lr_changed = True
                msg_parts.append(f"⚓ ANCHOR: Known breaking point ahead. Approaching with 0.5x LR.")
            
            if phase == "FOUNDATION" or phase == "EXPANSION":
                self.current_fraction = next_frac
                f_changed = True
                msg_parts.append(f"PROPULSION: Data -> {self.current_fraction*100:.0f}%")
                self.stabilization_epochs = 1
            elif phase == "DEEPENING":
                current_idx = self.res_ladder.index(self.current_res)
                next_res = self.res_ladder[current_idx + 1]
                res_ratio = next_res / self.current_res
                vram_growth = res_ratio ** 2.2 
                self.current_batch = max(1, int(self.current_batch / vram_growth))
                self.current_acc = max(1, self.target_effective_batch // self.current_batch)
                self.current_res = next_res
                r_changed = b_changed = True
                self.current_fraction = 0.5
                f_changed = True
                msg_parts.append(f"SPATIAL JUMP: {next_res}px | Data Reset 50%")
                self.stabilization_epochs = 3
            else:
                self.lr_multiplier = 0.5
                lr_changed = True
                msg_parts.append("REFINEMENT: SOTA Precision Cooling")

        self.prev_quality = current_quality
        if current_loss: self.prev_loss = current_loss
        
        final_msg = f"🚀 [{phase}] " + " | ".join(msg_parts) if msg_parts else ""
        if not final_msg:
            print(f"📡 [{phase}] Monitoring... [Acc: {current_quality:.4f}] [Failures: {failures}]")
            
        return f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, final_msg

    def get_dynamic_save_interval(self, avg_iter_time, total_iters):
        if avg_iter_time <= 0: return 0.2 
        epoch_duration_mins = (avg_iter_time * total_iters) / 60
        target_pct = 15 / max(1, epoch_duration_mins)
        return max(0.05, min(0.5, target_pct))

    def get_state(self):
        return {
            "sample_fraction": self.current_fraction,
            "input_size": self.current_res,
            "softmax_temp": self.current_temp,
            "logit_clamp": self.current_clamp,
            "lr_multiplier": self.lr_multiplier,
            "batch_size": self.current_batch,
            "accumulation_steps": self.current_acc,
            "stabilization_epochs": self.stabilization_epochs,
            "failure_log": self.failure_log
        }

    def load_state(self, state):
        if not state: return
        self.current_fraction = state.get("sample_fraction", self.current_fraction)
        raw_res = state.get("input_size", self.current_res)
        self.current_res = raw_res[1] if isinstance(raw_res, list) else raw_res
        self.current_temp = state.get("softmax_temp", self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)
        self.failure_log = state.get("failure_log", {})

    def recoil(self):
        old_frac = self.current_fraction
        self.current_fraction = max(0.15, old_frac - 0.15)
        self.current_temp = min(1.5, self.current_temp * 1.3)
        self.stabilization_epochs = 3
        return f"⚡ [GOVERNOR] RECOIL: Strategic Retreat to {self.current_fraction*100:.0f}%"

    def get_dynamic_save_interval(self, avg_iter_time, total_iters):
        if avg_iter_time <= 0: return 0.2 
        epoch_duration_mins = (avg_iter_time * total_iters) / 60
        target_pct = 15 / max(1, epoch_duration_mins)
        return max(0.05, min(0.5, target_pct))

    def get_state(self):
        return {
            "sample_fraction": self.current_fraction,
            "input_size": self.current_res,
            "softmax_temp": self.current_temp,
            "logit_clamp": self.current_clamp,
            "lr_multiplier": self.lr_multiplier,
            "batch_size": self.current_batch,
            "accumulation_steps": self.current_acc,
            "stabilization_epochs": self.stabilization_epochs
        }

    def load_state(self, state):
        if not state: return
        self.current_fraction = state.get("sample_fraction", self.current_fraction)
        raw_res = state.get("input_size", self.current_res)
        self.current_res = raw_res[1] if isinstance(raw_res, list) else raw_res
        self.current_temp = state.get("softmax_temp", self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)

    def recoil(self):
        old_frac = self.current_fraction
        self.current_fraction = max(0.15, old_frac - 0.15)
        self.current_temp = min(1.5, self.current_temp * 1.3)
        self.stabilization_epochs = 3
        return f"⚡ [GOVERNOR] RECOIL: Strategic Retreat to {self.current_fraction*100:.0f}%"

    def get_dynamic_save_interval(self, avg_iter_time, total_iters):
        if avg_iter_time <= 0: return 0.2 
        epoch_duration_mins = (avg_iter_time * total_iters) / 60
        target_pct = 15 / max(1, epoch_duration_mins)
        return max(0.05, min(0.5, target_pct))

    def get_state(self):
        return {
            "sample_fraction": self.current_fraction,
            "input_size": self.current_res,
            "softmax_temp": self.current_temp,
            "logit_clamp": self.current_clamp,
            "lr_multiplier": self.lr_multiplier,
            "batch_size": self.current_batch,
            "accumulation_steps": self.current_acc,
            "stabilization_epochs": self.stabilization_epochs
        }

    def load_state(self, state):
        if not state: return
        self.current_fraction = state.get("sample_fraction", self.current_fraction)
        raw_res = state.get("input_size", self.current_res)
        self.current_res = raw_res[1] if isinstance(raw_res, list) else raw_res
        self.current_temp = max(self.min_temp, state.get("softmax_temp", self.current_temp))
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)

    def recoil(self):
        old_frac = self.current_fraction
        # Recoil is now smarter: it never drops below 15% unless absolutely necessary
        self.current_fraction = max(0.15, old_frac - 0.15)
        self.current_temp = min(1.5, self.current_temp * 1.3)
        self.stabilization_epochs = 3
        return f"⚡ [GOVERNOR] RECOIL: Strategic Retreat to {self.current_fraction*100:.0f}% | Temp Heatup {self.current_temp:.2f}"
