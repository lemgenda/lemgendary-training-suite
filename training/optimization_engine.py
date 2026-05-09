import os
import torch
import math
import numpy as np

class SmartTrainingGovernor:
    """
    2026 Universal Autonomous Optimization Engine (v15.5 Nuclear).
    
    Numerical Priority Protocol (NPP) Features:
    - State-Persistence Guard (Failure logs survive reloads)
    - Turbulence Dampening (Prevents noise-induced recoils)
    - Proportional Manifold Stride (Balanced scaling)
    - Surgical State-Loop Penalties (Blacklists breaking points)
    """
    def __init__(self, model_info, stabilizers=None):
        self.model_info = model_info
        opt = model_info.get("optimization", {})
        self.enabled = opt.get("enabled", True)
        
        # --- Curriculum Ladder ---
        self.res_ladder = opt.get("res_ladder", [224, 384, 512, 640])
        self.target_effective_batch = opt.get("target_effective_batch", 24)
        
        # 2026: Numerical Stress Audit (Sentinel Response)
        self.recovery_streak = 0
        
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
        
        # --- Surgical Memory (v15.5) ---
        self.history = [] # Last 5 epochs [quality, loss]
        self.failure_log = {} # {(res, round(frac,2)): failure_count}
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.best_quality = 0.0
        self.stabilization_epochs = 0
        self.cooldown_remaining = 0 # New: Blocks Jolt/Sharpening after failure
        self.thermal_floor = {} # New: {(res, frac): min_safe_temp}
        self.lr_multiplier = 1.0
        self.last_action_epoch = 0
        self.epoch_count = 0
        self.session_epoch_count = 0 # 2026: Resumption Shield tracking
        self.min_delta = opt.get("min_delta", 0.0005)
        
    def get_phase(self):
        res_idx = self.res_ladder.index(self.current_res)
        if res_idx == 0 and self.current_fraction < 0.5: return "FOUNDATION"
        if self.current_fraction < 0.95: return "EXPANSION"
        if res_idx < len(self.res_ladder) - 1: return "DEEPENING"
        return "REFINEMENT"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None, current_loss=None):
        if not self.enabled: return False, False, False, False, False, False, ""
        self.epoch_count += 1
        self.session_epoch_count += 1
        
        # 2026 Resilience: Resumption Shield
        # Ignore massive quality drops in the first epoch of a session (Momentum Shock)
        # unless loss also explodes (NaN).
        is_resuming = self.session_epoch_count == 1 and self.epoch_count > 1
        if is_resuming:
            # Bypass audit for resumption epoch
            self.prev_quality = current_quality
            if current_loss: self.prev_loss = current_loss
            return False, False, False, False, False, False, "🛡️ [SHIELD] Resumption Shield Active. Buffering Momentum Shock."

        # --- NPP: Aggressive Recovery ---
        if sentinel_trigger_rate == 0:
            self.recovery_streak += 1
            if self.recovery_streak >= 2 and self.stabilization_epochs > 0:
                self.stabilization_epochs = 0
                print("🚀 [NPP] Stress at zero. Breaking stabilization lock.")
        else:
            self.recovery_streak = 0
        
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        
        # 1. Update Memory
        self.history.append((current_quality, current_loss))
        if len(self.history) > 5: self.history.pop(0)
        self.best_quality = max(self.best_quality, current_quality)
        
        # 2. Guard: Stabilization
        msg_parts = []
        # Senior Update: Emergency Breakout if manifold is clearly collapsing
        if self.stabilization_epochs > 0:
            if current_quality < self.best_quality * 0.90 and self.best_quality > 0:
                msg_parts.append(f"🚩 [BREAKOUT] Shield shattered! Quality dropped {(1-current_quality/self.best_quality)*100:.1f}%.")
                self.stabilization_epochs = 0
            else:
                self.stabilization_epochs -= 1
                # 2026: Recovery Velocity (Shorten cooldown if model is recovering fast)
                if current_quality - self.prev_quality > 0.05 and self.cooldown_remaining > 0:
                    self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
                    msg_parts.append("⚡ [RECOVERY] Rapid quality gain detected. Meditation shortened.")
                
                self.prev_quality = current_quality
                if current_loss: self.prev_loss = current_loss
                status_msg = f"📡 Anchoring Manifold... (Cooldown: {self.cooldown_remaining})" if self.cooldown_remaining > 0 else "📡 Anchoring Manifold..."
                if msg_parts: status_msg = " | ".join(msg_parts) + " | " + status_msg
                return False, False, False, False, False, False, status_msg

        f_changed = r_changed = lr_changed = t_changed = c_changed = b_changed = False
        self.lr_multiplier = 1.0
        phase = self.get_phase()

        # 3. NPP Diagnosis: Turbulence Dampening
        delta_q = current_quality - self.prev_quality
        is_turbulent = False
        if len(self.history) >= 3:
            q_values = [h[0] for h in self.history[-3:]]
            deltas = [q_values[i] - q_values[i-1] for i in range(1, len(q_values))]
            # NPP: Turbulence requires a minimum magnitude to prevent noise triggers
            if all(deltas[i] * deltas[i-1] < 0 for i in range(1, len(deltas))):
                if all(abs(d) > self.min_delta * 2 for d in deltas):
                    is_turbulent = True

        is_flat = abs(delta_q) < self.min_delta and len(self.history) >= 2
        
        # 2026 NPP: Fidelity Floor (Task 12.7)
        # If the model is stuck in a low-quality manifold (e.g. Accuracy high but SRCC low),
        # we relax the 'flatness' constraint to allow Jolts through the noise.
        fidelity_floor = self.best_quality * 0.8
        is_trapped = current_quality < fidelity_floor and len(self.history) >= 4
        if is_trapped:
            # Relax min_delta by 4x to ignore low-level metric jitter
            effective_min_delta = self.min_delta * 4
            is_flat = abs(delta_q) < effective_min_delta
            if is_flat: msg_parts.append("🕸️ [TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        is_regressing = delta_q < -0.01 
        is_collapsed = current_quality < 0.05 # Near-zero or negative correlation
        
        # 4. NPP LOOP DETECTION
        current_state = (self.current_res, round(self.current_fraction, 2))
        failures = self.failure_log.get(str(current_state), 0) # Store as string for JSON safety
        
        if is_regressing or is_turbulent or is_collapsed:
            self.failure_log[str(current_state)] = failures + 1
            msg_parts.append(f"⚠️ NPP FAILURE: State {current_state} (Count: {self.failure_log[str(current_state)]})")
            
            # --- EMERGENCY RECOIL (v15.5) ---
            if self.failure_log[str(current_state)] >= 2:
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                t_changed = True
                msg_parts.append("⛓️ NPP LOOP: Forcing Numerical Shakeup")
            
            old_frac = self.current_fraction
            self.current_fraction = max(0.15, self.current_fraction - 0.15)
            f_changed = True
            self.lr_multiplier = 0.7 
            lr_changed = True
            self.stabilization_epochs = 3
            self.cooldown_remaining = 5 # NPP Loop Mitigation: Force 5-epoch meditation
            
            # Record the breaking point temperature
            self.thermal_floor[str(current_state)] = self.current_temp * 1.1
            
            msg_parts.append(f"RECOIL: Strategic retreat to {self.current_fraction*100:.0f}%")

        # --- PROACTIVE COOLING (2026 Resilience) ---
        elif sentinel_trigger_rate > 0.15:
            self.current_temp = min(1.2, self.current_temp * 1.2)
            self.lr_multiplier = 0.75
            lr_changed = True
            t_changed = True
            self.stabilization_epochs = 2
            msg_parts.append(f"🧊 COOLING: Stress {sentinel_trigger_rate*100:.1f}% -> Temp {self.current_temp:.2f}")

        # --- PROPULSION: NPP Manifold Stride ---
        # 2026: Dynamic Stride Thresholds (Foundation vs Refinement)
        stride_threshold = 0.75 if self.current_res < 512 else 0.90
        if is_flat or (current_quality > stride_threshold and delta_q < self.min_delta):
            # 2026: The Jolt - Breaking Plateaus with LR Propulsion
            # Senior Update: Added Jolt cooldown (5 epochs)
            jolt_ready = (self.epoch_count - getattr(self, 'last_jolt_epoch', -10)) > 5 and self.cooldown_remaining == 0
            if is_flat and jolt_ready:
                jolt = self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5)
                # NPP: If trapped, increase Jolt intensity to break the classification trap
                if is_trapped: jolt *= 1.5
                self.lr_multiplier = float(jolt)
                lr_changed = True
                self.last_jolt_epoch = self.epoch_count
                msg_parts.append(f"⚡ JOLT: Breaking Plateau with {jolt:.2f}x LR Propulsion")

            
            next_frac = min(1.0, self.current_fraction + 0.15) # NPP: Smaller steps
            next_state = (self.current_res, round(next_frac, 2))
            
            if self.failure_log.get(str(next_state), 0) > 0:
                self.lr_multiplier = 0.6 # NPP: More cautious approach
                lr_changed = True
                msg_parts.append(f"⚓ ANCHOR: Caution ahead (Previous Failures). 0.6x LR.")
            
            if phase == "FOUNDATION" or phase == "EXPANSION":
                if self.current_fraction < 1.0:
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

        # --- Senior Feature: Gradual Temperature Sharpening (Success Branch) ---
        if not (is_regressing or is_turbulent or sentinel_trigger_rate > 0.15) and self.current_temp > self.min_temp:
            # 2026: VLM Temperature Relaxation (Foundation vs Refinement)
            if self.task_type != "quality":
                phase_min = 0.05 if phase == "REFINEMENT" else 0.1
            else:
                phase_min = self.min_temp # NIMA remains at 0.5 for stability
                
            # Check thermal floor for current state
            floor = max(phase_min, self.thermal_floor.get(str(current_state), self.min_temp))
            
            if self.cooldown_remaining == 0 and self.current_temp > floor:
                # 2026: Accelerated Sharpening for high-entropy phases
                sharpen_rate = 0.95 if self.current_temp > 1.2 else 0.98
                self.current_temp = max(floor, self.current_temp * sharpen_rate)
                t_changed = True
                msg_parts.append(f"💎 SHARPENING: Temp -> {self.current_temp:.2f}")
            elif self.cooldown_remaining > 0:
                msg_parts.append(f"🧘 MEDITATION: Cooldown active ({self.cooldown_remaining} epochs)")

        self.prev_quality = current_quality
        if current_loss: self.prev_loss = current_loss
        
        final_msg = f"🚀 [{phase}] " + " | ".join(msg_parts) if msg_parts else ""
            
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
            "failure_log": self.failure_log, # CRITICAL: Persist memory
            "thermal_floor": self.thermal_floor, # New
            "cooldown_remaining": self.cooldown_remaining, # New
            "epoch_count": self.epoch_count,
            "best_quality": self.best_quality
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
        self.failure_log = state.get("failure_log", {}) # CRITICAL: Load memory
        self.thermal_floor = state.get("thermal_floor", {}) # New
        self.cooldown_remaining = state.get("cooldown_remaining", 0) # New
        self.epoch_count = state.get("epoch_count", self.epoch_count)
        self.best_quality = state.get("best_quality", self.best_quality)

    def recoil(self):
        """Emergency Tactical Retreat triggered by hardware or manifold failure."""
        old_frac = self.current_fraction
        self.current_fraction = max(0.15, old_frac - 0.15)
        self.current_temp = min(1.5, self.current_temp * 1.3)
        self.stabilization_epochs = 3
        return f"⚡ [NPP] RECOIL: Retreat to {self.current_fraction*100:.0f}% | Temp Heatup {self.current_temp:.2f}"
