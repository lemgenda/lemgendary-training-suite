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
    def __init__(self, model_info, config=None, stabilizers=None):
        self.model_info = model_info
        self.config = config or {}
        opt = model_info.get("optimization", {})
        manifold_defaults = self.config.get("governor", {}).get("manifold", {})

        self.enabled = opt.get("enabled", True)

        # --- Curriculum Ladder ---
        self.res_ladder = opt.get("res_ladder")
        self.target_effective_batch = opt.get("target_effective_batch", manifold_defaults.get("target_effective_batch", 24))
        self.manifold_maturity = opt.get("manifold_maturity", manifold_defaults.get("maturity_soak", 5)) # 2026: Mandatory soak period (epochs)

        # 2026: Numerical Stress Audit (Sentinel Response)
        self.recovery_streak = 0

        # --- Persistent State ---
        self.current_fraction = opt.get("initial_fraction", manifold_defaults.get("initial_fraction", 0.5))
        self.current_batch = int(model_info.get("batch_size", 16)) if model_info.get("batch_size") and model_info.get("batch_size") != "auto" else 16
        self.current_acc = 1

        raw_size = model_info.get("input_size", 224)
        self.current_res = raw_size[1] if isinstance(raw_size, (list, tuple)) else raw_size

        # 2026: Dynamic Ladder Generation (Task 12.2.8)
        if not self.res_ladder:
            stride = manifold_defaults.get("resolution_stride", 128)
            max_res = manifold_defaults.get("max_resolution", 1024)
            # Build ladder starting from current_res up to max_res
            self.res_ladder = []
            curr = self.current_res
            while curr <= max_res:
                self.res_ladder.append(curr)
                curr += stride
            if not self.res_ladder: self.res_ladder = [self.current_res]

        self.stab = stabilizers or {}
        self.task_type = model_info.get("dataset_type", "quality")
        if isinstance(self.task_type, list): self.task_type = self.task_type[0]

        # --- 2026 Resilience: Hardware Resolution Cap (v19.1) ---
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8.0
        if vram_gb < 4.5 and self.task_type == "restoration":
            max_safe_res = 640
            self.res_ladder = [r for r in self.res_ladder if r <= max_safe_res]
            if not self.res_ladder: self.res_ladder = [max_safe_res]
            if self.current_res > max_safe_res:
                print(f" [GUARD] [GOVERNOR] Hardware Cap Active: Downscaling {self.current_res}px -> {max_safe_res}px for stability.")
                self.current_res = max_safe_res

        if self.current_res not in self.res_ladder:
            self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
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
        self.spatial_lock_remaining = 0 # 2026 v15.9: Blocks recoil after jump
        self.last_res_jump_epoch = -100

    def get_phase(self):
        res_idx = self.res_ladder.index(self.current_res)
        if res_idx == 0 and self.current_fraction < 0.5: return "FOUNDATION"
        if self.current_fraction < 0.95: return "EXPANSION"
        if res_idx < len(self.res_ladder) - 1: return "DEEPENING"
        return "REFINEMENT"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None, current_loss=None, plcc=0.0, force_jump=False):
        if not self.enabled and not force_jump: return False, False, False, False, False, False, ""
        self.epoch_count += 1
        self.session_epoch_count += 1

        if force_jump:
            try:
                # 2026 Resilience: Hardening Guard (v19.0)
                # We must stay at a resolution for at least manifold_maturity epochs before we can jump.
                # This ensures weights are stable even if SOTA was hit on the first epoch.
                epochs_at_res = self.epoch_count - self.last_res_jump_epoch
                print(f" [SEARCH] [HARDENING-DEBUG] Current Res: {self.current_res}px | Epochs at Res: {epochs_at_res} | Maturity Required: {self.manifold_maturity}")
                if epochs_at_res < self.manifold_maturity:
                    return False, False, False, False, False, False, f"[GUARD] [HARDENING] SOTA hit early, but locking at {self.current_res}px for weight stabilization (Manifold Maturity: {epochs_at_res}/{self.manifold_maturity})."

                current_idx = self.res_ladder.index(self.current_res)
                if current_idx < len(self.res_ladder) - 1:
                    next_res = self.res_ladder[current_idx + 1]
                    self.current_res = next_res
                    self.current_fraction = 0.5
                    self.last_res_jump_epoch = self.epoch_count
                    self.spatial_lock_remaining = 3
                    self.stabilization_epochs = 3
                    return True, True, False, False, False, True, f"[LAUNCH] [SOTA-FORCE] Jumping to {next_res}px Manifold..."
                else:
                    return False, False, False, False, False, False, "[SUCCESS] [SOTA-MAX] Already at maximum resolution."
            except: pass

        # 2026 Resilience: Resumption Shield
        # Ignore massive quality drops in the first epoch of a session (Momentum Shock)
        # unless loss also explodes (NaN).
        is_resuming = self.session_epoch_count == 1 and self.epoch_count > 1
        if is_resuming:
            # Bypass audit for resumption epoch
            self.prev_quality = current_quality
            if current_loss: self.prev_loss = current_loss
            return False, False, False, False, False, False, "[GUARD] [SHIELD] Resumption Shield Active. Buffering Momentum Shock."

        # --- NPP: Aggressive Recovery ---
        if sentinel_trigger_rate == 0:
            self.recovery_streak += 1
            if self.recovery_streak >= 2 and self.stabilization_epochs > 0:
                self.stabilization_epochs = 0
                print("[LAUNCH] [NPP] Stress at zero. Breaking stabilization lock.")
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
                msg_parts.append(f"[0x1f6a9] [BREAKOUT] Shield shattered! Quality dropped {(1-current_quality/self.best_quality)*100:.1f}%.")
                self.stabilization_epochs = 0
            else:
                self.stabilization_epochs -= 1
                # 2026: Recovery Velocity (Shorten cooldown if model is recovering fast)
                if current_quality - self.prev_quality > 0.05 and self.cooldown_remaining > 0:
                    self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
                    msg_parts.append("[0x26a1] [RECOVERY] Rapid quality gain detected. Meditation shortened.")

                self.prev_quality = current_quality
                if current_loss: self.prev_loss = current_loss
                status_msg = f"[SIGNAL] Anchoring Manifold... (Cooldown: {self.cooldown_remaining})" if self.cooldown_remaining > 0 else "[SIGNAL] Anchoring Manifold..."
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
            if is_flat: msg_parts.append("[0x1f578][0xfe0f] [TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        # 2026 NPP v15.6: Relaxed Regression Gate for high-noise Quality manifolds
        # Prevents "Panic Recoils" during natural SRCC/PLCC jitter
        regress_threshold = -0.03 if self.task_type == "quality" else -0.01
        is_regressing = delta_q < regress_threshold
        is_collapsed = (current_quality < 0.05) or (plcc < -0.1) # Near-zero or negative correlation

        # --- 2026 NPP v15.7: Momentum Guard ---
        # If Loss is stable/decreasing, we assume the regression is just metric noise.
        # We only retreat if BOTH quality drops AND loss explodes (>5% increase).
        loss_is_stable = (current_loss <= self.prev_loss * 1.05) if current_loss and self.prev_loss else True
        is_expanding = phase in ["FOUNDATION", "EXPANSION"]

        should_retreat = (is_regressing or is_turbulent or is_collapsed)
        # --- 2026: Thermal Shock Guard (v6.2.1) ---
        # If the linear correlation (PLCC) flips negative, the manifold is diffusing.
        # We must 'Shock' the temperature back to sharpness to restore bin separation.
        if plcc < -0.01 and self.current_temp > 0.7:
            self.current_temp = 0.5
            self.current_clamp = 20.0
            t_changed = c_changed = True
            msg_parts.append("[0x2744][0xfe0f] [THERMAL SHOCK] PLCC negative. Sharpening manifold (Temp -> 0.5).")

        # --- 2026 NPP v15.9: Spatial Lock ---
        if self.spatial_lock_remaining > 0:
            self.spatial_lock_remaining -= 1
            # Suppress quality regression unless loss is exploding (>25% increase)
            loss_is_exploding = (current_loss > self.prev_loss * 1.25) if current_loss and self.prev_loss else False
            if is_regressing and not loss_is_exploding:
                is_regressing = False
                msg_parts.append(f"[0x1f512] [SPATIAL LOCK] Buffering transition (Patience: {self.spatial_lock_remaining})")

        # 4. NPP LOOP DETECTION
        current_state = (self.current_res, round(self.current_fraction, 2))
        failures = self.failure_log.get(str(current_state), 0) # Store as string for JSON safety

        # --- 2026 NPP v15.8: Resonance Shield ---
        # Quality tasks (SRCC/PLCC) are naturally turbulent. We disable Turbulence Recoils
        # for these tasks and only rely on sustained Regression or Collapse.
        if self.task_type == "quality":
            is_turbulent = False # Turbulence is expected in high-entropy NIMA manifolds
            if is_expanding: msg_parts.append("[SIGNAL] [RESONANCE] Turbulence detected but shielded. Holding manifold.")

        should_retreat = (is_regressing or is_turbulent or is_collapsed)
        if is_expanding and loss_is_stable and not is_collapsed:
            should_retreat = False
            if is_regressing: msg_parts.append("[GUARD] [MOMENTUM] Jitter detected but Loss is stable. Holding manifold.")

        if should_retreat:
            self.failure_log[str(current_state)] = failures + 1
            msg_parts.append(f"[WARNING] NPP FAILURE: State {current_state} (Count: {self.failure_log[str(current_state)]})")

            # --- EMERGENCY RECOIL (v15.5) ---
            if self.failure_log[str(current_state)] >= 2:
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                t_changed = True
                msg_parts.append("[0x26d3][0xfe0f] NPP LOOP: Forcing Numerical Shakeup")


            # --- 2026 NPP v15.9: Strategic Spatial Retreat ---
            if self.epoch_count - self.last_res_jump_epoch < 8:
                # If we fail shortly after a jump, retreat to previous resolution at 100% data
                # instead of resetting the current resolution to 15% data.
                res_idx = self.res_ladder.index(self.current_res)
                if res_idx > 0:
                    self.current_res = self.res_ladder[res_idx - 1]
                    self.current_fraction = 1.0
                    r_changed = f_changed = True
                    self.lr_multiplier = 0.5
                    lr_changed = True
                    self.stabilization_epochs = 3
                    self.cooldown_remaining = 5
                    msg_parts.append(f"[0x21a9][0xfe0f] [SPATIAL RETREAT] Resetting to {self.current_res}px @ 100% Data Anchor")

            if not r_changed:
                old_frac = self.current_fraction
                self.current_fraction = max(0.15, self.current_fraction - 0.15)
                f_changed = True
                self.lr_multiplier = 0.7
                lr_changed = True
                self.stabilization_epochs = 1 if self.task_type == "quality" else 3
                self.cooldown_remaining = 3 if self.task_type == "quality" else 5
                msg_parts.append(f"RECOIL: Strategic retreat to {self.current_fraction*100:.0f}%")

        # --- PROACTIVE COOLING (2026 Resilience) ---
        elif sentinel_trigger_rate > 0.15:
            self.current_temp = min(1.2, self.current_temp * 1.2)
            self.lr_multiplier = 0.75
            lr_changed = True
            t_changed = True
            self.stabilization_epochs = 2
            msg_parts.append(f"[0x1f9ca] COOLING: Stress {sentinel_trigger_rate*100:.1f}% -> Temp {self.current_temp:.2f}")

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
                msg_parts.append(f"[0x26a1] JOLT: Breaking Plateau with {jolt:.2f}x LR Propulsion")


            next_frac = min(1.0, self.current_fraction + 0.15) # NPP: Smaller steps
            next_state = (self.current_res, round(next_frac, 2))

            if self.failure_log.get(str(next_state), 0) > 0:
                self.lr_multiplier = 0.6 # NPP: More cautious approach
                lr_changed = True
                msg_parts.append(f"[0x2693] ANCHOR: Caution ahead (Previous Failures). 0.6x LR.")

            if phase == "FOUNDATION" or phase == "EXPANSION":
                if self.current_fraction < 1.0:
                    self.current_fraction = next_frac
                    f_changed = True
                    msg_parts.append(f"PROPULSION: Data -> {self.current_fraction*100:.0f}%")
                    self.stabilization_epochs = 1
            elif phase == "DEEPENING":
                current_idx = self.res_ladder.index(self.current_res)
                next_res = self.res_ladder[current_idx + 1]
                self.current_res = next_res
                r_changed = b_changed = True
                self.current_fraction = 0.5
                f_changed = True
                self.last_res_jump_epoch = self.epoch_count
                self.spatial_lock_remaining = 3 # v15.9: 3 epochs of patience for the new resolution
                msg_parts.append(f"SPATIAL JUMP: {next_res}px | Data Reset 50% | Lock: ON")
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
                msg_parts.append(f"[0x1f48e] SHARPENING: Temp -> {self.current_temp:.2f}")
            elif self.cooldown_remaining > 0:
                msg_parts.append(f"[0x1f9d8] MEDITATION: Cooldown active ({self.cooldown_remaining} epochs)")

        self.prev_quality = current_quality
        if current_loss: self.prev_loss = current_loss

        # --- 2026: Universal Nuclear Safety Gate (Exit Point) ---
        if self.task_type == "quality":
            self.current_temp = min(1.0, self.current_temp)
            self.current_clamp = min(25.0, self.current_clamp) # Prevent logit explosion

        final_msg = f"[LAUNCH] [{phase}] " + " | ".join(msg_parts) if msg_parts else ""

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
            "spatial_lock_remaining": self.spatial_lock_remaining, # v15.9
            "last_res_jump_epoch": self.last_res_jump_epoch, # v15.9
            "epoch_count": self.epoch_count,
            "best_quality": self.best_quality
        }

    def load_state(self, state):
        if not state: return
        self.current_fraction = state.get("sample_fraction", self.current_fraction)
        raw_res = state.get("input_size", self.current_res)
        self.current_res = raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res
        self.current_temp = max(self.min_temp, state.get("softmax_temp", self.current_temp))
        if self.task_type == "quality": self.current_temp = min(1.0, self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)
        self.failure_log = state.get("failure_log", {}) # CRITICAL: Load memory
        self.thermal_floor = state.get("thermal_floor", {}) # New
        self.cooldown_remaining = state.get("cooldown_remaining", 0) # New
        self.spatial_lock_remaining = state.get("spatial_lock_remaining", 0) # v15.9
        self.last_res_jump_epoch = state.get("last_res_jump_epoch", -100) # v15.9
        self.epoch_count = state.get("epoch_count", self.epoch_count)
        self.best_quality = state.get("best_quality", self.best_quality)

    def recoil(self):
        """Emergency Tactical Retreat triggered by hardware or manifold failure."""
        old_frac = self.current_fraction
        self.current_fraction = max(0.15, old_frac - 0.15)
        self.current_temp = min(1.5, self.current_temp * 1.3)
        if self.task_type == "quality": self.current_temp = min(1.0, self.current_temp)
        self.stabilization_epochs = 3
        return f"[0x26a1] [NPP] RECOIL: Retreat to {self.current_fraction*100:.0f}% | Temp Heatup {self.current_temp:.2f}"

    def reset_best(self):
        """2026 Resilience: Memory Purge (v19.2).
        Resets the Governor's internal best metrics to allow for a fresh SOTA baseline
        at a new resolution rung.
        """
        self.best_quality = 0.0
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.history = []
        self.stabilization_epochs = 2 # Add a small soak period for the new baseline
        print(" [GOVERNOR] SOTA Memory Purged. Establishing fresh baseline for current manifold.")
