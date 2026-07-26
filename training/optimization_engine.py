import os
import torch
import math
import numpy as np
from training.telemetry import METRIC_DIRECTIONS, METRIC_WEIGHTS




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
        self.plateau_patience = opt.get("plateau_patience", self.config.get("governor", {}).get("plateau_patience", 6))
        
        # --- 2026 Resiliency: Dynamic Parameter Implementations ---
        self.plateau_priority = opt.get("plateau_priority", "data")
        self.fraction_increment = opt.get("fraction_increment", manifold_defaults.get("fraction_increment", 0.15))
        self.cooling_factor = opt.get("cooling_factor", 0.5) # Default 0.5 if not provided
        self.clamp_range = opt.get("clamp_range", [15.0, 45.0])
        gov_cfg = self.config.get("governor", {})
        self.jolt_cooldown = gov_cfg.get("jolt_cooldown_epochs", 5)
        self.stabilization_lock = gov_cfg.get("stabilization_lock_epochs", 3)
        self.breakout_threshold = gov_cfg.get("emergency_breakout_threshold", 0.10)
        self.sharpening_rate = gov_cfg.get("sharpening_cooling_rate", 0.98)

        # 2026: Numerical Stress Audit (Sentinel Response)
        self.recovery_streak = 0

        # --- Persistent State ---
        self.current_fraction = opt.get("initial_fraction", manifold_defaults.get("initial_fraction", 0.15))
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
        if vram_gb < 4.5 and self.task_type in ["restoration", "parameter_prediction"]:
            max_safe_res = 640 if self.task_type == "restoration" else 256
            self.res_ladder = [r for r in self.res_ladder if r <= max_safe_res]
            if not self.res_ladder: self.res_ladder = [max_safe_res]
            if self.current_res > max_safe_res:
                print(f" [GUARD] [GOVERNOR] Hardware Cap Active: Downscaling {self.current_res}px -> {max_safe_res}px for stability.")
                self.current_res = max_safe_res

        # 2026: Ensure training starts at the lowest resolution in the ladder for a fresh run
        if self.res_ladder and self.current_res != self.res_ladder[0]:
            print(f" [GUARD] [GOVERNOR] Aligning start resolution {self.current_res}px -> lowest rung {self.res_ladder[0]}px.")
            self.current_res = self.res_ladder[0]

        if self.current_res not in self.res_ladder:
            self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
        self.min_temp = float(self.stab.get("min_temp", 0.5 if self.task_type == "quality" else (0.01 if self.task_type == "parameter_prediction" else 0.1)))
        self.current_temp = self.stab.get("softmax_temp", self.min_temp)
        self.current_clamp = self.stab.get("logit_clamp", 15.0)

        # --- 2026 SOTA Dynamic Governance Targets ---
        self.current_rank_weight = float(self.stab.get("rank_weight", 0.8))
        self.current_rank_margin = float(self.stab.get("rank_margin", 0.10))

        # --- 2026 Resilience: Adaptive Anti-Loop Governance ---
        self.loop_breaker_enabled = opt.get("loop_breaker_enabled", True)
        self.loop_breaker_threshold = opt.get("loop_breaker_threshold", 2)
        self.loop_breaker_strategy = opt.get("loop_breaker_strategy", "auto") # "auto", "escalate", "relax", "none"
        self.consecutive_rollbacks = 0
        self.rollback_history = {}  # {resolution: rollback_count}
        self.gate_relaxation_epochs = 0
        self.sota_resolution = None
        self.breakout_lock = 0

        # --- Surgical Memory (v15.5) ---
        self.history = [] # Last 5 epochs [quality, loss]
        self.failure_log = {} # {(res, round(frac,2)): failure_count}
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.best_quality = 0.0
        self.stabilization_epochs = 0
        self.cooldown_remaining = 0 # New: Blocks Jolt/Sharpening after failure
        self.current_stress = 0.0 # 2026: Dynamic Stress Protocol Multiplier
        self.thermal_floor = {} # New: {(res, frac): min_safe_temp}
        self.lr_multiplier = 1.0
        self.last_action_epoch = 0
        self.epoch_count = 0
        self.session_epoch_count = 0 # 2026: Resumption Shield tracking
        self.max_stress_stuck_epochs = 0 # 2026 v16: Counts epochs at max stress with no progress
        # 2026 Resilience: Dynamic Delta for High-Range Quality Scores
        # Restoration tasks have Quality Scores in the 100s-500s range, making 0.0005 too small to ever trigger 'is_flat'.
        base_delta = opt.get("min_delta", 0.0005)
        # Parameter prediction uses MAE in [0,1] range - similar scale to quality tasks
        if self.task_type == "parameter_prediction":
            self.min_delta = base_delta
        else:
            self.min_delta = base_delta if self.task_type == "quality" else (base_delta * 100.0) # Scale up for restoration
        self.spatial_lock_remaining = 0 # 2026 v15.9: Blocks recoil after jump
        self.last_res_jump_epoch = 0

        # Calculate target Quality Score for non-quality tasks with sota_targets to scale stride_threshold
        self.sota_targets = opt.get("sota_targets", model_info.get("sota_targets", {}))
        self.target_quality_score = 1.0
        if self.sota_targets:
            target_score = 0.0
            for k, target_v in self.sota_targets.items():
                direction = METRIC_DIRECTIONS.get(k, True)
                weight = METRIC_WEIGHTS.get(k, 1)
                if direction:
                    target_score += target_v * weight
                else:
                    if k == 'fid': target_score += (100.0 - target_v) * weight
                    elif k == 'lpips': target_score += (1.0 - target_v) * weight
                    elif k == 'rank_margin': target_score += (10.0 - target_v) * weight
                    else: target_score += (1.0 / (target_v + 1e-6)) * weight
            if target_score > 0:
                if target_score <= 1.0 and self.task_type == "quality":
                    target_score *= 100.0
                self.target_quality_score = target_score

    def get_phase(self):
        res_idx = self.res_ladder.index(self.current_res)
        if res_idx == 0 and self.current_fraction < 0.5: return "FOUNDATION"
        
        if self.plateau_priority == "resolution":
            if res_idx < len(self.res_ladder) - 1: return "DEEPENING"
            if self.current_fraction < 1.0: return "EXPANSION"
        else: # Default: data priority
            if self.current_fraction < 1.0: return "EXPANSION"
            if res_idx < len(self.res_ladder) - 1: return "DEEPENING"
            
        return "REFINEMENT"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None, current_loss=None, plcc=0.0, target_std=None, force_jump=False, train_loss=None):
        if not self.enabled and not force_jump: return False, False, False, False, False, False, ""
        self.epoch_count += 1
        self.session_epoch_count += 1

        if getattr(self, 'breakout_lock', 0) > 0:
            self.breakout_lock -= 1

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
                    self.spatial_lock_remaining = self.stabilization_lock
                    self.stabilization_epochs = self.stabilization_lock
                    self.history = [] # 2026: Clear history to prevent cross-resolution trend contamination
                    return True, True, False, False, False, True, f"[LAUNCH] [SOTA-FORCE] Jumping to {next_res}px Manifold..."
                else:
                    return False, False, False, False, False, False, "[SUCCESS] [SOTA-MAX] Already at maximum resolution."
            except: pass

        # --- NPP: Aggressive Recovery ---
        if sentinel_trigger_rate == 0:
            self.recovery_streak += 1
            if self.recovery_streak >= 2 and self.stabilization_epochs > 0 and self.spatial_lock_remaining == 0:
                self.stabilization_epochs = 0
                print("[LAUNCH] [NPP] Stress at zero. Breaking stabilization lock.")
        else:
            self.recovery_streak = 0

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # 1. Update Memory
        self.history.append((current_quality, current_loss, train_loss))
        if len(self.history) > 5: self.history.pop(0)
        
        # --- 2026 NPP: Stress Deactivation ---
        # If the model successfully breaks the previous SOTA ceiling, shut off all dataset noise
        # so it can peacefully anchor the new manifold.
        if current_quality > self.best_quality and self.best_quality > 0.0:
            if getattr(self, 'current_stress', 0.0) > 0.0:
                self.current_stress = 0.0
                
        self.best_quality = max(self.best_quality, current_quality)

        # 2. Guard: Stabilization
        msg_parts = []
        if self.stabilization_epochs > 0:
            if current_quality < self.best_quality * (1.0 - self.breakout_threshold) and self.best_quality > 0:
                msg_parts.append(f"[BREAKOUT] Shield shattered! Quality dropped {(1-current_quality/self.best_quality)*100:.1f}%.")
                self.stabilization_epochs = 0
            else:
                self.stabilization_epochs -= 1
                # 2026: Recovery Velocity (Shorten cooldown and lock if model is recovering fast)
                recovery_delta = self.min_delta if self.task_type != "quality" else 0.05
                if current_quality - self.prev_quality > recovery_delta:
                    self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
                    self.stabilization_epochs = max(0, self.stabilization_epochs - 1)
                    msg_parts.append("[RECOVERY] Rapid quality gain detected. Cooldown and Lock shortened.")

                self.prev_quality = current_quality
                if current_loss: self.prev_loss = current_loss
                status_msg = f"[SIGNAL] Anchoring Manifold... (Cooldown: {self.cooldown_remaining})" if self.cooldown_remaining > 0 else "[SIGNAL] Anchoring Manifold..."
                if msg_parts: status_msg = " | ".join(msg_parts) + " | " + status_msg
                return False, False, False, False, False, False, status_msg

        # 2026 Resilience: Resumption Shield
        # Ignore massive quality drops in the first epoch of a session (Momentum Shock)
        # unless loss also explodes (NaN).
        is_resuming = self.session_epoch_count == 1 and self.epoch_count > 1
        is_regressing_shock = current_quality < self.best_quality * 0.95 and self.best_quality > 0
        if is_resuming and is_regressing_shock:
            # Bypass audit for resumption epoch to buffer momentum shock
            self.prev_quality = current_quality
            if current_loss: self.prev_loss = current_loss
            return False, False, False, False, False, False, "[GUARD] [SHIELD] Resumption Shield Active. Buffering Momentum Shock."

        # Overfitting Detection (NPP): check training vs validation loss trends over the last 3 epochs
        is_overfitting = False
        if len(self.history) >= 3:
            train_losses = [h[2] for h in self.history if len(h) > 2 and h[2] is not None]
            val_losses = [h[1] for h in self.history if h[1] is not None]
            if len(train_losses) >= 3 and len(val_losses) >= 3:
                # Train loss decreasing and Val loss increasing
                train_trend = train_losses[-1] - train_losses[-3]
                val_trend = val_losses[-1] - val_losses[-3]
                if train_trend < -1e-4 and val_trend > 1e-4:
                    is_overfitting = True

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

        # Stagnation check: check if we have hit plateau patience
        is_plateaued = (epochs_no_improve >= self.plateau_patience)
        is_flat_leg = abs(delta_q) < self.min_delta and len(self.history) >= 2
        is_flat = is_plateaued or is_flat_leg

        # 2026 NPP: Fidelity Floor (Task 12.7)
        # If the model is stuck in a low-quality manifold (e.g. Accuracy high but SRCC low),
        # we relax the 'flatness' constraint to allow Jolts through the noise.
        # Hardened Logic: Includes absolute threshold for quality tasks.
        abs_floor = 40.0 if self.task_type == "quality" else 0.0
        fidelity_floor = max(abs_floor, self.best_quality * 0.8)
        is_trapped = current_quality < fidelity_floor and len(self.history) >= 4
        if is_trapped:
            # Relax min_delta by 4x to ignore low-level metric jitter
            effective_min_delta = self.min_delta * 4
            is_flat_leg = abs(delta_q) < effective_min_delta
            is_flat = is_plateaued or is_flat_leg
            if is_flat: msg_parts.append("[TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        # 2026 NPP v15.6: Relaxed Regression Gate for high-noise Quality manifolds
        # Prevents "Panic Recoils" during natural SRCC/PLCC jitter
        regress_threshold = -0.03 if self.task_type in ["quality", "parameter_prediction"] else -0.01
        is_regressing = delta_q < (self.prev_quality * regress_threshold) if self.prev_quality else False
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
        # Low-Variance Guard: Skip thermal shock if target standard deviation is extremely narrow (< 0.15)
        if plcc < -0.01 and self.current_temp > 0.7 and (target_std is None or target_std >= 0.15):
            self.current_temp = 0.5
            self.current_clamp = 20.0
            t_changed = c_changed = True
            msg_parts.append("[THERMAL SHOCK] PLCC negative. Sharpening manifold (Temp -> 0.5).")

        # --- 2026 NPP v15.9: Spatial Lock ---
        if self.spatial_lock_remaining > 0:
            self.spatial_lock_remaining -= 1
            # Suppress quality regression unless loss is exploding (>25% increase)
            loss_is_exploding = (current_loss > self.prev_loss * 1.25) if current_loss and self.prev_loss else False
            if is_regressing and not loss_is_exploding:
                is_regressing = False
                msg_parts.append(f"[SPATIAL LOCK] Buffering transition (Patience: {self.spatial_lock_remaining})")

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

        # --- Overfitting Rescue Protocol ---
        if is_overfitting and is_expanding:
            should_retreat = False
            self.cooldown_remaining = 0
            msg_parts.append("[RESCUE] [OVERFITTING] Overfitting detected. Forcing dataset expansion to introduce variety.")
        elif is_overfitting and self.current_fraction >= 1.0:
            should_retreat = False
            self.cooldown_remaining = 0
            self.current_stress = min(5.0, self.current_stress + 1.0)
            msg_parts.append(f"[RESCUE] [OVERFITTING] Dataset exhausted. Deploying Stress Protocol (Level {self.current_stress}).")

        if should_retreat:
            self.failure_log[str(current_state)] = failures + 1
            msg_parts.append(f"[WARNING] NPP FAILURE: State {current_state} (Count: {self.failure_log[str(current_state)]})")

            # --- EMERGENCY RECOIL (v15.5) ---
            if self.failure_log[str(current_state)] >= 2:
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                t_changed = True
                msg_parts.append("NPP LOOP: Forcing Numerical Shakeup")


            # --- 2026 NPP v16.0: Proven-Manifold Protection & Strategic Recoil ---
            # Check if current resolution has already proven high fidelity (e.g. Quality Score > 85.0 or 75% of SOTA target)
            proven_threshold = max(85.0 if self.task_type == "quality" else 0.85, self.target_quality_score * 0.75)
            is_proven_manifold = self.best_quality >= proven_threshold

            can_spatial_retreat = (
                self.epoch_count - self.last_res_jump_epoch < 8 and 
                getattr(self, 'breakout_lock', 0) == 0 and 
                not is_proven_manifold
            )

            if can_spatial_retreat:
                # If we fail shortly after a jump AND the resolution never proved high quality,
                # retreat to previous resolution at 100% data anchor.
                res_idx = self.res_ladder.index(self.current_res)
                if res_idx > 0:
                    self.current_res = self.res_ladder[res_idx - 1]
                    self.current_fraction = 1.0
                    r_changed = f_changed = True
                    self.lr_multiplier = self.cooling_factor
                    lr_changed = True
                    self.stabilization_epochs = self.stabilization_lock
                    self.cooldown_remaining = 5
                    msg_parts.append(f"[SPATIAL RETREAT] Resetting to {self.current_res}px @ 100% Data Anchor")

            if not r_changed:
                # --- Intra-Resolution Data Recoil ---
                # If spatial retreat is blocked (proven manifold) or at base resolution,
                # step dataset fraction back to the last safe fraction if we expanded past initial_fraction.
                opt_cfg = self.model_info.get("optimization", {})
                initial_frac = opt_cfg.get("initial_fraction", 0.15)
                if self.current_fraction > initial_frac:
                    prev_frac = max(initial_frac, round(self.current_fraction - self.fraction_increment, 2))
                    if prev_frac < self.current_fraction:
                        self.current_fraction = prev_frac
                        f_changed = True
                        msg_parts.append(f"[DATA RECOIL] Stepping back dataset fraction: {self.current_fraction*100:.0f}% @ {self.current_res}px")

                self.lr_multiplier = self.cooling_factor
                lr_changed = True
                self.stabilization_epochs = self.stabilization_lock
                self.cooldown_remaining = 5
                if not f_changed:
                    msg_parts.append(f"RECOIL: Retaining data fraction at {self.current_fraction*100:.0f}% | Cooling LR to stabilize manifold")

        # --- PROACTIVE COOLING (2026 Resilience) ---
        elif sentinel_trigger_rate > 0.15:
            self.current_temp = min(1.2, self.current_temp * 1.2)
            self.lr_multiplier = 0.75
            self.current_stress = max(0.0, self.current_stress - 0.25) # Cool stress down
            lr_changed = True
            t_changed = True
            self.stabilization_epochs = 2
            msg_parts.append(f"COOLING: Stress {sentinel_trigger_rate*100:.1f}% -> Temp {self.current_temp:.2f} | DataStress {self.current_stress:.1f}")

        # --- PROPULSION: NPP Manifold Stride ---
        # 2026: Dynamic Stride Thresholds (Foundation vs Refinement)
        # 2026 v15.10: Propulsion is BLOCKED during cooldown/recoil to prevent
        # data fraction ramps while the model is supposed to be stabilizing.
        stride_threshold = 0.75 if self.current_res < 512 else 0.90
        # Scale threshold to match the task's score range.
        if self.target_quality_score > 1.0:
            stride_threshold = stride_threshold * self.target_quality_score
        propulsion_allowed = not should_retreat and self.cooldown_remaining == 0
        not_regressing = delta_q >= -self.min_delta
        
        # Overfitting overrides stagnation/cooldown to force data expansion
        trigger_propulsion = propulsion_allowed and (
            (is_overfitting and is_expanding) or 
            is_plateaued or 
            (not_regressing and (is_flat_leg or (current_quality > stride_threshold and delta_q < self.min_delta)))
        )
        if trigger_propulsion:
            # 2026: The Jolt - Breaking Plateaus with LR Propulsion
            # Senior Update: Added Jolt cooldown (5 epochs)
            jolt_ready = (self.epoch_count - getattr(self, 'last_jolt_epoch', -10)) > self.jolt_cooldown and self.cooldown_remaining == 0
            if is_flat and jolt_ready:
                jolt = self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5)
                # NPP: If trapped, increase Jolt intensity to break the classification trap
                if is_trapped: jolt *= 1.5
                self.lr_multiplier = float(jolt)
                lr_changed = True
                self.last_jolt_epoch = self.epoch_count
                msg_parts.append(f"JOLT: Breaking Plateau with {jolt:.2f}x LR Propulsion")


            next_frac = min(1.0, self.current_fraction + self.fraction_increment) # NPP: Smaller steps
            next_state = (self.current_res, round(next_frac, 2))

            if self.failure_log.get(str(next_state), 0) > 0:
                self.lr_multiplier = 0.6 # NPP: More cautious approach
                lr_changed = True
                msg_parts.append(f"ANCHOR: Caution ahead (Previous Failures). 0.6x LR.")

            if phase == "FOUNDATION" or phase == "EXPANSION":
                if self.current_fraction < 1.0:
                    self.current_fraction = next_frac
                    f_changed = True
                    msg_parts.append(f"PROPULSION: Data -> {self.current_fraction*100:.0f}%")
                    self.stabilization_epochs = 1
                    self.best_quality = current_quality
            elif phase == "DEEPENING":
                current_idx = self.res_ladder.index(self.current_res)
                next_res = self.res_ladder[current_idx + 1]
                
                # SOTA Validation Check before spatial jump
                if self.best_quality < self.target_quality_score * 0.80 and self.target_quality_score > 1.0:
                    self.lr_multiplier = self.cooling_factor
                    lr_changed = True
                    msg_parts.append(f"RECOIL: Insufficient Quality for spatial jump. Cooling LR.")
                    self.stabilization_epochs = self.stabilization_lock
                else:
                    self.current_res = next_res
                    r_changed = b_changed = True
                    self.current_fraction = 0.15
                    f_changed = True
                    self.last_res_jump_epoch = self.epoch_count
                    self.spatial_lock_remaining = self.stabilization_lock # patience for the new resolution
                    msg_parts.append(f"SPATIAL JUMP: {next_res}px | Data Reset 15% | Lock: ON")
                    self.stabilization_epochs = self.stabilization_lock
            else:
                # New Rule: If plateaued far from SOTA goal, deploy Stress to break local minima
                if getattr(self, 'current_stress', 0.0) < 5.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                    self.current_stress = min(5.0, getattr(self, 'current_stress', 0.0) + 1.0)
                    self.lr_multiplier = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
                    lr_changed = True
                    self.max_stress_stuck_epochs = 0 # Reset stuck counter: stress is still escalating
                    msg_parts.append(f"REFINEMENT: Trapped in Plateau. Deploying Stress Protocol (Level {self.current_stress}) & Jolting LR")
                elif getattr(self, 'current_stress', 0.0) >= 5.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                    # 2026 v16: Stress is maxed but SOTA is still far away.
                    # Do NOT cool the LR — that crushes momentum and freezes the model permanently.
                    # Instead, force a periodic jolt to keep probing the manifold.
                    jolt = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
                    self.lr_multiplier = jolt
                    lr_changed = True
                    self.max_stress_stuck_epochs = getattr(self, 'max_stress_stuck_epochs', 0) + 1
                    stuck_patience = self.plateau_patience * 2  # 2× plateau patience before declaring stuck
                    msg_parts.append(f"REFINEMENT: [MAX STRESS] Forcing Jolt (x{jolt:.2f}) to maintain momentum (Stuck: {self.max_stress_stuck_epochs}/{stuck_patience})")
                    if self.max_stress_stuck_epochs >= stuck_patience:
                        msg_parts.append(f"[STUCK] Max stress reached and no improvement for {self.max_stress_stuck_epochs} epochs. Architecture may be at capacity. Consider stopping or switching backbone.")
                else:
                    self.lr_multiplier = self.cooling_factor
                    lr_changed = True
                    msg_parts.append("REFINEMENT: SOTA Precision Cooling")

                    # --- 2026 Autonomous SOTA Adaptations ---
                    # Dynamically tune loss parameters on the fly without mid-training YAML changes
                    if self.task_type == "quality":
                        target_srcc = float(self.sota_targets.get("srcc", 0.9100)) if isinstance(self.sota_targets, dict) else 0.9100
                        target_emd  = float(self.sota_targets.get("rank_margin", 0.0700)) if isinstance(self.sota_targets, dict) else 0.0700
                        
                        # 1. Dynamically boost rank_weight (0.8 -> 1.5) if SRCC is below SOTA target
                        if plcc < target_srcc or (self.best_quality < self.target_quality_score):
                            if self.current_rank_weight < 1.5:
                                self.current_rank_weight = min(1.5, round(self.current_rank_weight + 0.1, 2))
                                msg_parts.append(f"[AUTONOMOUS GOVERNOR] Dynamic SRCC Boost: Rank Weight -> {self.current_rank_weight}")
                        
                        # 2. Dynamically tighten rank_margin (0.10 -> 0.05) when converging
                        if self.current_rank_margin > 0.05 and current_quality > (self.target_quality_score * 0.70):
                            self.current_rank_margin = max(0.05, round(self.current_rank_margin - 0.01, 2))
                            msg_parts.append(f"[AUTONOMOUS GOVERNOR] Dynamic Rank Margin Tightening -> {self.current_rank_margin}")

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
                sharpen_rate = 0.95 if self.current_temp > 1.2 else self.sharpening_rate
                self.current_temp = max(floor, self.current_temp * sharpen_rate)
                t_changed = True
                msg_parts.append(f"SHARPENING: Temp -> {self.current_temp:.2f}")
            elif self.cooldown_remaining > 0:
                msg_parts.append(f"MEDITATION: Cooldown active ({self.cooldown_remaining} epochs)")

        self.prev_quality = current_quality
        if current_loss: self.prev_loss = current_loss

        # --- 2026: Universal Nuclear Safety Gate (Exit Point) ---
        if self.task_type == "quality":
            # 2026 Resilience: Respect active recoils (allow >1.0 temp during shakeup)
            if self.cooldown_remaining == 0:
                self.current_temp = min(1.0, self.current_temp)
            self.current_clamp = min(self.clamp_range[1], self.current_clamp) # Prevent logit explosion

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
            "history": self.history, # CRITICAL: Persist plateau memory
            "thermal_floor": self.thermal_floor, # New
            "cooldown_remaining": self.cooldown_remaining, # New
            "spatial_lock_remaining": self.spatial_lock_remaining, # v15.9
            "last_res_jump_epoch": self.last_res_jump_epoch, # v15.9
            "epoch_count": self.epoch_count,
            "best_quality": self.best_quality,
            "stress": getattr(self, 'current_stress', 0.0),
            "last_jolt_epoch": getattr(self, 'last_jolt_epoch', -10),
            "max_stress_stuck_epochs": getattr(self, 'max_stress_stuck_epochs', 0),
            "consecutive_rollbacks": getattr(self, 'consecutive_rollbacks', 0),
            "rollback_history": getattr(self, 'rollback_history', {}),
            "gate_relaxation_epochs": getattr(self, 'gate_relaxation_epochs', 0),
            "sota_resolution": getattr(self, 'sota_resolution', None),
            "breakout_lock": getattr(self, 'breakout_lock', 0),
            "rank_weight": getattr(self, 'current_rank_weight', 0.8),
            "rank_margin": getattr(self, 'current_rank_margin', 0.10)
        }

    def load_state(self, state, preserve_curriculum=False):
        if not state: return
        if not preserve_curriculum:
            self.current_fraction = state.get("sample_fraction", self.current_fraction)
            raw_res = state.get("input_size", self.current_res)
            self.current_res = raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res
            
            # --- 2026 Resilience: Dynamic Resolution Ladder Sync ---
            if self.current_res not in self.res_ladder:
                self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
        else:
            # We are rolling back; extract the SOTA resolution at which the best checkpoint was saved
            raw_res = state.get("input_size", self.current_res)
            self.sota_resolution = raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res

        self.current_temp = max(self.min_temp, state.get("softmax_temp", self.current_temp))
        if self.task_type == "quality": self.current_temp = min(1.0, self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        if "rank_weight" in state: self.current_rank_weight = float(state["rank_weight"])
        if "rank_margin" in state: self.current_rank_margin = float(state["rank_margin"])
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)
        self.failure_log = state.get("failure_log", {}) # CRITICAL: Load memory
        self.history = state.get("history", []) # CRITICAL: Load plateau memory
        self.thermal_floor = state.get("thermal_floor", {}) # New
        self.cooldown_remaining = state.get("cooldown_remaining", 0) # New
        self.last_jolt_epoch = state.get("last_jolt_epoch", -10)
        self.spatial_lock_remaining = state.get("spatial_lock_remaining", 0) # v15.9
        self.last_res_jump_epoch = state.get("last_res_jump_epoch", 0) # v15.9
        self.epoch_count = state.get("epoch_count", self.epoch_count)
        self.best_quality = state.get("best_quality", self.best_quality)
        self.current_stress = state.get("stress", 0.0)
        self.max_stress_stuck_epochs = state.get("max_stress_stuck_epochs", 0) # v16
        
        # Restore loop-breaker states
        self.consecutive_rollbacks = state.get("consecutive_rollbacks", 0)
        raw_history = state.get("rollback_history", {})
        self.rollback_history = {int(k): v for k, v in raw_history.items()}
        self.gate_relaxation_epochs = state.get("gate_relaxation_epochs", 0)
        self.sota_resolution = state.get("sota_resolution", self.sota_resolution)
        self.breakout_lock = state.get("breakout_lock", 0)

    def recoil(self):
        """Emergency Tactical Retreat triggered by hardware or manifold failure."""
        # 2026 NPP: Keep the data fraction constant on same resolution to avoid loops.
        self.current_temp = min(1.5, self.current_temp * 1.3)
        if self.task_type == "quality": self.current_temp = min(1.0, self.current_temp)
        self.stabilization_epochs = 3
        return f"[NPP] RECOIL: Retaining data fraction at {self.current_fraction*100:.0f}% | Temp Heatup {self.current_temp:.2f}"

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

    def register_rollback(self):
        """Registers a SOTA rollback event and checks for loop conditions."""
        if not getattr(self, 'loop_breaker_enabled', True):
            return

        self.consecutive_rollbacks += 1
        res_key = self.current_res
        self.rollback_history[res_key] = self.rollback_history.get(res_key, 0) + 1
        
        # Check if loop threshold is met
        threshold = getattr(self, 'loop_breaker_threshold', 2)
        if self.rollback_history.get(res_key, 0) >= threshold:
            strategy = getattr(self, 'loop_breaker_strategy', 'auto')
            if strategy == "auto":
                # For quality models (PLCC/SRCC evaluation), resolution alignment is critical
                strategy = "escalate" if self.task_type == "quality" else "relax"
                
            current_idx = self.res_ladder.index(self.current_res) if (hasattr(self, 'res_ladder') and self.current_res in self.res_ladder) else -1
            has_higher_res = (current_idx >= 0 and current_idx < len(self.res_ladder) - 1)
            target_res = self.sota_resolution if (self.sota_resolution is not None and self.sota_resolution > self.current_res) else (self.res_ladder[current_idx + 1] if has_higher_res else None)

            if strategy == "escalate" and target_res is not None:
                old_res = self.current_res
                self.current_res = target_res
                if self.current_res not in self.res_ladder:
                    self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
                    
                self.rollback_history[self.current_res] = 0
                self.consecutive_rollbacks = 0
                self.gate_relaxation_epochs = 0
                self.current_fraction = 0.15 # Reset sample fraction for warmup on higher rung
                
                # Active lock to prevent the Governor from retreating back to old_res immediately
                self.breakout_lock = 8 
                
                print(f"\n================================================================================")
                print(f" [BREAKOUT] [GOVERNOR] Resolution-Regression Lock detected at {old_res}px!")
                print(f"   -> Automatically promoting training resolution: {old_res}px -> {self.current_res}px.")
                print(f"   -> Breakout retreat protection active for next 8 epochs.")
                print(f"================================================================================\n")
                
                self.reset_best()
                
            elif strategy in ["escalate", "relax"]:
                # Strategy B: Dynamic Gate Relaxation
                self.gate_relaxation_epochs = 6
                self.consecutive_rollbacks = 0
                print(f"\n================================================================================")
                print(f" [BREAKOUT] [GOVERNOR] Stagnation Rollback Lock detected at {res_key}px!")
                print(f"   -> Activating Dynamic Gate Relaxation for next 6 epochs to allow weights to settle.")
                print(f"================================================================================\n")

    def get_active_drift_gate(self, config_gate):
        """Returns the active drift gate, relaxing it if we are recovering from a loop."""
        if self.gate_relaxation_epochs > 0:
            self.gate_relaxation_epochs -= 1
            # Relax the gate significantly to prevent immediate rollbacks
            return min(0.80, config_gate * 0.85)
        return config_gate

    def get_active_regression_limit(self, config_limit):
        """Returns active regression limit, relaxing it during loop recovery so resolution jump can fire."""
        if self.gate_relaxation_epochs > 0 or self.rollback_history.get(self.current_res, 0) > 0:
            return max(config_limit, self.plateau_patience + 2)
        return config_limit

def export_webgpu_onnx(model, save_path, dummy_input_shape=(1, 3, 512, 512)):
    """
    Memory-Sentinel WebGPU Zero-Copy Exporter (LemGendary Cloud Link v17).
    Forces Opset 17 and fixed shapes (dynamic_axes=None) to ensure browser stability.
    Bypasses standard ONNX Slice errors for Transformers/GANs.
    """
    import torch
    import io
    from contextlib import redirect_stdout, redirect_stderr
    print(f" [MEMORY-SENTINEL] Exporting zero-copy WebGPU sharing payload to {save_path}...")
    
    # 2026 Resilience: DataParallel unwrapping guard
    model_to_export = model.module if hasattr(model, 'module') else model
    model_to_export.eval()
    
    try:
        device = next(model_to_export.parameters()).device
    except StopIteration:
        device = 'cpu'
    dummy_input = torch.randn(dummy_input_shape, device=device)
    
    try:
        # Suppress PyTorch's internal prints that contain emojis (✅) causing UnicodeEncodeError on Windows
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            torch.onnx.export(
                model_to_export,
                (dummy_input,),
                save_path,
                export_params=True,
                opset_version=17, # WebGPU Stability Target (updated from 15 based on torch requirements)
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=None # CRITICAL: Fixed shape 512x512 tile to prevent WebGPU Slice crashes
            )
        print(f" [MEMORY-SENTINEL] WebGPU ONNX export successful! Opset: 17, Shape: {dummy_input_shape}")
        return True
    except Exception as e:
        print(f" [MEMORY-SENTINEL] WebGPU export failed: {e}")
        return False
