import os
import torch
import math
import numpy as np

class SmartTrainingGovernor:
    """
    2026 Universal Autonomous Optimization Engine.
    
    v8.0 Nuclear Propulsion: 
    - Data Starvation Detection (Prevents regression at low fractions)
    - Loss-Gradient Priority (Momentum-aware escalation)
    - Proactive Spatial Scaling (Resolution jumps for high-acc plateaus)
    - Dynamic LR Warping
    """
    def __init__(self, model_info, stabilizers=None):
        self.model_info = model_info
        opt = model_info.get("optimization", {})
        self.enabled = opt.get("enabled", True)
        
        # --- Curriculum Config ---
        self.current_fraction = opt.get("initial_fraction", 0.1)
        self.fraction_increment = opt.get("fraction_increment", 0.15)
        self.res_ladder = opt.get("res_ladder", [224, 384, 512, 640])
        self.plateau_patience = opt.get("plateau_patience", 4)
        self.cooling_factor = opt.get("cooling_factor", 0.8)
        
        # --- Stabilization Constants ---
        self.min_delta = opt.get("min_delta", 1e-4) # Higher sensitivity for SOTA
        self.stabilization_epochs = 0
        self.consecutive_drift = 0
        self.stagnation_counter = 0
        self.starvation_counter = 0
        
        # --- Hardware Sentinel ---
        raw_batch = model_info.get("batch_size", "auto")
        self.target_effective_batch = opt.get("target_effective_batch", 24)
        self.current_batch = 16 if raw_batch == "auto" else int(raw_batch)
        self.current_acc = 1
        self.vram_safety_margin = 0.88 
        
        # --- Spatial State ---
        raw_size = model_info.get("input_size", 224)
        self.current_res = raw_size[1] if isinstance(raw_size, list) else raw_size
        if self.current_res not in self.res_ladder:
            self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
            
        # --- Thermal & Numerical State ---
        self.stab = stabilizers or {}
        self.task_type = model_info.get("dataset_type", "quality")
        if isinstance(self.task_type, list): self.task_type = self.task_type[0]
        self.min_temp = 0.5 if self.task_type == "quality" else 0.1
        self.current_temp = self.stab.get("softmax_temp", self.min_temp)
        self.current_clamp = self.stab.get("logit_clamp", 15.0)
        
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.lr_multiplier = 1.0
        self.current_strategy = "Nuclear Propulsion"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None, current_loss=None):
        if not self.enabled: return False, False, False, False, False, False, ""
        
        # 1. Manifold Physics Analysis
        improvement = current_quality - self.prev_quality
        significant_improvement = current_quality > (best_quality + self.min_delta)
        loss_improving = (current_loss < self.prev_loss) if current_loss is not None else True
        
        if self.stabilization_epochs > 0:
            self.stabilization_epochs -= 1
            self.prev_quality = current_quality
            if current_loss: self.prev_loss = current_loss
            return False, False, False, False, False, False, "📡 Manifold Cooling..."

        f_changed = r_changed = lr_changed = t_changed = c_changed = b_changed = False
        self.lr_multiplier = 1.0
        msg_parts = []

        # 2. Velocity Calculation
        effective_patience = max(3, self.plateau_patience)
        if significant_improvement:
            self.stagnation_counter = 0
            self.starvation_counter = 0
        else:
            self.stagnation_counter += 1

        # 3. STARVATION DETECTION (Critical v8.0)
        # If we are at a low fraction and quality is dropping but loss is healthy, the model is starving.
        is_starving = False
        if self.current_fraction < 0.15 and not significant_improvement and not loss_improving:
            self.starvation_counter += 1
            if self.starvation_counter >= 2:
                is_starving = True
                msg_parts.append("STARVATION DETECTED: Model is forgetting distribution at low fraction.")

        # 4. PROACTIVE RESOLUTION JUMP (v8.0)
        # If accuracy is very high but flat, don't wait for data mastery—jump resolution.
        high_fidelity_plateau = (current_quality > 0.94 and self.stagnation_counter >= 3)
        
        # 5. Decision Matrix
        if is_starving or self.stagnation_counter >= effective_patience or high_fidelity_plateau:
            
            # --- CURRICULUM ESCALATION ---
            if self.current_fraction < 1.0 and not high_fidelity_plateau:
                old_frac = self.current_fraction
                # Force aggressive jump if starving
                jump = self.fraction_increment * (2 if is_starving else 1)
                self.current_fraction = min(1.0, self.current_fraction + jump)
                f_changed = True
                reason = "STARVATION" if is_starving else "STAGNATION"
                msg_parts.append(f"PROPULSION: {reason} | Escalating Data to {self.current_fraction*100:.0f}%")
                self.stabilization_epochs = 2
                self.starvation_counter = 0
            
            # --- SPATIAL ESCALATION ---
            else:
                current_idx = self.res_ladder.index(self.current_res)
                if current_idx < len(self.res_ladder) - 1:
                    next_res = self.res_ladder[current_idx + 1]
                    res_ratio = next_res / self.current_res
                    vram_growth = res_ratio ** 2.2 
                    
                    self.current_batch = max(1, int(self.current_batch / vram_growth))
                    self.current_acc = max(1, self.target_effective_batch // self.current_batch)
                    self.current_res = next_res
                    r_changed = b_changed = True
                    
                    # Reset curriculum for new spatial depth
                    self.current_fraction = 0.5
                    f_changed = True
                    
                    # Thermal Excitation
                    self.current_temp = min(1.5, self.current_temp * 1.3)
                    t_changed = True
                    
                    reason = "HIGH-FIDELITY PLATEAU" if high_fidelity_plateau else "STAGNATION"
                    msg_parts.append(f"AUTONOMOUS JUMP: {reason} | Resolution -> {next_res}px | Temp -> {self.current_temp:.2f}")
                    self.stabilization_epochs = 4
                else:
                    # Final Precision Phase
                    self.lr_multiplier = 0.5
                    lr_changed = True
                    msg_parts.append("REFINEMENT: SOTA Boundary reached. Deep Cooling enabled.")

        # 6. Drift Correction (Damping)
        elif not significant_improvement and not loss_improving:
            self.consecutive_drift += 1
            if self.consecutive_drift >= 3:
                self.lr_multiplier = 0.8
                lr_changed = True
                msg_parts.append("DRIFT: Micro-cooling 0.8x to stabilize manifold")
                self.consecutive_drift = 0
        else:
            self.consecutive_drift = 0

        self.prev_quality = current_quality
        if current_loss: self.prev_loss = current_loss
        
        final_msg = "🚀 [GOVERNOR] " + " | ".join(msg_parts) if msg_parts else ""
        if not final_msg:
            patience_left = effective_patience - self.stagnation_counter
            print(f"📡 [GOVERNOR] Monitoring Manifold... [Acc: {current_quality:.4f}] [Patience: {patience_left}/{effective_patience}]")
            
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
