import os
import torch
import math
import numpy as np

class SmartTrainingGovernor:
    """
    2026 Universal Autonomous Optimization Engine.
    
    A backbone-agnostic curriculum manager that treats models as high-dimensional
    manifolds. It masterfully balances Data Breadth (Fraction) before Spatial Depth (Resolution),
    proactively managing VRAM via Gradient Accumulation and LR Scaling.
    
    v6.0: Refactored for Universal Curriculum (Data-First) and Quadratic VRAM Scaling.
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
        self.min_delta = opt.get("min_delta", 2e-3) # Failsafe default
        self.stabilization_epochs = 0
        self.consecutive_drift = 0
        self.stagnation_counter = 0
        
        # --- Hardware Sentinel (Backbone-Agnostic) ---
        raw_batch = model_info.get("batch_size", "auto")
        self.target_effective_batch = opt.get("target_effective_batch", 24)
        self.current_batch = 16 if raw_batch == "auto" else int(raw_batch)
        self.current_acc = 1
        self.vram_safety_margin = 0.88 # Aggressive but safe 88% ceiling
        
        # --- Spatial State ---
        raw_size = model_info.get("input_size", 224)
        self.current_res = raw_size[1] if isinstance(raw_size, list) else raw_size
        if self.current_res not in self.res_ladder:
            self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
            
        # --- Thermal & Numerical State ---
        self.stab = stabilizers or {}
        self.task_type = model_info.get("dataset_type", "quality")
        if isinstance(self.task_type, list): self.task_type = self.task_type[0]
        self.min_temp = 0.4 if self.task_type == "quality" else 0.1
        self.current_temp = self.stab.get("softmax_temp", self.min_temp)
        self.current_clamp = self.stab.get("logit_clamp", 15.0)
        self.max_clamp = 45.0
        
        self.prev_quality = 0.0
        self.lr_multiplier = 1.0
        self.current_strategy = "Building Foundation"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None):
        if not self.enabled: return False, False, False, False, False, False, ""
        
        # 1. Improvement Logic (Entropy-Aware)
        improvement = current_quality - self.prev_quality
        quality_improved = improvement > self.min_delta
        
        # Stabilization Shield: Don't trigger changes while the manifold is "cooling"
        if self.stabilization_epochs > 0:
            self.stabilization_epochs -= 1
            self.prev_quality = current_quality
            return False, False, False, False, False, False, "📡 Manifold Stabilizing..."

        f_changed = r_changed = lr_changed = t_changed = c_changed = b_changed = False
        self.lr_multiplier = 1.0
        msg_parts = []

        # 2. Regression Handling (Instant & Tiered)
        if not quality_improved and current_quality < self.prev_quality:
            self.consecutive_drift += 1
            
            # --- INSTANT VELOCITY DAMPING (User Request) ---
            # Micro-adjust on EVERY regression to maintain forward momentum
            self.lr_multiplier = 0.98 
            lr_changed = True
            self.current_clamp = max(self.min_clamp, self.current_clamp - 0.5)
            c_changed = True
            msg_parts.append(f"MICRO-ADJUST: Instant LR Cool (0.98x) | Clamp Guide (-0.5)")

            if self.consecutive_drift >= 3:
                self.lr_multiplier = 0.7 # Tiered Cooling
                msg_parts.append(f"RECOVERY: Hard Cooling 0.7x due to sustained drift")
                self.consecutive_drift = 0
        else:
            self.consecutive_drift = 0

        # 3. Dynamic Meta-Patience Curriculum
        # We scale patience based on manifold complexity:
        # - Higher resolution = Higher patience (heavy manifolds take longer to seat)
        # - Higher data fraction = Higher patience (foundation must be rock solid)
        res_factor = max(1.0, self.current_res / self.res_ladder[0])
        frac_factor = 1.5 if self.current_fraction >= 1.0 else 1.0
        effective_patience = max(3, int(self.plateau_patience * res_factor * frac_factor))
        
        is_stagnant = epochs_no_improve >= effective_patience
        
        if is_stagnant:
            self.stagnation_counter += 1
            # --- STAGE 1: MASTER THE DATA ---
            if self.current_fraction < 1.0:
                old_frac = self.current_fraction
                self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
                f_changed = True
                msg_parts.append(f"FOUNDATION: Expanding Data {old_frac*100:.0f}% -> {self.current_fraction*100:.0f}%")
                self.stabilization_epochs = 2
            
            # --- STAGE 2: MASTER THE PIXELS ---
            else:
                current_idx = self.res_ladder.index(self.current_res)
                if current_idx < len(self.res_ladder) - 1:
                    next_res = self.res_ladder[current_idx + 1]
                    
                    # QUADRATIC VRAM SCALING (Universal Law)
                    res_ratio = next_res / self.current_res
                    vram_growth_factor = res_ratio ** 2.2 
                    
                    # Proactively drop batch size
                    self.current_batch = max(1, int(self.current_batch / vram_growth_factor))
                    self.current_acc = max(1, self.target_effective_batch // self.current_batch)
                    
                    self.current_res = next_res
                    r_changed = True
                    b_changed = True
                    
                    # --- SAWTOOTH RESET (User Request) ---
                    # Reset data variety to 50% to allow fast adaptation to new pixels
                    old_frac = self.current_fraction
                    self.current_fraction = 0.5
                    f_changed = True
                    
                    # Thermal Relief
                    self.current_temp = min(0.6, self.current_temp * 1.2)
                    t_changed = True
                    
                    msg_parts.append(f"SAWTOOTH SHIFT: Resolution to {next_res}px | Data Reset {old_frac*100:.0f}%->50% | Batch {self.current_batch}")
                    self.stabilization_epochs = 4
                
                # --- STAGE 3: FINE-TUNE PRECISION ---
                else:
                    self.lr_multiplier = 0.5 # Deep Cooling
                    lr_changed = True
                    self.current_temp = max(self.min_temp, self.current_temp * self.cooling_factor)
                    t_changed = True
                    self.current_clamp = min(self.max_clamp, self.current_clamp + 5.0)
                    c_changed = True
                    msg_parts.append("REFINEMENT: Max Curriculum reached. Deep Cooling & Precision Clamp enabled.")
                    self.current_strategy = "Precision Tuning"

        self.prev_quality = current_quality
        final_msg = "⚡ [GOVERNOR] " + " | ".join(msg_parts) if msg_parts else ""
        if not final_msg:
            status = "STABLE" if regression_epochs == 0 else "REGRESSING" 
            patience_left = effective_patience - epochs_no_improve
            print(f"📡 [GOVERNOR] Scanning Manifold... [Status: {status}] [Patience: {patience_left}/{effective_patience}]")
            
        return f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, final_msg

    def get_dynamic_save_interval(self, avg_iter_time, total_iters):
        """
        2026 Resiliency: Targets a 15-minute 'Safety Window' for progress persistence.
        """
        if avg_iter_time <= 0: return 0.2 
        epoch_duration_mins = (avg_iter_time * total_iters) / 60
        if epoch_duration_mins < 15:
            return 0.0 
        target_pct = 15 / epoch_duration_mins
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
            "current_strategy": self.current_strategy,
            "stabilization_epochs": self.stabilization_epochs
        }

    def load_state(self, state):
        if not state: return
        self.current_fraction = state.get("sample_fraction", self.current_fraction)
        self.current_res = state.get("input_size", self.current_res)
        self.current_temp = state.get("softmax_temp", self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.current_strategy = state.get("current_strategy", self.current_strategy)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)

    def recoil(self):
        """Universal Tactical Retreat."""
        self.current_fraction = max(0.15, self.current_fraction - 0.2)
        self.current_temp = min(0.6, self.current_temp * 1.5)
        self.stabilization_epochs = 3
        return "⚡ [GOVERNOR] RECOIL: Retreating to 15% Data for stabilization."
