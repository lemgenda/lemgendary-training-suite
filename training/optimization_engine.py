import os
import torch

class SmartTrainingGovernor:
    """
    2026 Autonomous Optimization Engine.
    Detects metric plateaus and regressions to dynamically scale
    complexity (Resolution) and variety (Dataset Fraction), while
    managing Thermal (Temperature), Numerical (Clamping), Hardware (Batch/VRAM), 
    and Velocity (LR) states.
    
    Now includes 'Memory-Sentinel' logic for hardware-aware auto-batching.
    """
    def __init__(self, model_info, stabilizers=None):
        self.model_info = model_info
        
        # 2026 Configurable Optimization Block
        opt = model_info.get("optimization", {})
        self.enabled = opt.get("enabled", True)
        
        # Default to 10% (0.1) for the initial "Discovery Phase" if not specified
        self.current_fraction = opt.get("initial_fraction", model_info.get("sample_fraction", 0.1))
        self.fraction_increment = opt.get("fraction_increment", 0.15)
        self.plateau_patience = opt.get("plateau_patience", 6)
        self.cooling_factor = opt.get("cooling_factor", 0.8)
        self.plateau_priority = opt.get("plateau_priority", "resolution") # 'resolution' or 'data'
        
        # 2026 Smart Clamp range [min, max]
        clamp_range = opt.get("clamp_range", [15.0, 45.0])
        self.min_clamp = float(clamp_range[0])
        self.max_clamp = float(clamp_range[1])

        # Hardware Sentinel State
        raw_batch = model_info.get("batch_size", 16)
        self.initial_batch = 16 if raw_batch == "auto" else int(raw_batch)
        self.current_batch = self.initial_batch
        self.current_acc = int(model_info.get("accumulation_steps", 1))
        self.effective_batch_goal = self.current_batch * self.current_acc
        self.vram_safety_margin = 0.85 # Trigger scaling if >85% VRAM used

        # Hyperparameters (Thermal, Velocity, Clamping)
        self.stab = stabilizers or {}
        self.current_temp = self.stab.get("softmax_temp", 0.1)
        self.current_clamp = self.stab.get("logit_clamp", 20.0)
        self.lr_multiplier = 1.0
        self.prev_quality = 0.0
        self.consecutive_drift = 0
        
        # Resolution Ladder: Progression steps to force feature discovery
        self.res_ladder = opt.get("res_ladder", [128, 256, 384, 512])
        raw_size = model_info.get("input_size", 256)
        if isinstance(raw_size, list):
            self.current_res = raw_size[1] if len(raw_size)==3 else raw_size[0]
        else:
            self.current_res = raw_size
            
        # Ensure current_res is in the ladder or at least a starting point
        if self.current_res not in self.res_ladder:
            self.res_ladder.append(self.current_res)
            self.res_ladder.sort()

        self.manifold_shift_pending = False
        self.current_strategy = f"Efficiency ({self.current_fraction*100:.0f}% Data)"

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs):
        """
        Analyzes the trajectory of the manifold to prescribe optimizations.
        Returns: (f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, msg)
        """
        if not self.enabled:
            return False, False, False, False, False, False, ""

        f_changed = False
        r_changed = False
        lr_changed = False
        t_changed = False
        c_changed = False
        b_changed = False
        self.lr_multiplier = 1.0 # Reset multiplier each audit; train.py applies it to active optimizer
        msg_parts = []

        # 1. Hardware-Aware Scaling (Memory Sentinel) - Check before shifts
        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(0).total_memory
            mem_reserved = torch.cuda.memory_reserved(0)
            mem_usage_ratio = mem_reserved / mem_total
            
            # Predictive Scaling: If resolution is about to increase, 
            # assume memory will increase by roughly (NewRes / OldRes)^2
            predicted_ratio = mem_usage_ratio
            if epochs_no_improve >= self.plateau_patience:
                current_idx = -1
                try: current_idx = self.res_ladder.index(self.current_res)
                except ValueError: pass
                if current_idx != -1 and current_idx < len(self.res_ladder) - 1:
                    next_res = self.res_ladder[current_idx + 1]
                    predicted_ratio *= (next_res / self.current_res)**1.8 # Conservative squared scaling

            # If predicted ratio > safety limit, or current ratio is already high
            if predicted_ratio > self.vram_safety_margin and self.current_batch > 1:
                old_batch = self.current_batch
                self.current_batch = max(1, self.current_batch // 2)
                self.current_acc = self.effective_batch_goal // self.current_batch
                if self.current_batch != old_batch:
                    b_changed = True
                    msg_parts.append(f"PROTECTING VRAM: Scaling Batch {old_batch}->{self.current_batch} | Accumulation 1->{self.current_acc}")

        # 2. Velocity & Numerical Management (Drift Sentinel)
        if current_quality < (self.prev_quality - 1e-6):
            self.consecutive_drift += 1
            if self.consecutive_drift >= 2:
                self.lr_multiplier = 0.7
                lr_changed = True
                
                # Tighten the logit clamp to force stability during regression
                old_clamp = self.current_clamp
                self.current_clamp = max(self.min_clamp, self.current_clamp - 5.0)
                if self.current_clamp != old_clamp:
                    c_changed = True
                    msg_parts.append(f"TIGHTENING CLAMP to {self.current_clamp:.1f} (Resilience)")
                
                msg_parts.append(f"DAMPENING LR (0.7x) due to consecutive drift")
                self.consecutive_drift = 0
        else:
            self.consecutive_drift = 0

        # 3. Dataset Expansion on Regression
        if regression_epochs >= 1 and self.current_fraction < 1.0:
            self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
            f_changed = True
            msg_parts.append(f"EXPANDING VARIETY (+{self.fraction_increment*100:.0f}%) to {self.current_fraction*100:.0f}%")

        # 4. Resolution Scaling & Thermal Cooling on Plateau
        if epochs_no_improve >= self.plateau_patience:
            if self.plateau_priority == "data" and self.current_fraction < 1.0:
                # Perceptual/Texture Priority: Expand variety before scaling complexity
                self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
                f_changed = True
                msg_parts.append(f"PLATEAU (Data-Priority): Expanding variety to {self.current_fraction*100:.0f}%")
            else:
                # Fidelity/Architecture Priority: Shift Resolution
                current_idx = -1
                try: current_idx = self.res_ladder.index(self.current_res)
                except ValueError: pass
                
                if current_idx != -1 and current_idx < len(self.res_ladder) - 1:
                    # Final Hardware Guard: If we are already at batch 1 and predicted OOM, cap resolution.
                    if predicted_ratio > self.vram_safety_margin and self.current_batch == 1:
                        msg_parts.append(f"RESOLUTION CAPPED: VRAM limited at {self.current_res}px")
                    else:
                        self.current_res = self.res_ladder[current_idx + 1]
                        r_changed = True
                        
                        # Thermal Cooling & Clamp Relaxation
                        self.current_temp = max(0.05, self.current_temp * self.cooling_factor)
                        t_changed = True
                        
                        old_clamp = self.current_clamp
                        self.current_clamp = min(self.max_clamp, self.current_clamp + 5.0)
                        if self.current_clamp != old_clamp:
                            c_changed = True
                        
                        msg_parts.append(f"SCALING to {self.current_res}px | RELAXING CLAMP to {self.current_clamp:.1f}")
                elif self.current_fraction < 1.0:
                    # Resolution maxed, fallback to variety expansion
                    self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
                    f_changed = True
                    msg_parts.append(f"PLATEAU (Res-Maxed): Expanding variety to {self.current_fraction*100:.0f}%")
                else:
                    # Everything maxed, try a 2.0x LR Jolt to break the plateau
                    self.lr_multiplier = 2.0
                    lr_changed = True
                    msg_parts.append(f"PLATEAU BREAKER: Injecting 2.0x LR Jolt")

        self.prev_quality = current_quality
        
        final_msg = ""
        if msg_parts:
            self.manifold_shift_pending = True
            final_msg = f"⚡ [SMART GOVERNOR] " + " | ".join(msg_parts)
            
        return f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, final_msg

    def get_state(self):
        """Returns the updated parameters for re-initialization."""
        raw_size = self.model_info.get("input_size", 256)
        if isinstance(raw_size, list) and len(raw_size) == 3:
            new_size = [raw_size[0], self.current_res, self.current_res]
        else:
            new_size = self.current_res
            
        return {
            "sample_fraction": self.current_fraction,
            "input_size": new_size,
            "softmax_temp": self.current_temp,
            "logit_clamp": self.current_clamp,
            "lr_multiplier": self.lr_multiplier,
            "batch_size": self.current_batch,
            "accumulation_steps": self.current_acc
        }
