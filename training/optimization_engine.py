import os
import torch

class SmartTrainingGovernor:
    """
    2026 Autonomous Optimization Engine.
    Detects metric plateaus and regressions to dynamically scale
    complexity (Resolution) and variety (Dataset Fraction), while
    managing Thermal (Temperature), Numerical (Clamping), Hardware (Batch/VRAM), 
    and Velocity (LR) states.
    
    v5.1: Added Jolt Recoil protection and configurable jolt multipliers.
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
        self.jolt_multiplier = opt.get("jolt_multiplier", 2.0)
        
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
        self.jolt_active = False # Tracks if we just injected energy
        self.stagnation_counter = 0 # Tracks horizontal stagnation across epochs
        self.min_delta = opt.get("min_delta", 1e-3) # 2026: 0.1% required to reset plateau
        
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

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None):
        # 2026 Resilience: Stricter Plateau Detection
        # horizontal_stagnation occurs if quality doesn't improve meaningfully, regardless of loss.
        quality_improved = current_quality > (self.prev_quality * (1.0 + self.min_delta))
        is_stagnant = epochs_no_improve >= self.plateau_patience or (not quality_improved and epochs_no_improve >= self.plateau_patience // 2)
        
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

        # Predict VRAM scaling only if resolution shift is possible
        predicted_ratio = 1.0
        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(0).total_memory
            mem_reserved = torch.cuda.memory_reserved(0)
            mem_usage_ratio = mem_reserved / mem_total
            predicted_ratio = mem_usage_ratio

        # 1. Velocity & Numerical Management (Drift Sentinel)
        # If the model regresses for 2 consecutive epochs, dampen the LR and tighten the clamp.
        if current_quality < (self.prev_quality - 1e-6):
            self.consecutive_drift += 1
            
            # Post-Jolt Regression: Model couldn't handle the power, throttle back immediately.
            if self.jolt_active:
                self.lr_multiplier = 0.5
                lr_changed = True
                msg_parts.append(f"JOLT RECOIL: Immediate 0.5x LR damping to stabilize manifold")
                self.jolt_active = False
            
            if self.consecutive_drift >= 2:
                self.lr_multiplier = min(self.lr_multiplier, 0.7)
                lr_changed = True
                
                # Tighten the logit clamp to force stability during regression
                old_clamp = self.current_clamp
                self.current_clamp = max(self.min_clamp, self.current_clamp - 5.0)
                if self.current_clamp != old_clamp:
                    c_changed = True
                    msg_parts.append(f"TIGHTENING CLAMP to {self.current_clamp:.1f} (Resilience)")
                
                msg_parts.append(f"DAMPENING LR ({self.lr_multiplier}x) due to consecutive drift")
                self.consecutive_drift = 0
        else:
            self.consecutive_drift = 0
            self.jolt_active = False # Safe back in the fold

        # 1.1 Numerical Sentinel Feedback Loop (v5.6)
        # If sentinel_trigger_rate > 5%, it means the model is fighting its boundaries.
        if sentinel_trigger_rate > 0.05:
            self.lr_multiplier = min(self.lr_multiplier, 0.8)
            lr_changed = True
            msg_parts.append(f"NUMERICAL STRESS ({sentinel_trigger_rate*100:.1f}%): Cooling LR 0.8x to seat manifold")
            # If stress is extreme (>20%), force resolution scaling pause by inflating patiente
            if sentinel_trigger_rate > 0.20:
                epochs_no_improve = 0 # Trick the engine into thinking we just improved to delay scaling
                msg_parts.append(f"EXTREME STRESS: Scaling Milestones Delayed")

        # 2. Hardware-Aware Scaling (Memory Sentinel) - Check before shifts
        if torch.cuda.is_available() and predicted_ratio > 0.0:
            if is_stagnant:
                current_idx = -1
                try: current_idx = self.res_ladder.index(self.current_res)
                except ValueError: pass
                if current_idx != -1 and current_idx < len(self.res_ladder) - 1:
                    next_res = self.res_ladder[current_idx + 1]
                    predicted_ratio *= (next_res / self.current_res)**1.8 # Conservative squared scaling

            if predicted_ratio > self.vram_safety_margin and self.current_batch > 1:
                old_batch = self.current_batch
                self.current_batch = max(1, self.current_batch // 2)
                self.current_acc = self.effective_batch_goal // self.current_batch
                if self.current_batch != old_batch:
                    b_changed = True
                    msg_parts.append(f"PROTECTING VRAM: Scaling Batch {old_batch}->{self.current_batch} | Accumulation 1->{self.current_acc}")

        # 3. Dataset Expansion on Regression (Incremental Discovery)
        if regression_epochs >= 1 and self.current_fraction < 1.0 and not f_changed:
            self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
            f_changed = True
            msg_parts.append(f"EXPANDING VARIETY (+{self.fraction_increment*100:.0f}%) to {self.current_fraction*100:.0f}%")

        # 4. Resolution Scaling & Thermal Cooling on Plateau
        if is_stagnant:
            self.stagnation_counter += 1
            msg_parts.append(f"STAGNATION ({self.stagnation_counter})")
            if self.plateau_priority == "data" and self.current_fraction < 1.0:
                self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
                f_changed = True
                msg_parts.append(f"PLATEAU (Data-Priority): Expanding variety to {self.current_fraction*100:.0f}%")
            else:
                current_idx = -1
                try: current_idx = self.res_ladder.index(self.current_res)
                except ValueError: pass
                
                if current_idx != -1 and current_idx < len(self.res_ladder) - 1:
                    if predicted_ratio > self.vram_safety_margin and self.current_batch == 1:
                        msg_parts.append(f"RESOLUTION CAPPED: VRAM limited at {self.current_res}px")
                    else:
                        self.current_res = self.res_ladder[current_idx + 1]
                        r_changed = True
                        self.current_temp = max(0.05, self.current_temp * self.cooling_factor)
                        t_changed = True
                        
                        old_clamp = self.current_clamp
                        self.current_clamp = min(self.max_clamp, self.current_clamp + 5.0)
                        if self.current_clamp != old_clamp:
                            c_changed = True
                        
                        msg_parts.append(f"SCALING to {self.current_res}px | RELAXING CLAMP to {self.current_clamp:.1f}")
                elif self.current_fraction < 1.0:
                    self.current_fraction = min(1.0, self.current_fraction + self.fraction_increment)
                    f_changed = True
                    msg_parts.append(f"PLATEAU (Res-Maxed): Expanding variety to {self.current_fraction*100:.0f}%")
                else:
                    # Everything maxed, force a High-Energy Manifold Jolt (v6.1.17)
                    if current_lr and base_lr:
                        self.lr_multiplier = base_lr / current_lr
                        msg_parts.append(f"PLATEAU BREAKER: Injecting High-Energy Manifold Jolt ({self.lr_multiplier:.1f}x)")
                    else:
                        self.lr_multiplier = self.jolt_multiplier
                        msg_parts.append(f"PLATEAU BREAKER: Injecting {self.jolt_multiplier}x LR Jolt")
                    
                    lr_changed = True
                    self.jolt_active = True
                    # Warm up Thermal Thermal Shield to allow exploration
                    self.current_temp = min(0.3, self.current_temp * 2.0)
                    t_changed = True
                
                # Reset stagnation counter ONLY if a change was actually made
                if f_changed or r_changed or lr_changed:
                    self.stagnation_counter = 0

        self.prev_quality = current_quality
        
        final_msg = ""
        if msg_parts:
            self.manifold_shift_pending = True
            final_msg = f"⚡ [SMART GOVERNOR] " + " | ".join(msg_parts)
        else:
            # 2026: Visible Monitoring (Audit v5.2)
            # This ensures the user knows the engine is alive and scanning the horizon.
            status = "STABLE" if regression_epochs == 0 else "REGRESSING" 
            patience_left = self.plateau_patience - epochs_no_improve
            print(f"📡 [SMART GOVERNOR] Scanning Manifold... [Status: {status}] [Patience: {patience_left}]")
            
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
            "accumulation_steps": self.current_acc,
            "consecutive_drift": self.consecutive_drift,
            "stagnation_counter": self.stagnation_counter,
            "jolt_active": self.jolt_active,
            "current_strategy": self.current_strategy
        }

    def load_state(self, state):
        """Restores the governor state from a checkpoint."""
        if not state: return
        self.current_fraction = state.get("sample_fraction", self.current_fraction)
        
        raw_size = state.get("input_size", self.current_res)
        if isinstance(raw_size, list) and len(raw_size) == 3:
            self.current_res = raw_size[1]
        else:
            self.current_res = raw_size
            
        self.current_temp = state.get("softmax_temp", self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        self.lr_multiplier = state.get("lr_multiplier", self.lr_multiplier)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.consecutive_drift = state.get("consecutive_drift", self.consecutive_drift)
        self.stagnation_counter = state.get("stagnation_counter", self.stagnation_counter)
        self.jolt_active = state.get("jolt_active", self.jolt_active)
        self.current_strategy = state.get("current_strategy", self.current_strategy)

    def recoil(self):
        """
        Force a tactical retreat in complexity upon a technical rollback.
        Scales back resolution and variety to allow the model to re-seat safely.
        """
        msg_parts = []
        
        # 1. Step back in resolution ladder
        current_idx = -1
        try: current_idx = self.res_ladder.index(self.current_res)
        except ValueError: pass
        
        if current_idx > 0:
            old_res = self.current_res
            self.current_res = self.res_ladder[current_idx - 1]
            msg_parts.append(f"Resolution {old_res}->{self.current_res}px")
            
        # 2. Step back in dataset variety
        if self.current_fraction > 0.15: # Don't drop below bare minimum
            old_frac = self.current_fraction
            self.current_fraction = max(0.15, self.current_fraction - self.fraction_increment)
            msg_parts.append(f"Variety {old_frac*100:.0f}%->{self.current_fraction*100:.0f}%")
            
        # 3. Cool down Thermal/Clamping
        self.current_temp = min(0.15, self.current_temp * 1.5) # Warm back up to allow exploration
        self.current_clamp = max(self.min_clamp, self.current_clamp - 5.0) # Tighten clamp
        
        # 4. Reset internal counters to prevent immediate re-scaling
        self.consecutive_drift = 0
        self.jolt_active = False
        
        final_msg = ""
        if msg_parts:
            final_msg = f"⚡ [SMART GOVERNOR] RECOIL ENGAGED: " + " | ".join(msg_parts)
            self.manifold_shift_pending = True
        
        return final_msg
