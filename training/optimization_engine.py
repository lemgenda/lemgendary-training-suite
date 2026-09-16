import os
import torch
import math
import numpy as np
from training.telemetry import METRIC_DIRECTIONS, METRIC_WEIGHTS


class SmartTrainingGovernor:
    """
    2026 Universal Autonomous Optimization Engine (v15.6 Nuclear).

    Numerical Priority Protocol (NPP) Features:
    - State-Persistence Guard
    - Turbulence Dampening
    - Proportional Manifold Stride
    - Surgical State-Loop Penalties
    - Dynamic Batch Growth (v15.6)
    """

    @property
    def current_fraction(self):
        return getattr(self, '_current_fraction', 1.0)

    @current_fraction.setter
    def current_fraction(self, value):
        self._current_fraction = value

    def __init__(self, model_info, config=None, stabilizers=None):
        self.model_info = model_info
        self.config = config or {}
        opt = model_info.get("optimization", {})
        manifold_defaults = self.config.get("governor", {}).get("manifold", {})

        self.enabled = opt.get("enabled", True)

        self.res_ladder = opt.get("res_ladder")
        self.target_effective_batch = opt.get("target_effective_batch", manifold_defaults.get("target_effective_batch", 24))
        self.manifold_maturity = opt.get("manifold_maturity", manifold_defaults.get("maturity_soak", 5))
        self.plateau_patience = opt.get("plateau_patience", self.config.get("governor", {}).get("plateau_patience", 6))

        self.plateau_priority = opt.get("plateau_priority", "data")
        self.fraction_increment = opt.get("fraction_increment", manifold_defaults.get("fraction_increment", 0.15))
        self.cooling_factor = opt.get("cooling_factor", 0.5)
        self.clamp_range = opt.get("clamp_range", [15.0, 45.0])
        gov_cfg = self.config.get("governor", {})
        self.jolt_cooldown = gov_cfg.get("jolt_cooldown_epochs", 5)
        self.stabilization_lock = gov_cfg.get("stabilization_lock_epochs", 3)
        self.breakout_threshold = gov_cfg.get("emergency_breakout_threshold", 0.10)
        self.sharpening_rate = gov_cfg.get("sharpening_cooling_rate", 0.98)

        self.recovery_streak = 0

        self.current_fraction = model_info.get("sample_fraction") or opt.get("initial_fraction", manifold_defaults.get("initial_fraction", 0.15))
        self.current_batch = int(model_info.get("batch_size", 16)) if model_info.get("batch_size") and model_info.get("batch_size") != "auto" else 16
        self.current_acc = 1

        raw_size = model_info.get("input_size", 224)
        self.current_res = raw_size[1] if isinstance(raw_size, (list, tuple)) else raw_size

        if not self.res_ladder:
            stride = manifold_defaults.get("resolution_stride", 128)
            max_res = manifold_defaults.get("max_resolution", 1024)
            self.res_ladder = []
            curr = self.current_res
            while curr <= max_res:
                self.res_ladder.append(curr)
                curr += stride
            if not self.res_ladder: self.res_ladder = [self.current_res]

        self.stab = stabilizers or {}
        self.task_type = model_info.get("dataset_type", "quality")
        if isinstance(self.task_type, list): self.task_type = self.task_type[0]

        if self.task_type == "forex":
            self.plateau_patience = max(self.plateau_patience, 15)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8.0
        if self.task_type in ["restoration", "enhancement", "face", "parameter_prediction"]:
            if vram_gb < 4.5:
                max_safe_res = 256 if self.task_type == "parameter_prediction" else 384
            elif vram_gb < 8.5:
                max_safe_res = 384
            elif vram_gb < 16.5:
                max_safe_res = 512
            else:
                max_safe_res = 1024

            cfg_max_res = self.config.get("hardware", {}).get("max_allowed_resolution")
            if cfg_max_res:
                max_safe_res = min(max_safe_res, int(cfg_max_res))

            self.res_ladder = [r for r in self.res_ladder if r <= max_safe_res]
            if not self.res_ladder: self.res_ladder = [max_safe_res]
            if self.current_res is not None and self.current_res > max_safe_res:
                print(f" [GUARD] [GOVERNOR] Hardware VRAM Cap Active ({vram_gb:.1f}GB): Clamping {self.current_res}px -> {max_safe_res}px for memory stability.")
                self.current_res = max_safe_res

        if self.res_ladder and self.current_res != self.res_ladder[0] and self.task_type != "forex":
            print(f" [GUARD] [GOVERNOR] Aligning start resolution {self.current_res}px -> lowest rung {self.res_ladder[0]}px.")
            self.current_res = self.res_ladder[0]

        if self.current_res is not None and self.current_res not in self.res_ladder:
            self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
        if self.task_type == "forex":
            self.min_temp = max(0.75, float(self.stab.get("min_temp", 0.75)))
            self.current_temp = max(self.min_temp, float(self.stab.get("softmax_temp", 1.0)))
        else:
            self.min_temp = float(self.stab.get("min_temp", 0.5 if self.task_type == "quality" else (0.01 if self.task_type == "parameter_prediction" else 0.1)))
            self.current_temp = self.stab.get("softmax_temp", self.min_temp)
        self.current_clamp = self.stab.get("logit_clamp", 15.0)

        self.current_rank_weight = float(self.stab.get("rank_weight", 0.8))
        self.current_rank_margin = float(self.stab.get("rank_margin", 0.10))
        self.max_rank_weight = float(self.stab.get("max_rank_weight", 1.5))
        self.min_rank_margin = float(self.stab.get("min_rank_margin", 0.05))
        self.current_spearman_weight = float(self.stab.get("soft_spearman_weight", 0.5))

        self.current_emd_weight    = float(self.stab.get('emd_weight', 1.0))
        self.current_ssim_weight   = float(self.stab.get('ssim_weight', 0.0))
        self.current_huber_delta   = float(self.stab.get('huber_delta', 1.0))
        self.current_conf_gate_str = float(self.stab.get('conf_gate_strength', 1.0))
        self.current_lpips_weight  = float(self.stab.get('lpips_weight', 0.025))
        self.current_mag_weight    = float(self.stab.get('mag_weight', 0.5))
        self.current_dir_weight    = float(self.stab.get('dir_weight', 1.0))

        self.metric_focus_epochs_remaining = 0
        self.metric_focus_target = None
        self.metric_focus_last_fired = {}

        self.loop_breaker_enabled = opt.get("loop_breaker_enabled", True)
        self.loop_breaker_threshold = opt.get("loop_breaker_threshold", 2)
        self.loop_breaker_strategy = opt.get("loop_breaker_strategy", "auto")
        self.consecutive_rollbacks = 0
        self.rollback_history = {}
        self.gate_relaxation_epochs = 0
        self.sota_resolution = None
        self.breakout_lock = 0

        self.history = []
        self.failure_log = {}
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.best_quality = 0.0
        self.stabilization_epochs = 0
        self.cooldown_remaining = 0
        self.current_stress = 0.0
        self.thermal_floor = {}
        self.lr_multiplier = 1.0
        self.head_lr_multiplier = 1.0
        self.jolt_window_remaining = 0
        self.trigger_mini_swa = False
        self.last_action_epoch = 0
        self.epoch_count = 0
        self.session_epoch_count = 0
        self.max_stress_stuck_epochs = 0
        self.last_jolt_epoch = -10

        base_delta = opt.get("min_delta", 0.0005)
        if self.task_type == "parameter_prediction":
            self.min_delta = base_delta
        else:
            self.min_delta = base_delta if self.task_type == "quality" else (base_delta * 100.0)
        self.spatial_lock_remaining = 0
        self.last_res_jump_epoch = 0

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
                    elif k == 'max_drawdown': target_score += max(0.0, 100.0 - target_v) * weight
                    elif k in ['tp_mae', 'sl_mae']: target_score += max(0.0, 50.0 - target_v) * weight
                    elif k == 'dir_entropy': target_score += max(0.0, 1.099 - target_v) * weight
                    else: target_score += (1.0 / (target_v + 1e-6)) * weight
            if target_score > 0:
                if target_score <= 1.0 and self.task_type == "quality":
                    target_score *= 100.0
                self.target_quality_score = target_score

    # =========================================================================
    # 2026 v15.6: DYNAMIC BATCH GOVERNOR
    # =========================================================================
    def suggest_batch_growth(self, current_batch, current_acc, target_eff, vram_free_ratio):
        """
        If VRAM headroom > 40%, propose doubling physical batch (halving accumulation).
        Returns (new_batch, new_acc) or (current_batch, current_acc) if no change.
        Caller is responsible for rebuilding DataLoaders when batch changes.
        """
        if current_batch < 1:
            return current_batch, current_acc
        if vram_free_ratio < 0.40:
            return current_batch, current_acc
        new_batch = min(current_batch * 2, target_eff)
        if new_batch == current_batch:
            return current_batch, current_acc
        new_acc = max(1, target_eff // new_batch)
        return new_batch, new_acc

    def _identify_lagging_metric(self, metrics_dict):
        results = []
        for key, target in self.sota_targets.items():
            current = metrics_dict.get(key)
            if current is None or target is None or target == 0:
                continue
            is_higher_better = METRIC_DIRECTIONS.get(key, True)
            if is_higher_better:
                deficit = max(0.0, (target - current) / abs(target))
            else:
                deficit = max(0.0, (current - target) / abs(target))
            if deficit > 0:
                results.append((key, deficit))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _apply_targeted_optimizations(self, lagging_list, metrics_dict):
        msg_parts = []
        if not lagging_list:
            return msg_parts

        primary_metric, deficit = lagging_list[0]

        severity = "MILD"
        if primary_metric in ['srcc', 'plcc', 'accuracy', 'dir_acc', 'win_rate']:
            if deficit >= 0.10: severity = "CRITICAL"
            elif deficit >= 0.05: severity = "SEVERE"
        else:
            if deficit >= 0.35: severity = "CRITICAL"
            elif deficit >= 0.15: severity = "SEVERE"

        if primary_metric == 'srcc':
            if severity == "MILD":
                self.current_spearman_weight = min(2.0, round(self.current_spearman_weight + 0.15, 2))
                self.current_rank_weight = min(self.max_rank_weight, round(self.current_rank_weight + 0.10, 2))
            elif severity == "SEVERE":
                self.current_spearman_weight = min(3.0, round(self.current_spearman_weight + 0.50, 2))
                self.current_rank_weight = min(self.max_rank_weight, round(self.current_rank_weight + 0.40, 2))
                self.current_rank_margin = self.min_rank_margin
                self.current_emd_weight = max(0.1, round(self.current_emd_weight - 0.05, 2))
            elif severity == "CRITICAL":
                self.current_spearman_weight = 3.0
                self.current_rank_weight = 2.0
                self.current_rank_margin = self.min_rank_margin
                self.current_emd_weight = max(0.1, round(self.current_emd_weight - 0.05, 2))
                self.metric_focus_target = 'srcc'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'plcc':
            if severity == "MILD":
                self.current_emd_weight = min(2.0, round(self.current_emd_weight + 0.10, 2))
                self.current_rank_weight = max(0.0, round(self.current_rank_weight - 0.10, 2))
            elif severity == "SEVERE":
                self.current_emd_weight = min(2.0, round(self.current_emd_weight + 0.25, 2))
                self.current_rank_weight = max(0.0, round(self.current_rank_weight - 0.30, 2))
                self.current_spearman_weight = max(0.0, round(self.current_spearman_weight - 0.10, 2))
            elif severity == "CRITICAL":
                self.current_emd_weight = 1.5
                self.current_rank_weight = 0.3
                self.current_spearman_weight = 0.2
                self.metric_focus_target = 'plcc'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'rank_margin':
            self.current_rank_margin = max(self.min_rank_margin, round(self.current_rank_margin - (deficit * 0.05), 3))
        elif primary_metric == 'psnr':
            if severity == "MILD":
                self.current_lpips_weight = max(0.005, round(self.current_lpips_weight - 0.003, 4))
            elif severity == "SEVERE":
                self.current_lpips_weight = max(0.005, round(self.current_lpips_weight - 0.010, 4))
                self.current_ssim_weight = max(0.0, round(self.current_ssim_weight - 0.02, 3))
            elif severity == "CRITICAL":
                self.current_lpips_weight = 0.005
                self.head_lr_multiplier = min(self.lr_multiplier * 1.5, self.head_lr_multiplier * 1.3)
        elif primary_metric == 'ssim':
            if severity == "MILD":
                self.current_ssim_weight = min(0.2, round(self.current_ssim_weight + 0.02, 3))
                self.current_lpips_weight = min(0.1, round(self.current_lpips_weight + 0.002, 4))
            elif severity == "SEVERE":
                self.current_ssim_weight = min(0.2, round(self.current_ssim_weight + 0.06, 3))
                self.current_lpips_weight = min(0.1, round(self.current_lpips_weight + 0.005, 4))
            elif severity == "CRITICAL":
                self.current_ssim_weight = 0.15
                self.metric_focus_target = 'ssim'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'lpips':
            if severity == "MILD":
                self.current_lpips_weight = min(0.1, round(self.current_lpips_weight + 0.005, 4))
            elif severity == "SEVERE":
                self.current_lpips_weight = min(0.1, round(self.current_lpips_weight + 0.015, 4))
            elif severity == "CRITICAL":
                self.current_lpips_weight = 0.08
                self.metric_focus_target = 'lpips'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'fid':
            if severity == "MILD":
                self.current_stress = min(5.0, getattr(self, 'current_stress', 0.0) + 0.5)
            elif severity == "SEVERE":
                self.current_stress = min(5.0, getattr(self, 'current_stress', 0.0) + 1.0)
            elif severity == "CRITICAL":
                self.current_stress = 5.0
                self.metric_focus_target = 'fid'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'map50':
            self.head_lr_multiplier = min(self.lr_multiplier * 1.5, getattr(self, 'head_lr_multiplier', self.lr_multiplier) * 1.3)
            self.current_stress = min(5.0, getattr(self, 'current_stress', 0.0) + 0.5)
        elif primary_metric in ['map50_95', 'map_hard']:
            if primary_metric == 'map_hard' and severity == "MILD":
                pass
            else:
                self.head_lr_multiplier = min(self.lr_multiplier * 1.5, getattr(self, 'head_lr_multiplier', self.lr_multiplier) * 1.3)
        elif primary_metric in ['accuracy', 'accuracy_vqa', 'miou', 'map_medium']:
            if severity == "MILD" or primary_metric != 'accuracy':
                self.current_temp = max(self.min_temp, self.current_temp * 0.95)
                if primary_metric in ['miou', 'map_medium']:
                    self.current_stress = min(5.0, getattr(self, 'current_stress', 0.0) + 0.5)
                if primary_metric != 'accuracy':
                    self.head_lr_multiplier = min(self.lr_multiplier * 1.5, getattr(self, 'head_lr_multiplier', self.lr_multiplier) * 1.2)
            elif severity == "SEVERE":
                self.current_temp = self.min_temp
                self.head_lr_multiplier = min(self.lr_multiplier * 1.5, getattr(self, 'head_lr_multiplier', self.lr_multiplier) * 1.2)
            elif severity == "CRITICAL":
                self.current_temp = self.min_temp
                self.metric_focus_target = 'accuracy'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'mae':
            if severity == "MILD":
                self.current_huber_delta = max(0.01, self.current_huber_delta * 0.8)
            elif severity == "SEVERE":
                self.current_huber_delta = max(0.01, self.current_huber_delta * 0.5)
                self.head_lr_multiplier = min(self.lr_multiplier * 1.5, getattr(self, 'head_lr_multiplier', self.lr_multiplier) * 1.2)
            elif severity == "CRITICAL":
                self.current_huber_delta = 0.1
                self.metric_focus_target = 'mae'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'dir_acc':
            if severity == "MILD":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.10, 2))
            elif severity == "SEVERE":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.30, 2))
            elif severity == "CRITICAL":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.50, 2))
                self.metric_focus_target = 'dir_acc'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'win_rate':
            if severity == "MILD":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.10, 2))
                self.current_conf_gate_str = min(2.0, round(self.current_conf_gate_str + 0.1, 2))
            elif severity == "SEVERE":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.25, 2))
                self.current_conf_gate_str = min(2.0, round(self.current_conf_gate_str + 0.3, 2))
            elif severity == "CRITICAL":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.40, 2))
                self.metric_focus_target = 'win_rate'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'profit_factor':
            if severity == "MILD":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.10, 2))
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.05, 2))
            elif severity == "SEVERE":
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.20, 2))
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.15, 2))
            elif severity == "CRITICAL":
                self.metric_focus_target = 'profit_factor'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'sharpe_ratio':
            if severity == "MILD":
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.05, 2))
            elif severity == "SEVERE":
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.15, 2))
                self.current_conf_gate_str = min(2.0, round(self.current_conf_gate_str + 0.2, 2))
            elif severity == "CRITICAL":
                self.current_mag_weight = 0.1
                self.metric_focus_target = 'sharpe_ratio'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'sortino_ratio':
            if severity == "MILD":
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.05, 2))
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.05, 2))
            elif severity == "SEVERE":
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.10, 2))
                self.current_dir_weight = min(2.0, round(self.current_dir_weight + 0.15, 2))
            elif severity == "CRITICAL":
                self.metric_focus_target = 'sortino_ratio'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'max_drawdown':
            if severity == "MILD":
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.08, 2))
            elif severity == "SEVERE":
                self.current_mag_weight = max(0.1, round(self.current_mag_weight - 0.20, 2))
                self.current_conf_gate_str = min(2.0, round(self.current_conf_gate_str + 0.3, 2))
            elif severity == "CRITICAL":
                self.current_mag_weight = 0.1
                self.current_conf_gate_str = 2.0
                self.metric_focus_target = 'max_drawdown'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'tp_mae':
            if severity == "MILD":
                self.current_mag_weight = min(2.0, round(self.current_mag_weight + 0.10, 2))
            elif severity == "SEVERE":
                self.current_mag_weight = min(2.0, round(self.current_mag_weight + 0.25, 2))
            elif severity == "CRITICAL":
                self.current_mag_weight = min(2.0, round(self.current_mag_weight + 0.40, 2))
                self.metric_focus_target = 'tp_mae'
                self.metric_focus_epochs_remaining = 5
        elif primary_metric == 'sl_mae':
            if severity == "MILD":
                self.current_mag_weight = min(2.0, round(self.current_mag_weight + 0.06, 2))
            elif severity == "SEVERE":
                self.current_mag_weight = min(2.0, round(self.current_mag_weight + 0.15, 2))
            elif severity == "CRITICAL":
                self.current_mag_weight = min(2.0, round(self.current_mag_weight + 0.25, 2))
                self.metric_focus_target = 'sl_mae'
                self.metric_focus_epochs_remaining = 5

        msg_parts.append(f"[OMNI-GOVERNOR] [{severity}] {primary_metric.upper()} Deficit ({deficit*100:.1f}%) -> Target Optimizations Applied.")

        if self.metric_focus_epochs_remaining > 0 and self.metric_focus_target == primary_metric:
            last_fired = self.metric_focus_last_fired.get(primary_metric, 0)
            if self.epoch_count - last_fired > 10 or last_fired == 0:
                self.metric_focus_last_fired[primary_metric] = self.epoch_count
                msg_parts.append(f"[METRIC FOCUS BURST] Triggered for {primary_metric.upper()} (5 Epochs)")
            else:
                self.metric_focus_epochs_remaining = 0
                self.metric_focus_target = None
                msg_parts.append(f"[METRIC FOCUS BURST] Queued for {primary_metric.upper()} (Cooldown active)")

        return msg_parts

    def get_phase(self):
        if self.task_type == "forex":
            return "CURRICULUM_FOLD"
        if self.current_res not in self.res_ladder:
            return "REFINEMENT"

        res_idx = self.res_ladder.index(self.current_res)
        phase = "REFINEMENT"

        if res_idx == 0 and self.current_fraction < 0.5:
            phase = "FOUNDATION"
        elif self.plateau_priority == "resolution":
            if res_idx < len(self.res_ladder) - 1:
                phase = "DEEPENING"
            elif self.current_fraction < 1.0:
                phase = "EXPANSION"
        else:
            if self.current_fraction < 1.0:
                phase = "EXPANSION"
            elif res_idx < len(self.res_ladder) - 1:
                phase = "DEEPENING"

        return phase

    def audit_epoch(self, current_quality, best_quality, epochs_no_improve, regression_epochs, sentinel_trigger_rate=0.0, current_lr=None, base_lr=None, current_loss=None, plcc=0.0, srcc=0.0, target_std=None, force_jump=False, train_loss=None, metrics_dict=None):
        if not self.enabled and not force_jump: return False, False, False, False, False, False, False, ""
        self.epoch_count += 1
        self.session_epoch_count += 1

        if getattr(self, 'metric_focus_epochs_remaining', 0) > 0:
            self.metric_focus_epochs_remaining -= 1
            if self.metric_focus_target == 'srcc':
                self.current_spearman_weight = 3.0
                self.current_rank_weight = 2.0
            elif self.metric_focus_target == 'plcc':
                self.current_emd_weight = 1.5
                self.current_rank_weight = 0.3
                self.current_spearman_weight = 0.2
            elif self.metric_focus_target == 'ssim':
                self.current_ssim_weight = 0.15
            elif self.metric_focus_target == 'lpips':
                self.current_lpips_weight = 0.08
            elif self.metric_focus_target == 'fid':
                self.current_stress = 5.0
            elif self.metric_focus_target == 'mae':
                self.current_huber_delta = 0.1
            elif self.metric_focus_target in ('dir_acc', 'win_rate'):
                self.current_dir_weight = 2.0
                if self.metric_focus_target == 'win_rate': self.current_conf_gate_str = 1.5
            elif self.metric_focus_target == 'profit_factor':
                self.current_dir_weight = 2.0
                self.current_mag_weight = 0.2
            elif self.metric_focus_target == 'max_drawdown':
                self.current_mag_weight = 0.1
                self.current_conf_gate_str = 2.0
            elif self.metric_focus_target in ('tp_mae', 'sl_mae'):
                self.current_mag_weight = 1.2

            self.lr_multiplier = getattr(self, 'cooling_factor', 0.8)
            self.head_lr_multiplier = min(self.lr_multiplier * 2.0, 1.5)
            self.jolt_window_remaining = 5

            if self.metric_focus_epochs_remaining == 0:
                self.metric_focus_target = None

        if getattr(self, 'breakout_lock', 0) > 0:
            self.breakout_lock -= 1

        if force_jump:
            try:
                epochs_at_res = self.epoch_count - self.last_res_jump_epoch
                print(f" [SEARCH] [HARDENING-DEBUG] Current Res: {self.current_res}px | Epochs at Res: {epochs_at_res} | Maturity Required: {self.manifold_maturity}")
                if epochs_at_res < self.manifold_maturity:
                    return False, False, False, False, False, False, False, f"[GUARD] [HARDENING] SOTA hit early, but locking at {self.current_res}px for weight stabilization (Manifold Maturity: {epochs_at_res}/{self.manifold_maturity})."

                current_idx = self.res_ladder.index(self.current_res)
                if current_idx < len(self.res_ladder) - 1:
                    next_res = self.res_ladder[current_idx + 1]
                    self.current_res = next_res
                    self.current_fraction = 0.5
                    self.last_res_jump_epoch = self.epoch_count
                    self.spatial_lock_remaining = self.stabilization_lock
                    self.stabilization_epochs = self.stabilization_lock
                    self.history = []
                    return True, True, False, False, False, True, False, f"[LAUNCH] [SOTA-FORCE] Jumping to {next_res}px Manifold..."
                else:
                    return False, False, False, False, False, False, False, "[SUCCESS] [SOTA-MAX] Already at maximum resolution."
            except Exception as e:
                print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

        if sentinel_trigger_rate == 0:
            self.recovery_streak += 1
            if self.recovery_streak >= 2 and self.stabilization_epochs > 0 and self.spatial_lock_remaining == 0:
                self.stabilization_epochs = 0
                print("[LAUNCH] [NPP] Stress at zero. Breaking stabilization lock.")
        else:
            self.recovery_streak = 0

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        self.history.append((current_quality, current_loss, train_loss))
        if len(self.history) > 5: self.history.pop(0)

        if current_quality > self.best_quality and self.best_quality > 0.0:
            if getattr(self, 'current_stress', 0.0) > 0.0:
                self.current_stress = 0.0

        self.best_quality = max(self.best_quality, current_quality)

        msg_parts = []
        if self.stabilization_epochs > 0:
            if current_quality < self.best_quality * (1.0 - self.breakout_threshold) and self.best_quality > 0:
                msg_parts.append(f"[BREAKOUT] Shield shattered! Quality dropped {(1-current_quality/self.best_quality)*100:.1f}%.")
                self.stabilization_epochs = 0
            else:
                self.stabilization_epochs -= 1
                recovery_delta = self.min_delta if self.task_type != "quality" else 0.05
                if current_quality - self.prev_quality > recovery_delta:
                    self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
                    self.stabilization_epochs = max(0, self.stabilization_epochs - 1)
                    msg_parts.append("[RECOVERY] Rapid quality gain detected. Cooldown and Lock shortened.")

                self.prev_quality = current_quality
                if current_loss: self.prev_loss = current_loss
                status_msg = f"[SIGNAL] Anchoring Manifold... (Cooldown: {self.cooldown_remaining})" if self.cooldown_remaining > 0 else "[SIGNAL] Anchoring Manifold..."
                if msg_parts: status_msg = " | ".join(msg_parts) + " | " + status_msg
                return False, False, False, False, False, False, False, status_msg

        is_resuming = self.session_epoch_count == 1 and self.epoch_count > 1
        is_regressing_shock = current_quality < self.best_quality * 0.95 and self.best_quality > 0
        if is_resuming and is_regressing_shock:
            self.prev_quality = current_quality
            if current_loss: self.prev_loss = current_loss
            return False, False, False, False, False, False, False, "[GUARD] [SHIELD] Resumption Shield Active. Buffering Momentum Shock."

        is_overfitting = False
        if len(self.history) >= 3:
            train_losses = [h[2] for h in self.history if len(h) > 2 and h[2] is not None]
            val_losses = [h[1] for h in self.history if h[1] is not None]
            if len(train_losses) >= 3 and len(val_losses) >= 3:
                train_trend = train_losses[-1] - train_losses[-3]
                val_trend = val_losses[-1] - val_losses[-3]
                if train_trend < -1e-4 and val_trend > 1e-4:
                    is_overfitting = True

        f_changed = r_changed = lr_changed = t_changed = c_changed = b_changed = False
        self.lr_multiplier = 1.0
        phase = self.get_phase()

        delta_q = current_quality - self.prev_quality
        is_turbulent = False
        if len(self.history) >= 3:
            q_values = [h[0] for h in self.history[-3:]]
            deltas = [q_values[i] - q_values[i-1] for i in range(1, len(q_values))]
            if all(deltas[i] * deltas[i-1] < 0 for i in range(1, len(deltas))):
                if all(abs(d) > self.min_delta * 2 for d in deltas):
                    is_turbulent = True

        is_plateaued = (epochs_no_improve >= self.plateau_patience)
        is_flat_leg = abs(delta_q) < self.min_delta and len(self.history) >= 2
        is_flat = is_plateaued or is_flat_leg

        abs_floor = 40.0 if self.task_type == "quality" else 0.0
        fidelity_floor = max(abs_floor, self.best_quality * 0.8)
        is_trapped = current_quality < fidelity_floor and len(self.history) >= 4
        if is_trapped:
            effective_min_delta = self.min_delta * 4
            is_flat_leg = abs(delta_q) < effective_min_delta
            is_flat = is_plateaued or is_flat_leg
            if is_flat: msg_parts.append("[TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        regress_threshold = -0.03 if self.task_type in ["quality", "parameter_prediction"] else -0.01
        is_regressing = delta_q < (self.prev_quality * regress_threshold) if self.prev_quality else False

        if self.task_type == "forex":
            is_collapsed = current_quality < 45.0
        else:
            is_collapsed = (current_quality < 0.05) or (plcc < -0.1)

        loss_is_stable = (current_loss <= self.prev_loss * 1.05) if current_loss and self.prev_loss else True
        is_expanding = phase in ["FOUNDATION", "EXPANSION"]

        should_retreat = (is_regressing or is_turbulent or is_collapsed)
        if plcc < -0.01 and self.current_temp > 0.7 and (target_std is None or target_std >= 0.15):
            self.current_temp = 0.5
            self.current_clamp = 20.0
            t_changed = c_changed = True
            msg_parts.append("[THERMAL SHOCK] PLCC negative. Sharpening manifold (Temp -> 0.5).")

        if self.spatial_lock_remaining > 0:
            self.spatial_lock_remaining -= 1
            loss_is_exploding = (current_loss > self.prev_loss * 1.25) if current_loss and self.prev_loss else False
            if is_regressing and not loss_is_exploding:
                is_regressing = False
                msg_parts.append(f"[SPATIAL LOCK] Buffering transition (Patience: {self.spatial_lock_remaining})")

        current_state = (self.current_res, round(self.current_fraction, 2))
        failures = self.failure_log.get(str(current_state), 0)

        if self.task_type in ["quality", "forex"]:
            is_turbulent = False
            if is_expanding: msg_parts.append("[SIGNAL] [RESONANCE] Turbulence detected but shielded. Holding manifold.")

        should_retreat = (is_regressing or is_turbulent or is_collapsed)
        if is_expanding and loss_is_stable and not is_collapsed:
            should_retreat = False
            if is_regressing: msg_parts.append("[GUARD] [MOMENTUM] Jitter detected but Loss is stable. Holding manifold.")

        if is_overfitting and is_expanding:
            should_retreat = False
            self.cooldown_remaining = 0
            msg_parts.append("[RESCUE] [OVERFITTING] Overfitting detected. Forcing dataset expansion to introduce variety.")
        elif is_overfitting and self.current_fraction >= 1.0:
            should_retreat = False
            self.cooldown_remaining = 0
            if self.task_type == "forex":
                self.current_stress = min(2.0, self.current_stress + 0.5)
                msg_parts.append(f"[REGULARIZE] [OVERFITTING] Temporal fold divergence detected. Applying mild stabilization (Level {self.current_stress}).")
            else:
                self.current_stress = min(5.0, self.current_stress + 1.0)
                msg_parts.append(f"[RESCUE] [OVERFITTING] Dataset exhausted. Deploying Stress Protocol (Level {self.current_stress}).")

        if should_retreat:
            self.failure_log[str(current_state)] = failures + 1
            msg_parts.append(f"[WARNING] NPP FAILURE: State {current_state} (Count: {self.failure_log[str(current_state)]})")

            if self.failure_log[str(current_state)] >= 2:
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                t_changed = True
                msg_parts.append("NPP LOOP: Forcing Numerical Shakeup")

            proven_threshold = max(85.0 if self.task_type == "quality" else 0.85, self.target_quality_score * 0.75)
            is_proven_manifold = self.best_quality >= proven_threshold

            can_spatial_retreat = (
                self.epoch_count - self.last_res_jump_epoch < 8 and
                getattr(self, 'breakout_lock', 0) == 0 and
                not is_proven_manifold
            )

            if can_spatial_retreat and self.current_res in self.res_ladder:
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

        elif sentinel_trigger_rate > 0.15:
            self.current_temp = min(1.2, self.current_temp * 1.2)
            self.lr_multiplier = 0.75
            self.current_stress = max(0.0, self.current_stress - 0.25)
            lr_changed = True
            t_changed = True
            self.stabilization_epochs = 2
            msg_parts.append(f"COOLING: Stress {sentinel_trigger_rate*100:.1f}% -> Temp {self.current_temp:.2f} | DataStress {self.current_stress:.1f}")

        stride_threshold = 0.75 if (self.current_res is not None and self.current_res < 512) else 0.90
        if self.target_quality_score > 1.0:
            stride_threshold = stride_threshold * self.target_quality_score
        propulsion_allowed = not should_retreat and self.cooldown_remaining == 0
        not_regressing = delta_q >= -self.min_delta

        trigger_propulsion = propulsion_allowed and (
            (is_overfitting and is_expanding) or
            is_plateaued or
            (not_regressing and (is_flat_leg or (current_quality > stride_threshold and delta_q < self.min_delta)))
        )

        if getattr(self, 'jolt_window_remaining', 0) > 0:
            self.jolt_window_remaining -= 1
            collapse_threshold = (self.prev_quality * -0.03) if (self.prev_quality and self.prev_quality > 1.0) else -0.015
            if delta_q < collapse_threshold or should_retreat or is_regressing:
                self.jolt_window_remaining = 0
                self.lr_multiplier = self.cooling_factor
                self.head_lr_multiplier = self.cooling_factor
                lr_changed = True
                msg_parts.append(f"[JOLT SHIELD] Early collapse triggered (Regression: {delta_q:.4f} < {collapse_threshold:.4f}). Cooling LR.")
            else:
                lr_changed = True
                msg_parts.append(f"SUSTAINED JOLT: Window Active ({self.jolt_window_remaining} epochs remaining | Head LR: {self.head_lr_multiplier:.2f}x | Backbone LR: {self.lr_multiplier:.2f}x)")

        if trigger_propulsion:
            jolt_ready = (self.epoch_count - getattr(self, 'last_jolt_epoch', -10)) > self.jolt_cooldown and self.cooldown_remaining == 0
            if is_flat and jolt_ready and getattr(self, 'jolt_window_remaining', 0) == 0:
                jolt = self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5)
                if is_trapped: jolt *= 1.5
                if self.task_type == "forex":
                    jolt *= 2.0
                jolt_base = float(jolt)

                if phase == "REFINEMENT":
                    self.lr_multiplier = round(jolt_base * 0.5, 3)
                    self.head_lr_multiplier = min(round(self.lr_multiplier * 3.0, 3), round(jolt_base * 1.5, 3))
                else:
                    self.lr_multiplier = jolt_base
                    self.head_lr_multiplier = jolt_base

                window_size = 5 if self.task_type == "forex" else 3
                self.jolt_window_remaining = window_size
                lr_changed = True
                self.last_jolt_epoch = self.epoch_count
                msg_parts.append(f"JOLT: Breaking Plateau with Head-Differential Propulsion (Head: {self.head_lr_multiplier:.2f}x, Backbone: {self.lr_multiplier:.2f}x | {window_size}-Epoch Window)")

            next_frac = min(1.0, self.current_fraction + self.fraction_increment)
            next_state = (self.current_res, round(next_frac, 2))

            if self.failure_log.get(str(next_state), 0) > 0:
                self.lr_multiplier = 0.6
                self.head_lr_multiplier = 0.6
                lr_changed = True
                msg_parts.append(f"ANCHOR: Caution ahead (Previous Failures). 0.6x LR.")

            if phase in ("FOUNDATION", "EXPANSION"):
                if self.current_fraction < 1.0:
                    self.current_fraction = next_frac
                    f_changed = True
                    msg_parts.append(f"PROPULSION: Data -> {self.current_fraction*100:.0f}%")
                    self.stabilization_epochs = 1
                    self.best_quality = current_quality
            elif phase == "DEEPENING" and self.current_res is not None and self.current_res in self.res_ladder:
                current_idx = self.res_ladder.index(int(self.current_res))
                next_res = self.res_ladder[current_idx + 1]

                if self.best_quality < self.target_quality_score * 0.80 and self.target_quality_score > 1.0:
                    self.lr_multiplier = self.cooling_factor
                    self.head_lr_multiplier = self.cooling_factor
                    lr_changed = True
                    msg_parts.append(f"RECOIL: Insufficient Quality for spatial jump. Cooling LR.")
                    self.stabilization_epochs = self.stabilization_lock
                else:
                    self.current_res = next_res
                    r_changed = b_changed = True
                    self.current_fraction = 0.15
                    f_changed = True
                    self.last_res_jump_epoch = self.epoch_count
                    self.spatial_lock_remaining = self.stabilization_lock
                    msg_parts.append(f"SPATIAL JUMP: {next_res}px | Data Reset 15% | Lock: ON")
                    self.stabilization_epochs = self.stabilization_lock
            else:
                if self.task_type == "forex":
                    if getattr(self, 'current_stress', 0.0) < 2.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                        self.current_stress = min(2.0, getattr(self, 'current_stress', 0.0) + 0.5)
                        self.lr_multiplier = 1.0
                        self.head_lr_multiplier = 1.10
                        self.jolt_window_remaining = 2
                        lr_changed = True
                        msg_parts.append(f"REFINEMENT: Plateau in Temporal Fold. Deploying Gentle Fold Tuning (Head LR: 1.10x)")
                    else:
                        if getattr(self, 'jolt_window_remaining', 0) == 0:
                            self.lr_multiplier = self.cooling_factor
                            self.head_lr_multiplier = self.cooling_factor
                            lr_changed = True
                            msg_parts.append("REFINEMENT: Precision Cooling")
                elif getattr(self, 'current_stress', 0.0) < 5.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                    self.current_stress = min(5.0, getattr(self, 'current_stress', 0.0) + 1.0)
                    jolt_base = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
                    self.lr_multiplier = round(jolt_base * 0.5, 3)
                    self.head_lr_multiplier = min(round(self.lr_multiplier * 3.0, 3), round(jolt_base * 1.5, 3))
                    self.jolt_window_remaining = 3
                    lr_changed = True
                    self.max_stress_stuck_epochs = 0
                    msg_parts.append(f"REFINEMENT: Trapped in Plateau. Deploying Stress Protocol (Level {self.current_stress}) & Differential Jolt")
                elif getattr(self, 'current_stress', 0.0) >= 5.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                    jolt_base = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
                    self.lr_multiplier = jolt_base
                    self.head_lr_multiplier = jolt_base
                    lr_changed = True
                    self.max_stress_stuck_epochs = getattr(self, 'max_stress_stuck_epochs', 0) + 1
                    stuck_patience = self.plateau_patience * 2
                    msg_parts.append(f"REFINEMENT: [MAX STRESS] Forcing Jolt (x{jolt_base:.2f}) to maintain momentum (Stuck: {self.max_stress_stuck_epochs}/{stuck_patience})")
                    if self.max_stress_stuck_epochs >= stuck_patience:
                        self.trigger_mini_swa = True
                        msg_parts.append(f"[MINI-SWA PULSE] Triggering weight averaging pulse across top checkpoints (Stuck: {self.max_stress_stuck_epochs} epochs)")
                else:
                    if getattr(self, 'jolt_window_remaining', 0) == 0:
                        self.lr_multiplier = self.cooling_factor
                        self.head_lr_multiplier = self.cooling_factor
                        lr_changed = True
                        msg_parts.append("REFINEMENT: SOTA Precision Cooling")

                    if metrics_dict is None:
                        metrics_dict = {'plcc': plcc, 'srcc': srcc}

                    lagging = self._identify_lagging_metric(metrics_dict)
                    if lagging:
                        adjustments = self._apply_targeted_optimizations(lagging, metrics_dict)
                        msg_parts.extend(adjustments)

        if not (is_regressing or is_turbulent or sentinel_trigger_rate > 0.15) and self.current_temp > self.min_temp:
            if self.task_type == "forex":
                phase_min = self.min_temp
            elif self.task_type != "quality":
                phase_min = 0.05 if phase == "REFINEMENT" else 0.1
            else:
                phase_min = self.min_temp

            floor = max(phase_min, self.thermal_floor.get(str(current_state), self.min_temp))

            if self.cooldown_remaining == 0 and self.current_temp > floor:
                sharpen_rate = 0.95 if self.current_temp > 1.2 else self.sharpening_rate
                self.current_temp = max(floor, self.current_temp * sharpen_rate)
                t_changed = True
                msg_parts.append(f"SHARPENING: Temp -> {self.current_temp:.2f}")
            elif self.cooldown_remaining > 0:
                msg_parts.append(f"MEDITATION: Cooldown active ({self.cooldown_remaining} epochs)")

        self.prev_quality = current_quality
        if current_loss: self.prev_loss = current_loss

        if self.task_type == "quality":
            if self.cooldown_remaining == 0:
                self.current_temp = min(1.0, self.current_temp)
            self.current_clamp = min(self.clamp_range[1], self.current_clamp)

        early_stop_triggered = False
        if epochs_no_improve >= self.plateau_patience:
            if getattr(self, 'metric_focus_epochs_remaining', 0) == 0 and self.stabilization_epochs == 0 and self.cooldown_remaining == 0:
                early_stop_triggered = True
                msg_parts.append(f"[EARLY STOPPING] Patience exceeded ({epochs_no_improve}/{self.plateau_patience}). Fold exhausted.")

        final_msg = f"[LAUNCH] [{phase}] " + " | ".join(msg_parts) if msg_parts else ""

        return f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, early_stop_triggered, final_msg

    def get_dynamic_save_interval(self, avg_iter_time, total_iters):
        # 2026 v15.6: Floor raised to 15%. On 11-hour epochs this drops from 20
        # checkpoint writes/epoch to 7, saving 2-3 minutes of blocking I/O per epoch.
        if avg_iter_time <= 0: return 0.2
        epoch_duration_mins = (avg_iter_time * total_iters) / 60
        # Very long epochs (>4h) -> save only at midpoint and end
        if epoch_duration_mins > 240:
            return 0.5
        target_pct = 15 / max(1, epoch_duration_mins)
        return max(0.15, min(0.5, target_pct))

    def veto_resolution_jump(self, fallback_res, reason="VRAM physical ceiling"):
        self.current_res = fallback_res
        self.spatial_lock_remaining = max(self.spatial_lock_remaining, self.stabilization_lock)
        self.stabilization_epochs = max(self.stabilization_epochs, self.stabilization_lock)
        self.cooldown_remaining = max(self.cooldown_remaining, 5)
        if hasattr(self, 'res_ladder') and self.res_ladder:
            self.res_ladder = [r for r in self.res_ladder if r <= fallback_res]
            if not self.res_ladder:
                self.res_ladder = [fallback_res]
        print(f" [GUARD] [GOVERNOR] Spatial jump VETOED ({reason}). Anchoring at {fallback_res}px (Lock: ON).")

    def get_state(self):
        return {
            "sample_fraction": self.current_fraction,
            "input_size": self.current_res,
            "softmax_temp": self.current_temp,
            "logit_clamp": self.current_clamp,
            "lr_multiplier": self.lr_multiplier,
            "head_lr_multiplier": getattr(self, 'head_lr_multiplier', self.lr_multiplier),
            "jolt_window_remaining": getattr(self, 'jolt_window_remaining', 0),
            "batch_size": self.current_batch,
            "accumulation_steps": self.current_acc,
            "stabilization_epochs": self.stabilization_epochs,
            "failure_log": self.failure_log,
            "history": self.history,
            "thermal_floor": self.thermal_floor,
            "cooldown_remaining": self.cooldown_remaining,
            "spatial_lock_remaining": self.spatial_lock_remaining,
            "last_res_jump_epoch": self.last_res_jump_epoch,
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
            "rank_margin": getattr(self, 'current_rank_margin', 0.10),
            "soft_spearman_weight": getattr(self, 'current_spearman_weight', 0.5),
            "lpips_weight": getattr(self, 'current_lpips_weight', 0.025),
            "mag_weight": getattr(self, 'current_mag_weight', 0.5),
            "dir_weight": getattr(self, 'current_dir_weight', 1.0),
            "emd_weight": getattr(self, 'current_emd_weight', 1.0),
            "ssim_weight": getattr(self, 'current_ssim_weight', 0.0),
            "huber_delta": getattr(self, 'current_huber_delta', 1.0),
            "conf_gate_str": getattr(self, 'current_conf_gate_str', 1.0),
            "metric_focus_epochs_remaining": getattr(self, 'metric_focus_epochs_remaining', 0),
            "metric_focus_target": getattr(self, 'metric_focus_target', None),
            "metric_focus_last_fired": getattr(self, 'metric_focus_last_fired', {}),
        }

    def load_state(self, state, preserve_curriculum=False):
        if not state: return
        if not preserve_curriculum:
            self.current_fraction = state.get("sample_fraction", self.current_fraction)
            raw_res = state.get("input_size", self.current_res)
            self.current_res = raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res

            if self.task_type == "forex" and self.epoch_count <= 2:
                opt_cfg = self.model_info.get("optimization", {})
                init_frac = opt_cfg.get("initial_fraction", 0.15)
                saved_frac = state.get("sample_fraction", 1.0)
                if saved_frac >= 0.99:
                    if init_frac < 0.99:
                        print(f" [RESILIENCY] Overriding legacy 100% fraction from checkpoint with configured curriculum initial fraction: {init_frac*100:.1f}%.")
                        self.current_fraction = init_frac

            if self.current_res is not None and self.current_res not in self.res_ladder:
                self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
        else:
            raw_res = state.get("input_size", self.current_res)
            self.sota_resolution = raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res

        self.current_temp = max(self.min_temp, state.get("softmax_temp", self.current_temp))
        if self.task_type == "quality": self.current_temp = min(1.0, self.current_temp)
        self.current_clamp = state.get("logit_clamp", self.current_clamp)
        if "rank_weight" in state: self.current_rank_weight = float(state["rank_weight"])
        if "rank_margin" in state: self.current_rank_margin = float(state["rank_margin"])
        if "soft_spearman_weight" in state: self.current_spearman_weight = float(state["soft_spearman_weight"])
        if "lpips_weight" in state: self.current_lpips_weight = float(state["lpips_weight"])
        if "mag_weight" in state: self.current_mag_weight = float(state["mag_weight"])
        if "dir_weight" in state: self.current_dir_weight = float(state["dir_weight"])
        if "emd_weight" in state: self.current_emd_weight = float(state["emd_weight"])
        if "ssim_weight" in state: self.current_ssim_weight = float(state["ssim_weight"])
        if "huber_delta" in state: self.current_huber_delta = float(state["huber_delta"])
        if "conf_gate_str" in state: self.current_conf_gate_str = float(state["conf_gate_str"])
        self.metric_focus_epochs_remaining = state.get("metric_focus_epochs_remaining", 0)
        self.metric_focus_target = state.get("metric_focus_target", None)
        raw_mflf = state.get("metric_focus_last_fired", {})
        self.metric_focus_last_fired = {k: int(v) for k, v in raw_mflf.items()}
        self.lr_multiplier = state.get("lr_multiplier", self.lr_multiplier)
        self.head_lr_multiplier = state.get("head_lr_multiplier", self.lr_multiplier)
        self.jolt_window_remaining = state.get("jolt_window_remaining", 0)
        self.current_batch = state.get("batch_size", self.current_batch)
        self.current_acc = state.get("accumulation_steps", self.current_acc)
        self.stabilization_epochs = state.get("stabilization_epochs", 0)
        self.failure_log = state.get("failure_log", {})
        self.history = state.get("history", [])
        self.thermal_floor = state.get("thermal_floor", {})
        self.cooldown_remaining = state.get("cooldown_remaining", 0)
        self.last_jolt_epoch = state.get("last_jolt_epoch", -10)
        self.spatial_lock_remaining = state.get("spatial_lock_remaining", 0)
        self.last_res_jump_epoch = state.get("last_res_jump_epoch", 0)
        self.epoch_count = state.get("epoch_count", self.epoch_count)
        self.best_quality = state.get("best_quality", self.best_quality)
        self.current_stress = state.get("stress", 0.0)
        self.max_stress_stuck_epochs = state.get("max_stress_stuck_epochs", 0)

        self.consecutive_rollbacks = state.get("consecutive_rollbacks", 0)
        raw_history = state.get("rollback_history", {})
        self.rollback_history = {int(k): v for k, v in raw_history.items()}
        self.gate_relaxation_epochs = state.get("gate_relaxation_epochs", 0)
        self.sota_resolution = state.get("sota_resolution", self.sota_resolution)
        self.breakout_lock = state.get("breakout_lock", 0)

    def recoil(self):
        self.current_temp = min(1.5, self.current_temp * 1.3)
        if self.task_type == "quality": self.current_temp = min(1.0, self.current_temp)
        self.stabilization_epochs = 3
        return f"[NPP] RECOIL: Retaining data fraction at {self.current_fraction*100:.0f}% | Temp Heatup {self.current_temp:.2f}"

    def reset_best(self):
        self.best_quality = 0.0
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.history = []
        self.stabilization_epochs = 2
        print(" [GOVERNOR] SOTA Memory Purged. Establishing fresh baseline for current manifold.")

    def register_rollback(self):
        if not getattr(self, 'loop_breaker_enabled', True):
            return

        self.consecutive_rollbacks += 1
        res_key = self.current_res
        self.rollback_history[res_key] = self.rollback_history.get(res_key, 0) + 1

        threshold = getattr(self, 'loop_breaker_threshold', 2)
        if self.rollback_history.get(res_key, 0) >= threshold:
            strategy = getattr(self, 'loop_breaker_strategy', 'auto')
            if strategy == "auto":
                strategy = "escalate" if self.task_type == "quality" else "relax"

            current_idx = self.res_ladder.index(self.current_res) if (hasattr(self, 'res_ladder') and self.current_res in self.res_ladder) else -1
            has_higher_res = (0 <= current_idx < len(self.res_ladder) - 1)
            target_res = self.sota_resolution if (self.sota_resolution is not None and self.sota_resolution > self.current_res) else (self.res_ladder[current_idx + 1] if has_higher_res else None)

            if strategy == "escalate" and target_res is not None:
                old_res = self.current_res
                self.current_res = target_res
                if self.current_res not in self.res_ladder:
                    self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))

                self.rollback_history[self.current_res] = 0
                self.consecutive_rollbacks = 0
                self.gate_relaxation_epochs = 0
                self.current_fraction = 0.15

                self.breakout_lock = 8

                print(f"\n================================================================================")
                print(f" [BREAKOUT] [GOVERNOR] Resolution-Regression Lock detected at {old_res}px!")
                print(f"   -> Automatically promoting training resolution: {old_res}px -> {self.current_res}px.")
                print(f"   -> Breakout retreat protection active for next 8 epochs.")
                print(f"================================================================================\n")

                self.reset_best()

            elif strategy in ["escalate", "relax"]:
                self.gate_relaxation_epochs = 6
                self.consecutive_rollbacks = 0
                print(f"\n================================================================================")
                print(f" [BREAKOUT] [GOVERNOR] Stagnation Rollback Lock detected at {res_key}px!")
                print(f"   -> Activating Dynamic Gate Relaxation for next 6 epochs to allow weights to settle.")
                print(f"================================================================================\n")

    def get_active_drift_gate(self, config_gate):
        if self.gate_relaxation_epochs > 0:
            self.gate_relaxation_epochs -= 1
            return min(0.80, config_gate * 0.85)
        return config_gate

    def get_active_regression_limit(self, config_limit):
        if self.gate_relaxation_epochs > 0 or self.rollback_history.get(self.current_res, 0) > 0:
            return max(config_limit, self.plateau_patience + 2)
        return config_limit


def export_webgpu_onnx(model, save_path, dummy_input_shape=(1, 3, 512, 512)):
    import torch
    import io
    from contextlib import redirect_stdout, redirect_stderr
    print(f" [MEMORY-SENTINEL] Exporting zero-copy WebGPU sharing payload to {save_path}...")

    model_to_export = model.module if hasattr(model, 'module') else model
    model_to_export.eval()

    try:
        device = next(model_to_export.parameters()).device
    except StopIteration:
        device = 'cpu'
    dummy_input = torch.randn(dummy_input_shape, device=device)

    try:
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            torch.onnx.export(
                model_to_export,
                (dummy_input,),
                save_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=None
            )
        print(f" [MEMORY-SENTINEL] WebGPU ONNX export successful! Opset: 17, Shape: {dummy_input_shape}")
        return True
    except Exception as e:
        print(f" [MEMORY-SENTINEL] WebGPU export failed: {e}")
        return False