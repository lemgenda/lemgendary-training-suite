import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, Blip2ForConditionalGeneration, BitsAndBytesConfig

class LLaVA_v1_5(nn.Module):
    """
    LemGendary LLaVA v1.5 Master Wrapper (v2.0 Hardened).
    Implements 4-bit Quantization and LoRA for Phase 5 Multimodal Training.
    """
    def __init__(self, checkpoint=None, **kwargs):
        super().__init__()
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
        
        # 2026: SOTA 4-bit Quantization for VRAM seating
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # 2026: Inject LoRA Adapter via PEFT
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "v_proj"], # Projections for multimodal alignment
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            print("✨ [MULTIMODAL] LLaVA-v1.6 Hardened with QLoRA.")
        except ImportError:
            print("⚠️ [RESILIENCE] PEFT not found. LLaVA remains in Standard mode.")
        
    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

class BLIP_2(nn.Module):
    """
    LemGendary BLIP-2 Master Wrapper (v2.0 Hardened).
    Implements QLoRA for efficient cross-modal reasoning.
    """
    def __init__(self, checkpoint=None, **kwargs):
        super().__init__()
        model_id = "Salesforce/blip2-opt-2.7b"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q", "v"], # Query/Value projections in Q-Former and Language Model
                lora_dropout=0.05,
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            print("✨ [MULTIMODAL] BLIP-2 Hardened with QLoRA.")
        except ImportError:
            pass
        
    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
