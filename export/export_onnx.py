import torch
import os
from models.multitask_restorer import MultiTaskRestorer

def export():
    model = MultiTaskRestorer().eval()
    dummy = torch.randn(1, 3, 256, 256)
    
    tasks = ["denoise", "deblur", "derain", "dehaze", "lowlight", "superres"]
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for task in tasks:
        output_path = f"checkpoints/multitask_{task}.onnx"
        print(f"Exporting ONNX for task: {task} to {output_path}...")
        
        class FixedTaskModel(torch.nn.Module):
            def __init__(self, model, task):
                super().__init__()
                self.model = model
                self.task = task
            def forward(self, x):
                pred, weights = self.model(x, self.task)
                return pred, weights # Return both for browser diagnostics
        
        task_model = FixedTaskModel(model, task)
        
        torch.onnx.export(
            task_model,
            dummy,
            output_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output", "task_weights"],
            dynamic_axes=None
        )

if __name__ == "__main__":
    export()
