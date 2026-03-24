import numpy as np

def analyze(image):
    # Professional Analysis Heuristics
    luma = np.mean(image)
    std = np.std(image)
    
    return {
        "face": False, # Placeholder for RetinaFace
        "blur": std < 40,
        "noise": std > 80,
        "lowlight": luma < 100
    }

def decide(analysis):
    if analysis["noise"]:
        return "denoise"
    if analysis["blur"]:
        return "deblur"
    if analysis["lowlight"]:
        return "lowlight"
    return "superres"

def run_pipeline(image, model_manager, tile_size=256, overlap=32):
    """
    Next-Gen inference pipeline with learned task routing.
    """
    # If using the new MoE model, we don't need manual analyze/decide triggers
    # The model handles soft-routing internally.
    
    h, w, _ = image.shape
    stride = tile_size - overlap
    
    if h <= tile_size and w <= tile_size:
        padded = np.pad(image, ((0, tile_size-h), (0, tile_size-w), (0, 0)), mode='reflect')
        # MoE model returns (pred, weights)
        restored, weights = model_manager.run(padded, task=None) 
        
        # Log detected tasks
        task_idx = weights[0].argmax()
        print(f"Auto-Detected Primary Task: {task_idx}")
        
        return restored[:h, :w]

    output = np.zeros_like(image, dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    
    tile_weight = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        tile_weight[:overlap, :] *= ramp[:, np.newaxis]
        tile_weight[-overlap:, :] *= ramp[::-1, np.newaxis]
        tile_weight[:, :overlap] *= ramp[np.newaxis, :]
        tile_weight[:, -overlap:] *= ramp[np.newaxis, ::-1]

    for y in range(0, h - overlap, stride):
        for x in range(0, w - overlap, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size
            
            tile = image[y_start:y_end, x_start:x_end]
            restored_tile = model_manager.run(tile, task)
            
            local_weight = tile_weight.copy()
            if y_start == 0: local_weight[:overlap, :] = 1.0
            if y_end == h:   local_weight[-overlap:, :] = 1.0
            if x_start == 0: local_weight[:, :overlap] = 1.0
            if x_end == w:   local_weight[:, -overlap:] = 1.0
            
            output[y_start:y_end, x_start:x_end] += restored_tile * local_weight[:, :, np.newaxis]
            weight[y_start:y_end, x_start:x_end] += local_weight

    weight[weight == 0] = 1.0
    return (output / weight[:, :, np.newaxis]).astype(np.uint8)
