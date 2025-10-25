#!/usr/bin/env python3
"""
Convert BGE embedding model to Core ML format for iOS.
"""

import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer
import numpy as np

def convert_bge_to_coreml():
    print("📦 Loading BGE model...")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    # Set to eval mode
    model.eval()
    
    print("🔧 Creating example inputs...")
    # Create example inputs for tracing
    example_text = "sample text for tracing"
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("🎯 Tracing model...")
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (input_ids, attention_mask),
            strict=False
        )
    
    print("🍎 Converting to Core ML...")
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 512), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 512), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="last_hidden_state")
        ],
        minimum_deployment_target=ct.target.iOS16,
    )
    
    # Add metadata
    mlmodel.short_description = "BGE Small EN v1.5 - Sentence Embedding Model"
    mlmodel.author = "BAAI"
    mlmodel.license = "MIT"
    mlmodel.version = "1.5"
    
    # Save the model
    output_path = "BGEEmbedder.mlmodel"
    mlmodel.save(output_path)
    
    print(f"✅ Model saved to: {output_path}")
    print(f"📏 Model size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print("\n📋 Next steps:")
    print("1. Copy BGEEmbedder.mlmodel to app/ios/Runner/")
    print("2. Open Xcode and drag the file into the project")
    print("3. Xcode will auto-generate Swift code for the model")

if __name__ == "__main__":
    import os
    convert_bge_to_coreml()
