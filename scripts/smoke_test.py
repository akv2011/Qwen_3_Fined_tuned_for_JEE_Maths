#!/usr/bin/env python3
"""
GPU Smoke Test for JEE Math Models
Quick validation script for Qwen2.5-Math and Aryabhata models on 16GB GPU
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_gpu():
    """Check GPU availability and specs"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {device_name}")
    print(f"✅ VRAM: {total_memory:.1f} GB")
    
    if total_memory < 15:
        print("⚠️  Warning: Less than 16GB VRAM detected")
    
    return True

def test_qwen_math():
    """Test Qwen2.5-Math-1.5B-Instruct"""
    model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    print(f"\n🧪 Testing {model_id}")
    
    try:
        # Load with memory management
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map="auto",
        )
        print("✅ Model loaded successfully")
        
        # Simple math test
        prompt = "Solve: 2x + 5 = 11. What is x?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ Response: {response[len(prompt):].strip()}")
        print(f"✅ Time: {generation_time:.2f}s")
        print(f"✅ Peak VRAM: {peak_memory:.2f} GB")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ Qwen test failed: {e}")
        return False

def test_aryabhata():
    """Test Aryabhata-1.0"""
    model_id = "PhysicsWallahAI/Aryabhata-1.0"
    print(f"\n🧪 Testing {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map="auto",
        )
        print("✅ Model loaded successfully")
        
        # Physics/math test
        prompt = "Calculate the kinetic energy of a 2kg object moving at 5 m/s."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ Response: {response[len(prompt):].strip()}")
        print(f"✅ Time: {generation_time:.2f}s")
        print(f"✅ Peak VRAM: {peak_memory:.2f} GB")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ Aryabhata test failed: {e}")
        if "CUDA out of memory" in str(e):
            print("💡 Try reducing max_new_tokens or use CPU")
        return False

def main():
    """Run all smoke tests"""
    print("🚀 JEE Math Models - GPU Smoke Test")
    print("=" * 50)
    
    # Check environment
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    if not check_gpu():
        print("❌ GPU check failed - exiting")
        return False
    
    # Run tests
    results = []
    
    print("\n" + "=" * 50)
    results.append(test_qwen_math())
    
    print("\n" + "=" * 50)
    results.append(test_aryabhata())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SMOKE TEST SUMMARY")
    print(f"✅ Qwen2.5-Math: {'PASS' if results[0] else 'FAIL'}")
    print(f"✅ Aryabhata-1.0: {'PASS' if results[1] else 'FAIL'}")
    
    if all(results):
        print("🎉 All tests passed! GPU setup ready for JEE math training.")
        return True
    else:
        print("⚠️  Some tests failed. Check GPU memory and dependencies.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
