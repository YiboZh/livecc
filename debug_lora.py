#!/usr/bin/env python3
"""
Debug script to verify LoRA setup
Run this to check if LoRA can be applied to your model correctly.
"""
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig

def test_lora_application(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    print(f"Testing LoRA application on {model_name}")
    print("="*80)
    
    # Load model
    print("\n1. Loading model...")
    config = AutoConfig.from_pretrained(model_name)
    print(f"   Architecture: {config.architectures[0]}")
    
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Use CPU to avoid memory issues
    )
    print(f"   Model loaded successfully")
    
    # Print some module names
    print("\n2. Sample module names in the model:")
    module_names = [name for name, _ in model.named_modules()]
    
    # Look for common patterns
    patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"]
    print("\n   Modules containing target patterns:")
    for pattern in patterns:
        matching = [name for name in module_names if pattern in name]
        if matching:
            print(f"   - {pattern}: {len(matching)} modules found")
            if len(matching) <= 3:
                for m in matching:
                    print(f"     └─ {m}")
            else:
                print(f"     └─ {matching[0]}")
                print(f"     └─ {matching[1]}")
                print(f"     └─ ...")
                print(f"     └─ {matching[-1]}")
    
    # Test LoRA application
    print("\n3. Applying LoRA...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    
    try:
        model = get_peft_model(model, lora_config)
        print("   ✅ LoRA applied successfully!")
        
        # Check trainable parameters
        print("\n4. Trainable parameters:")
        model.print_trainable_parameters()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        if trainable_params == 0:
            print("\n   ❌ ERROR: No trainable parameters!")
            print("   This means LoRA adapters were not created.")
        else:
            print(f"\n   ✅ SUCCESS: {trainable_params:,} trainable parameters ({100*trainable_params/total_params:.2f}%)")
            
            # Show some LoRA layer names
            print("\n5. LoRA adapter layers:")
            lora_layers = [name for name, _ in model.named_modules() if "lora" in name.lower()]
            for layer in lora_layers[:10]:
                print(f"   - {layer}")
            if len(lora_layers) > 10:
                print(f"   ... and {len(lora_layers)-10} more")
                
    except Exception as e:
        print(f"   ❌ ERROR applying LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-VL-7B-Instruct"
    
    success = test_lora_application(model_name)
    
    if success:
        print("\n" + "="*80)
        print("✅ LoRA test PASSED - your setup should work for training")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ LoRA test FAILED - check the errors above")
        print("="*80)
        sys.exit(1)

