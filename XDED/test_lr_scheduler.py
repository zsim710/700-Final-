#!/usr/bin/env python3
"""
Test the NoamScheduler to see what learning rate schedule it produces.
This helps diagnose the learning rate issue from training.
"""

import torch
import matplotlib.pyplot as plt
from speechbrain.nnet.schedulers import NoamScheduler

# Create dummy model and optimizer
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Test both configurations
print("="*80)
print("TESTING NOAM SCHEDULER CONFIGURATIONS")
print("="*80)

configs = [
    {"name": "Without model_size (BROKEN)", "lr_initial": 1e-4, "n_warmup_steps": 10000, "model_size": None},
    {"name": "With model_size=144 (FIXED)", "lr_initial": 1e-4, "n_warmup_steps": 10000, "model_size": 144},
    {"name": "With model_size=256", "lr_initial": 1e-4, "n_warmup_steps": 10000, "model_size": 256},
]

plt.figure(figsize=(12, 6))

for config in configs:
    print(f"\n{config['name']}:")
    print(f"  lr_initial={config['lr_initial']}, warmup_steps={config['n_warmup_steps']}, model_size={config['model_size']}")
    
    # Reset optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['lr_initial']
    
    scheduler = NoamScheduler(
        lr_initial=config['lr_initial'],
        n_warmup_steps=config['n_warmup_steps'],
        model_size=config['model_size']
    )
    
    learning_rates = []
    steps = []
    
    # Simulate 50 epochs with 179 steps per epoch (batch_size=8, 1430 samples)
    num_steps = 50 * 179
    
    for step in range(num_steps):
        current_lr, new_lr = scheduler(optimizer)
        learning_rates.append(new_lr)
        steps.append(step)
        
        # Print first 10 steps and some milestones
        if step < 10 or step in [100, 500, 1000, 5000, 10000]:
            print(f"    Step {step}: LR = {new_lr:.8f}")
    
    # Plot
    plt.plot(steps, learning_rates, label=config['name'], linewidth=2)
    
    # Print summary
    print(f"  Max LR reached: {max(learning_rates):.8f}")
    print(f"  LR at step 8950 (epoch 50): {learning_rates[8949]:.8f}")

plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('NoamScheduler Learning Rate Schedule Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_schedule_comparison.png', dpi=150)
print(f"\n✅ Plot saved to: lr_schedule_comparison.png")

# Show expected behavior
print("\n" + "="*80)
print("EXPECTED BEHAVIOR:")
print("="*80)
print("Without model_size:")
print("  - normalize = warmup_steps^0.5 = sqrt(10000) = 100")
print("  - This scales DOWN the learning rate by 100x!")
print("  - Max LR ≈ 1e-4 / 100 = 1e-6 (TOO SMALL)")
print("\nWith model_size=144:")
print("  - normalize = model_size^(-0.5) = 1/sqrt(144) = 1/12 ≈ 0.083")
print("  - This scales UP the learning rate slightly")
print("  - Max LR ≈ 1e-4 * 12 ≈ 1.2e-3 (GOOD)")
print("\nConclusion: Always set model_size parameter for NoamScheduler!")
