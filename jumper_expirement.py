import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import math, random
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR # Added a standard scheduler for example
from torch.optim.swa_utils import update_bn
from sgd_jumper import Jumper
from utils import cifar_loaders, resnet18_cifar #, vgg11_bn_cifar

class JumperExperiment:
    def __init__(self, model_class, train_loader, test_loader, num_classes, device="cuda"):
        self.model_class = model_class
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.results = []
        self.trajectories = [] 
        self.loss_log = [] 

    def evaluate(self, model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    # Added scheduler_gen parameter (defaults to None)
    def run(self, opt_name, optimizer_gen, scheduler_gen=None, epochs=50):
        print(f"\n--- Running: {opt_name} ---")
        model = self.model_class(self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        steps_per_epoch = len(self.train_loader)

        optimizer = optimizer_gen(model.parameters())
        print(f"Initialized optimizer: {optimizer.__class__.__name__}")
        # Instantiate scheduler if provided
        scheduler = scheduler_gen(optimizer) if scheduler_gen else None
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        period = 1.02
        for epoch in range(1, epochs + 1):
            start_epoch_time = time.perf_counter()

            model.train()
            total_loss = 0.0
            n = 0
            
            # Print current LR at the start of the epoch
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} starting | learning_rate: {current_lr:.4e}")

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # OPTIONAL: Uncomment if using a batch-level scheduler (e.g., OneCycleLR)
                # if scheduler is not None: scheduler.step()
                
                total_loss += loss.item()
                n += 1
                
            # Pre-Jump
            total_loss /= n
            acc_pre = self.evaluate(model)
            Pre_Jump_time = time.perf_counter() - start_epoch_time
            
            # Jump logic
            immediate_post_jump_time = 0.0
            update_bn_time = 0.0
            period = period * 1.02

            if hasattr(optimizer, 'jump'):
                if epoch > 2 and epoch <= 200:
                    optimizer.jump()
                    immediate_post_jump_time = time.perf_counter() - Pre_Jump_time - start_epoch_time
                    update_bn_time = time.perf_counter() - immediate_post_jump_time - Pre_Jump_time - start_epoch_time
                    acc_post = self.evaluate(model)
                else:
                    acc_post = 'N/A'
            else:
                acc_post = None
                
            # Step the scheduler at the end of the epoch (Standard for StepLR, CosineAnnealingLR)
            if scheduler is not None:
                scheduler.step()

            Jump_time = time.perf_counter() - start_epoch_time - Pre_Jump_time
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc_pre:.2f}% | Post-Jump Acc: {acc_post if acc_post else 'N/A'} | Time:{Pre_Jump_time:.1f}s + {Jump_time:.1f}s , period:{period}")

            self.results.append({
                'optimizer': opt_name, 'epoch': epoch, 'test_acc': acc_pre, 
                'jump_acc': acc_post, 'vram_mb': vram, 'time': time.perf_counter() - start_time
            })

    def export(self, prefix):
        pd.DataFrame(self.results).to_csv(f"{prefix}_results.csv", index=False)
        pd.DataFrame(self.trajectories).to_csv(f"{prefix}_weights.csv", index=False)
        pd.DataFrame(self.loss_log).to_csv(f"{prefix}_loss_log.csv", index=False)



# --- Execution Block ---
if __name__ == "__main__":

    train_loader, test_loader, num_classes = cifar_loaders(dataset="cifar10", batch_size=128)
    epochs_count = 200

    cosine_schd_gen = lambda opt : CosineAnnealingLR(opt, T_max=epochs_count//2) 
    multi_step_gen = lambda opt : torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 160], gamma=0.1) 
    exp = JumperExperiment(resnet18_cifar, train_loader, test_loader, num_classes=num_classes)

    # 1. Custom Jumper optimizer (internally schedules LR based on your setup)
    exp.run("Jumper", 
    optimizer_gen=lambda p: Jumper(p,steps_per_epoch=len(train_loader),lr=0.8,jump_mult=20.0,fit_type='log',ocilation=0.1,momentum=0.0,weight_decay=5e-4),
     epochs=epochs_count,
     scheduler_gen=cosine_schd_gen)  
    
    # 2. SGD with a Cosine Annealing Scheduler passed explicitly

    exp.run(
        "SGD", 
        optimizer_gen=lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=5e-4),
        scheduler_gen=cosine_schd_gen, 
        epochs=epochs_count
            )
    # 3. Vanilla SGD (No scheduler passed; defaults to None)
    
    exp.run("AdamW",
        optimizer_gen=lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=5e-4, betas=(0.9, 0.999)),
        scheduler_gen=cosine_schd_gen,
        epochs=epochs_count)

    exp.export("./benchmarks/jumper_exp")