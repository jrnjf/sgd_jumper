# 🦘 SGD-Jumper

**SGD-Jumper** is a PyTorch optimizer that accelerates convergence and improves generalization. By tracking historical weight trajectories, it projects and performs strategic "jumps" into promising loss regions, allowing the model to escape flat valleys and local minima more effectively than traditional gradient methods.

## 📋 Requirements

-  Python 3.8+
    
-  PyTorch 1.12+
    
- Torchvision _(to run experiment only)_
    
-  Pandas _(to run experiment only)_
    
-  Numpy
    

## 🛠️ Usage

### 📥 Installation

```bash
git clone https://github.com/jrnjf/sgd_jumper.git

cd sgd-jumper
```
### 💻 Code Snippet


Python

```python
from sgd_jumper import Jumper
from utils import resnet18_cifar

model = resnet18_cifar(num_classes=10)
steps_per_epoch = len(train_loader)

optimizer = Jumper(
    model.parameters(),
    steps_per_epoch=steps_per_epoch,
    lr=0.5,
    lr_SWR=10,
    momentum=0.0,
    gamma=20.0,
    fit_type='log',        # Options: 'linear', 'log', 'sqrt'
    weight_decay=1e-4
)
```

### Initialization Variables

- **`params`**: Iterable of parameters to optimize or dicts defining parameter groups.
    
- **`steps_per_epoch`**: Total number of batch steps contained within one training epoch.
    
- **`lr`**: Base learning rate.
    
- **`ocilation`**: Amplitude for the built-in cyclic triangle-wave learning modulation loop.
    
- **`momentum`**: SGD momentum.
    
- **`gamma`**: Proportional multiplier step distance scale applied during trend extrapolation for `fit_type='log'` and  `gamma =~[0-50]` for `fit_type=linear`or `sqrt` then `gamma=~[0-2]` 
    
- **`fit_type`**: Mathematical kernel used to fit weight trajectories (`linear`, `log`, or `sqrt`).
    
- **`weight_decay`**: L2 regularization coefficient weight decay penalty.
    
## 📊 Performance

### 🖥️ Evaluation Environment

- 📁 **Dataset:** CIFAR-10
    
- 🤖 **Model:** Modified ResNet-18
    
- 🔌 **Hardware:** Tested on an NVIDIA RTX A4000 GPU
    

As demonstrated in the validation curves below, SGD-Jumper achieves rapid early acceleration, climbing into higher accuracy boundaries significantly faster than SGD-M and ultimately establishing a higher final generalization plateau than AdamW.
![Acc vs Epoch curve](https://github.com/jrnjf/sgd_jumper/blob/main/benchmarks/acc_epoch.png)

### ⏱️ Wall-Clock Target Acquisition Speed (Seconds to reach threshold)

|**Optimizer**|**90% Acc**|**91% Acc**|**92% Acc**|**93% Acc**|**94% Acc**|**95% Acc**|
|---|---|---|---|---|---|---|
|**AdamW**|413.20|509.43|733.91|1524.13|——|——|
|**SGD-M**|1941.83|2040.61|2199.11|2378.02|3009.72|3193.44|
|**SGD-Jumper**|**303.53**|**381.19**|**520.44**|**1214.50**|**2386.52**|**3414.65**|

## 🚀 Run the experiment

To replicate these benchmark results and track weight evolution paths locally, execute the pipeline wrapper:
```bash
git clone https://github.com/jrnjf/sgd_jumper.git

cd sgd-jumper

python3 jumper_experiment.py
```