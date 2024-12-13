import torch
import torch.nn as nn
from peft.tuners import lora
from transformers import BertForSequenceClassification

class SparseSVDLinear(lora.layer.Linear):
    def __init__(self, module, adapter_name, beta=0.8, *args, **kwargs):
        super().__init__(module, adapter_name, *args, **kwargs)
        self.beta = beta  # Коэффициент сглаживания
        
        sensitivity_A = torch.zeros_like(self.lora_A['default'].weight)
        sensitivity_B = torch.zeros_like(self.lora_B['default'].weight)
        self.register_buffer('sensitivity_A', sensitivity_A)
        self.register_buffer('sensitivity_B', sensitivity_B)
    def update_sensitivity(self):
        with torch.no_grad():
            grad_A = self.lora_A['default'].weight.grad
            grad_B = self.lora_B['default'].weight.grad
            if grad_A is not None:
                sensitivity_A = torch.abs(self.lora_A['default'].weight * grad_A)
                self.sensitivity_A = self.beta * self.sensitivity_A + (1 - self.beta) * sensitivity_A
            if grad_B is not None:
                sensitivity_B = torch.abs(self.lora_B['default'].weight * grad_B)
                self.sensitivity_B = self.beta * self.sensitivity_B + (1 - self.beta) * sensitivity_B

    def apply_sparsity(self, tau):
        self.update_sensitivity()
        with torch.no_grad():
            lora_A = self.lora_A['default'].weight
            lora_B = self.lora_B['default'].weight
            A_threshold = torch.quantile(self.sensitivity_A, 1 - tau, dim=1, keepdim=True)
            self.lora_A['default'].weight[self.sensitivity_A < A_threshold] = 0

            B_threshold = torch.quantile(self.sensitivity_B, 1 - tau, dim=0, keepdim=True)
            self.lora_B['default'].weight[self.sensitivity_B < B_threshold] = 0 
