import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from peft.tuners import lora
from peft.tuners.lora import LoraModel
from pathlib import Path


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
                sensitivity_A = torch.abs(self.lora_A['default'].weight * grad_A) #Подсчет чувствительности на шаге
                self.sensitivity_A = self.beta * self.sensitivity_A + (1 - self.beta) * sensitivity_A #Усреднение
            if grad_B is not None:
                sensitivity_B = torch.abs(self.lora_B['default'].weight * grad_B)
                self.sensitivity_B = self.beta * self.sensitivity_B + (1 - self.beta) * sensitivity_B

    def apply_sparsity(self, tau):
        self.update_sensitivity()
        with torch.no_grad():
            A_threshold = torch.quantile(self.sensitivity_A, 1 - tau, dim=1, keepdim=True) #отбираем лучшие веса по одной размерности
            self.lora_A['default'].weight[self.sensitivity_A < A_threshold] = 0 #остальные веса обнуляются

            B_threshold = torch.quantile(self.sensitivity_B, 1 - tau, dim=0, keepdim=True)
            self.lora_B['default'].weight[self.sensitivity_B < B_threshold] = 0 


class RoseLoraModel(LoraModel):
    '''
    Модель с примененным алгоритмом RoseLoRA
        
    '''
    def __init__(self, base_model, config, adapter_name="default", total_step = 1000, sparsity=0.1, beta=0.8): 
        super().__init__(base_model, config, adapter_name) #инициализируем как обычную LoRA
        self.adapter_name = adapter_name
        self.total_iterations = total_step
        self.sparsity = sparsity
        self.beta = beta
        self.t_i = int(0.1 * self.total_iterations)  # 10% от T - прожиг
        self.t_f = int(0.8 * self.total_iterations)  # 80% от T - точка окончания процедуры разреживания
        self.roseLora_modules = []
        self.replace_svd_linear_with_sparse(self.model.bert, beta=self.beta)

    def replace_svd_linear_with_sparse(self, model, beta=0.8):
        '''
        Переопределяет все слои LoRA как слои RoseLoRA
        
        """
        Аргументы:
        
            model - модель, к которой необходимо применить RoseLoRA
        
            beta (default = 0.8) - параметр сглаживания чувствительности весов по итерациям
        
        '''
        for name, module in model.named_children():
            if isinstance(module, lora.layer.Linear):
                r = module.r[self.adapter_name] if isinstance(module.r, dict) else module.r
                lora_alpha = module.lora_alpha[self.adapter_name] if isinstance(module.lora_alpha, dict) else module.lora_alpha
                dropout = module.lora_dropout['default'].p

                wrapper_layer = SparseSVDLinear(
                    module.base_layer,
                    self.adapter_name,
                    beta=beta,
                    in_features=module.base_layer.in_features,
                    out_features=module.base_layer.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    dropout=dropout,
                )
                setattr(model, name, wrapper_layer)
                self.roseLora_modules.append(wrapper_layer)
            else:
                self.replace_svd_linear_with_sparse(module, beta=beta)

    def update_and_allocate(self, step, visualize_weights = True):
        '''
        Итерация обновления весов: вычисление бюджета ненулевых весов, разреживание необходимых матриц
        '''
        tau = self.get_sparsity_budget(step)
        i = 0
        for module in self.roseLora_modules:
            if isinstance(module, SparseSVDLinear):
                i += 1
                module.apply_sparsity(tau)
                
        if step % 10 == 0 and visualize_weights: #сохранение картинок для визуализации
            with torch.no_grad():
                first_layer = self.model.bert.encoder.layer[0].attention.self.query
                lora_A = first_layer.lora_A['default'].weight
                lora_B = first_layer.lora_B['default'].weight

                plt.spy((lora_B @ lora_A).detach().cpu())
                Path("Plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(f'Plots/it_{step}.png')

    def get_sparsity_budget(self, step):
        '''
        Вычисление необходимой доли ненулевых параметров на каждом шаге
        '''
        if step <= self.t_i:
            return 1
        if step <= self.t_f:
            return self.sparsity + (1 - self.sparsity) * (1 - (step - self.t_i) / (self.t_f - self.t_i))**3
        return self.sparsity