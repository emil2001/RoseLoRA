import matplotlib.pyplot as plt
from peft import LoraModel
from peft.tuners import lora
from peft.tuners.lora import LoraConfig, LoraModel
from .roselora_layer import SparseSVDLinear

class RoseLoraModel(LoraModel):
    def __init__(self, base_model, config, adapter_name="default", total_step = 1000, sparsity=0.1, beta=0.8):
        super().__init__(base_model, config, adapter_name)

        self.adapter_name = adapter_name
        self.total_iterations = total_step
        self.sparsity = sparsity
        self.beta = beta
        self.t_i = int(0.1 * self.total_iterations)  # 10% of T
        self.t_f = int(0.8 * self.total_iterations)  # 80% of T
        
        self.roseLora_modules = []
        self.replace_svd_linear_with_sparse(self.model, beta=self.beta)

    def replace_svd_linear_with_sparse(self, model, beta=0.8):
        for name, module in model.named_children():
            if isinstance(module, lora.layer.Linear):
                if isinstance(module.base_layer, lora.layer.Conv1D):
                    in_features = module.base_layer.weight.shape[0]
                    out_features = module.base_layer.weight.shape[1]
                else:
                    in_features = module.base_layer.in_features
                    out_features = module.base_layer.out_features

                r = module.r[self.adapter_name] if isinstance(module.r, dict) else module.r
                lora_alpha = module.lora_alpha[self.adapter_name] if isinstance(module.lora_alpha, dict) else module.lora_alpha
                dropout = module.lora_dropout['default'].p

                wrapper_layer = SparseSVDLinear(
                    module.base_layer,
                    self.adapter_name,
                    beta=beta,
                    in_features=in_features,
                    out_features=out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    dropout=dropout,
                )
                setattr(model, name, wrapper_layer)
                self.roseLora_modules.append(wrapper_layer)
            else:
                self.replace_svd_linear_with_sparse(module, beta=beta)

    def update_and_allocate(self, step):
        tau = self.get_sparsity_budget(step)
        i = 0
        for module in self.roseLora_modules:
            if isinstance(module, SparseSVDLinear):
                i += 1
                module.apply_sparsity(tau)

        print(f"------> Iteration: {step}, tau: {tau}")


    def apply_sparsity(self, step):
        # Budget of the percentage of remaining parameters at the t-th iteration
        tau = self.get_sparsity_budget(step)

        for module in self.roseLora_modules:
            if isinstance(module, SparseSVDLinear):
                
                module.apply_sparsity(tau)

    def get_sparsity_budget(self, step):
        if step <= self.t_i:
            return 1
        if step <= self.t_f:
            return self.sparsity + (1 - self.sparsity) * (1 - (step - self.t_i) / (self.t_f - self.t_i))**3
        return self.sparsity
            
