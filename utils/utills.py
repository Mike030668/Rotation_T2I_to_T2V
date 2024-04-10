import torch
import gc

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        for _ in range(3):
          torch.cuda.empty_cache()
          torch.cuda.ipc_collect()


def replace_in_memory(device, to_gpu = [], to_cpu = [], ):
    if len(to_cpu):
        for obj in to_cpu:
           if obj.device != 'cpu': obj.to('cpu')
    flush_memory()

    if len(to_gpu):
        for obj in to_gpu:
           if obj.device != 'gpu': obj.to(device)
    flush_memory()