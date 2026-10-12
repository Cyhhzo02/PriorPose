"""GPU-only gradient reduction, same per-destination order as CPU reference.

Only initial cached permutation generation is on CPU; gradient/index tensors
stay on GPU. Floating-point sums have one writer and a fixed serial order.
Target: FP32 PointNet2 backwards. No silent fallback to atomic/CPU reductions.
"""
import hashlib
from functools import lru_cache
import torch


def extension():
    try:
        from pointnet2 import _ordered
    except ImportError as error:
        raise ImportError(
            "Build the PointNet++ extensions with "
            "'python -m pip install --no-build-isolation -v ./model/pointnet2'"
        ) from error
    return _ordered

class OrderedCudaBackward:
    def __init__(self,reduction_seed=None):
        if reduction_seed is not None and (type(reduction_seed) is not int or reduction_seed<0):
            raise ValueError('Invalid reduction seed')
        self.seed=reduction_seed
        self.ext=extension()

    @lru_cache(maxsize=32)
    def order(self,operation,length,count,device):
        if self.seed is None:
            return torch.arange(length,device=device,dtype=torch.long)
        key=f'priorpose-reduction-v1:{self.seed}:{operation}:{length}:{count}'
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
        generator=torch.Generator(device='cpu').manual_seed(seed)
        return torch.randperm(length,generator=generator,device='cpu').to(device)

    def reduce(self,grad,indices,count,operation,weight=None):
        if not grad.is_cuda or grad.dtype != torch.float32 or indices.device != grad.device:
            raise ValueError('Matching CUDA tensors and FP32 gradients required')
        if count<=0 or indices.shape[0]!=grad.shape[0]:
            raise ValueError('Invalid count/batch')
        idx=indices.reshape(indices.shape[0],-1).long()
        batch,length=idx.shape
        if length==0:
            return grad.new_zeros(batch,grad.shape[1],count)
        if idx.min().item()<0 or idx.max().item()>=count:
            raise ValueError('Index outside destination range')
        permutation=self.order(operation,length,count,str(grad.device))
        # Unique integer keys: no unstable-sort ties and no floating atomics.
        ranks=torch.arange(length,device=grad.device,dtype=torch.long)
        keys=idx.index_select(1,permutation)*length+ranks
        sorted_keys,sorted_ranks=torch.sort(keys,dim=1)
        positions=permutation[sorted_ranks].contiguous()
        boundaries=(torch.arange(count+1,device=grad.device,dtype=torch.long)*length).expand(batch,-1).contiguous()
        offsets=torch.searchsorted(sorted_keys,boundaries).contiguous()
        interpolation=weight is not None
        if interpolation:
            if weight.device!=grad.device or weight.dtype!=grad.dtype or weight.shape!=indices.shape or indices.shape[-1]!=3:
                raise ValueError('Invalid interpolation weights')
            weights=weight.contiguous()
        else:
            weights=grad.new_empty(0)
        return self.ext.reduce(grad.contiguous().reshape(batch,grad.shape[1],-1),positions,offsets,weights,count,interpolation)

    def gather(self,grad,idx,count):
        return self.reduce(grad,idx,count,'gather')
    def group(self,grad,idx,count):
        return self.reduce(grad,idx,count,'group')
    def interpolate(self,grad,idx,weight,count):
        return self.reduce(grad,idx,count,'interpolate',weight)

def install(pointnet_extension,reduction_seed=None):
    backend=OrderedCudaBackward(reduction_seed)
    pointnet_extension.gather_points_grad=backend.gather
    pointnet_extension.group_points_grad=backend.group
    pointnet_extension.three_interpolate_grad=backend.interpolate
    return backend
