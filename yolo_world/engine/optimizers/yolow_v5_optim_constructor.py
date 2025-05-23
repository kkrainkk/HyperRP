# Copyright (c) Tencent Inc. All rights reserved.
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper, DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils import mmcv_full_available
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm

from mmyolo.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                             OPTIMIZERS)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOWv5OptimizerConstructor(DefaultOptimWrapperConstructor):
    """YOLO World v5 constructor for optimizers."""

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None) -> None:
        super().__init__(optim_wrapper_cfg, paramwise_cfg)
        self.base_total_batch_size = self.paramwise_cfg.pop(
            'base_total_batch_size', 64)

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = '',
                   is_dcn_module: Optional[Union[int, float]] = None) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', None)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', None)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', None)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', None)
        flat_decay_mult = self.paramwise_cfg.get('flat_decay_mult', None)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', None)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if bypass_duplicate and self._is_in(param_group, params):
                print_log(
                    f'{prefix} is duplicate. It is skipped since '
                    f'bypass_duplicate={bypass_duplicate}',
                    logger='current',
                    level=logging.WARNING)
                continue
            if not param.requires_grad:
                params.append(param_group)
                continue

            # if the parameter match one of the custom keys, ignore other rules
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    # add custom settings to param_group
                    for k, v in custom_keys[key].items():
                        param_group[k] = v
                    break

            # NOTE: the behavious is different from MMDetection
            # bias_lr_mult affects all bias parameters
            # except for norm.bias dcn.conv_offset.bias
            if name == 'bias' and not (
                    is_norm or is_dcn_module) and bias_lr_mult is not None:
                param_group['lr'] = self.base_lr * bias_lr_mult

            if (prefix.find('conv_offset') != -1 and is_dcn_module
                    and dcn_offset_lr_mult is not None
                    and isinstance(module, torch.nn.Conv2d)):
                # deal with both dcn_offset's bias & weight
                param_group['lr'] = self.base_lr * dcn_offset_lr_mult

            # apply weight decay policies
            if self.base_wd is not None:
                # norm decay
                if is_norm and norm_decay_mult is not None:
                    param_group[
                        'weight_decay'] = self.base_wd * norm_decay_mult
                # bias lr and decay
                elif (name == 'bias' and not is_dcn_module
                      and bias_decay_mult is not None):
                    param_group[
                        'weight_decay'] = self.base_wd * bias_decay_mult
                # depth-wise conv
                elif is_dwconv and dwconv_decay_mult is not None:
                    param_group[
                        'weight_decay'] = self.base_wd * dwconv_decay_mult
                # flatten parameters except dcn offset
                elif (param.ndim == 1 and not is_dcn_module
                      and flat_decay_mult is not None):
                    param_group[
                        'weight_decay'] = self.base_wd * flat_decay_mult
            params.append(param_group)
            for key, value in param_group.items():
                if key == 'params':
                    continue
                full_name = f'{prefix}.{name}' if prefix else name
                print_log(
                    f'paramwise_options -- {full_name}:{key}={value}',
                    logger='current')

        if mmcv_full_available():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, 'module'):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        optimizer_cfg = self.optimizer_cfg.copy()

        # follow the original yolov5 implementation
        if 'batch_size_per_gpu' in optimizer_cfg:
            batch_size_per_gpu = optimizer_cfg.pop('batch_size_per_gpu')
            # No scaling if total_batch_size is less than
            # base_total_batch_size, otherwise linear scaling.
            total_batch_size = get_world_size() * batch_size_per_gpu
            accumulate = max(
                round(self.base_total_batch_size / total_batch_size), 1)
            scale_factor = total_batch_size * \
                accumulate / self.base_total_batch_size

            if scale_factor != 1:
                weight_decay = optimizer_cfg.get('weight_decay', 0)
                weight_decay *= scale_factor
                optimizer_cfg['weight_decay'] = weight_decay
                print_log(f'Scaled weight_decay to {weight_decay}', 'current')

        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # set param-wise lr and weight decay recursively
            params: List = []
            self.add_params(params, model)
            optimizer_cfg['params'] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper



from torch.optim import Optimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class UniqueParamOptimConstructor(DefaultOptimWrapperConstructor):
    """确保参数不重复的优化器构造器"""

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None) -> None:
        super().__init__(optim_wrapper_cfg, paramwise_cfg)
        self.used_param_ids = set()

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = '',
                   is_dcn_module: Optional[Union[bool, nn.Module]] = None) -> None:
        # 遍历当前模块的参数
        for name, param in module.named_parameters(recurse=False):
            param_id = id(param)
            if param_id in self.used_param_ids:
                continue
            self.used_param_ids.add(param_id)
            # 将参数封装为张量列表
            param_group = {'params': [param]}  # 🔴 每个参数单独成组
            params.append(param_group)
        # 递归处理子模块
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix)

    def __call__(self, model: nn.Module) -> OptimWrapper:
        params: List[dict] = []
        self.add_params(params, model)
        optimizer_cfg = self.optimizer_cfg.copy()
        # 直接传递参数组列表（非嵌套）
        optimizer_cfg['params'] = params
        optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        return OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))


from typing import Dict, List, Set, Optional
import torch.nn as nn
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SafeYOLOWOptimConstructor(DefaultOptimWrapperConstructor):
    """安全版优化器构造器（支持参数共享）"""

    def __init__(self, optim_wrapper_cfg: Dict, paramwise_cfg: Optional[Dict] = None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg)

        # 全局参数跟踪系统（基于参数ID）
        self._param_registry: Set[int] = set()

    def add_params(self, params: List[Dict], module: nn.Module, prefix: str = ''):
        """改进的递归参数添加方法"""
        # 处理当前模块参数（带完整路径）
        for name, param in module.named_parameters(recurse=False):
            param_id = id(param)
            full_name = f"{prefix}.{name}" if prefix else name

            # 防御机制：全局参数ID检查
            if param_id in self._param_registry:
                print_log(f"忽略重复参数: {full_name} (已注册于其他路径)", 'current')
                continue
            self._param_registry.add(param_id)

            # 生成参数组配置
            param_group = self._create_param_group(param, full_name)
            if param_group:
                params.append(param_group)

        # 递归处理子模块（基于实例ID生成唯一路径）
        for child_name, child_module in module.named_children():
            child_id = id(child_module)
            child_prefix = f"{prefix}.{child_name}[{child_id}]" if prefix else f"{child_name}[{child_id}]"
            self.add_params(params, child_module, child_prefix)

    def _create_param_group(self, param: nn.Parameter, full_name: str) -> Optional[Dict]:
        """创建参数组配置（支持冻结）"""
        # 示例：冻结文本模型参数
        if 'text_model' in full_name:
            param.requires_grad = False
            return None

        # 默认配置
        return {
            'params': [param],
            'lr': self.base_lr,
            'weight_decay': self.base_wd if 'bias' not in full_name else 0.0
        }

    def __call__(self, model: nn.Module) -> OptimWrapper:
        # 重置注册表
        self._param_registry.clear()

        # 构建参数组
        params = []
        self.add_params(params, model)

        # 构建优化器
        optim_cfg = self.optimizer_cfg.copy()
        optim_cfg['params'] = params
        return OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg.copy(),
            default_args={'optimizer': OPTIMIZERS.build(optim_cfg)}
        )



