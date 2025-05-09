# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from yolo_world.models.necks.yolo_world_pafpn import YOLOWorldPAFPN


class HyperVisualizer:
    """特征图可视化器，保持原始风格"""

    def __init__(self):
        self.feats = []  # 存储特征图
        self.feat_names = []  # 存储特征名称

    def _process_feature(self, feat):
        """处理特征图，保持原始方式"""
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()

        # 保持原始维度处理
        if feat.ndim == 4:  # [B,C,H,W]
            feat = feat[0]  # 取第一个样本
        if feat.ndim == 3:  # [C,H,W]
            feat = feat.mean(axis=0)  # 保持通道平均
        elif feat.ndim == 2:  # [H,W]
            pass
        else:
            raise ValueError(f"不支持的维度: {feat.shape}")
        return feat

    def _normalize_feature(self, feat):
        """归一化特征图，保持原始方式"""
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-9)
        return np.uint8(255 * feat)

    def draw_features(self, save_path):
        """绘制特征图，保持原始可视化风格"""
        try:
            # 处理两个特征图
            input_feat = self._process_feature(self.feats[0])  # Hyper输入
            output_feat = self._process_feature(self.feats[1])  # Hyper输出

            # 归一化
            img1 = self._normalize_feature(input_feat)
            img2 = self._normalize_feature(output_feat)

            # 应用颜色映射（保持VIRIDIS）
            viz1 = cv2.applyColorMap(img1, cv2.COLORMAP_VIRIDIS)
            viz2 = cv2.applyColorMap(img2, cv2.COLORMAP_VIRIDIS)

            # 创建纯黑背景（保持原始风格）
            max_height = max(viz1.shape[0], viz2.shape[0])
            black_bg = np.zeros((max_height, viz1.shape[1] + viz2.shape[1], 3), dtype=np.uint8)

            # 放置特征图（保持居中）
            y_offset1 = (max_height - viz1.shape[0]) // 2
            y_offset2 = (max_height - viz2.shape[0]) // 2
            black_bg[y_offset1:y_offset1 + viz1.shape[0], :viz1.shape[1]] = viz1
            black_bg[y_offset2:y_offset2 + viz2.shape[0], viz1.shape[1]:] = viz2

            # 保持原始无标注风格
            # 如需添加标注可取消以下注释
            # cv2.putText(black_bg, "Hyper Input", (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # cv2.putText(black_bg, "Hyper Output", (viz1.shape[1] + 10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imwrite(save_path, black_bg)
            print(f"特征图已保存至：{save_path}")

        except IndexError:
            print("错误：未捕获到足够的特征图")
        except Exception as e:
            print(f"可视化失败：{str(e)}")


def inference(model, image_path, texts, test_pipeline):
    visualizer = HyperVisualizer()

    # 查找YOLOWorldPAFPN模块
    yolo_pafpn = None
    for name, module in model.named_modules():
        if isinstance(module, YOLOWorldPAFPN):
            yolo_pafpn = module
            print(f"找到PAFPN模块：{name}")
            break

    if not yolo_pafpn:
        raise RuntimeError("未找到YOLOWorldPAFPN模块")

    # 注册钩子
    hook_handles = []

    def forward_hook(module, inputs, outputs):
        """捕获forward的img_feats[0]"""
        if isinstance(outputs, tuple) and len(outputs) >= 1:
            visualizer.feats.append(outputs[0].clone())  # img_feats[0]
            print(f"捕获img_feats[0]形状：{outputs[0].shape}")

    def hyper_hook(module, inputs, outputs):
        """捕获 YOLOWorldPAFPN 的输入 img_feats[0]"""
        img_feats = inputs[0]  # inputs是 (img_feats, txt_feats)
        if isinstance(img_feats, (list, tuple)) and len(img_feats) > 0:
           visualizer.feats.append(img_feats[0].clone())
           print(f"捕获 img_feats[0] 形状：{img_feats[0].shape}")

    # 修改点：替换为forward hook
    hook_handles.append(yolo_pafpn.register_forward_hook(hyper_hook))
    hook_handles.append(yolo_pafpn.register_forward_hook(forward_hook))
    #hook_handles.append(yolo_pafpn.register_forward_hook(hyper_hook))

    try:
        # 读取图像（保持原始流程）
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")

        # 预处理流程（保持原始流程）
        data_info = test_pipeline(dict(
            img=cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),
            img_id=0,
            texts=texts
        ))

        # 执行推理（保持原始流程）
        with torch.no_grad():
            output = model.test_step({
                'inputs': data_info['inputs'].unsqueeze(0),
                'data_samples': [data_info['data_samples']]
            })[0]

        # 生成可视化（输出路径保持原始格式）
        save_path = osp.splitext(image_path)[0] + '_hyper_features.jpg'
        visualizer.draw_features(save_path)

        # 后处理检测结果（保持原始流程）
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores > 0.3]

        return (
            pred_instances.bboxes.cpu().numpy(),
            pred_instances.labels.cpu().numpy(),
            [texts[x][0] for x in pred_instances.labels.cpu().numpy()],
            pred_instances.scores.cpu().numpy()
        )

    finally:
        # 清理钩子（保持原始流程）
        for handle in hook_handles:
            handle.remove()



if __name__ == "__main__":
    # 配置路径（保持原始路径格式）
    config_path = "D:/YOLO-World-master/configs/finetune_coco/OVOD_local.py"
    checkpoint_path = "C:/Users/86198/Desktop/epoch_1.pth"
    test_image = "D:/YOLO-World-master/demo/sample_images/2371791.jpg"

    # 初始化模型（保持原始流程）
    cfg = Config.fromfile(config_path)
    cfg.work_dir = osp.join('./work_dirs')
    model = init_detector(cfg, checkpoint_path, device='cuda:0')

    # 配置数据流水线（保持原始流程）
    pipeline_cfg = get_test_pipeline_cfg(cfg)
    pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(pipeline_cfg)

    # 执行推理（保持原始调用方式）
    print(f"开始检测：{test_image}")
    try:
        boxes, labels, texts, scores = inference(
            model, test_image,
            texts=[['person'], ['man'], ['head']],
            test_pipeline=test_pipeline
        )

        # 打印结果（保持原始格式）
        print("\n检测结果：")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            print(f"目标{i + 1}: {texts[i]}({label}) 置信度：{score:.2f} 坐标：{box.astype(int)}")

    except Exception as e:
        print(f"检测错误：{str(e)}")