
# Copyright (c) Tencent Inc. All rights reserved.
import os
import os.path as osp
import json
import tempfile
from typing import List, Optional, Dict
from collections import defaultdict
from mmengine.fileio import get_local_path
from mmdet.datasets.coco import CocoDataset
from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
@DATASETS.register_module()
class VGGroundingDataset(BatchShapePolicyDataset, CocoDataset):
    """支持多图片目录的Visual Genome数据集"""

    METAINFO = {
        'classes': (),
        'palette': [(220, 20, 60)]
    }

    def __init__(
            self,
            ann_file: str,
            img_info_file: str,
            data_root: str = '',  # 新增data_root参数
            data_prefix: Dict[str, List[str]] = dict(img=['VG_100K', 'VG_100K_2']),
            class_text_path: Optional[str] = None,
            max_classes: int = 200,
            test_mode: bool = False,
             ** kwargs
    ):
        # === 初始化路径参数 ===
        self.data_root = data_root
        self._raw_ann_file = ann_file
        self.img_info_file = img_info_file
        self.data_prefix = data_prefix
        self.class_text_path = class_text_path
        self.max_classes = max_classes

        # === 核心初始化流程 ===
        self._init_synonym_rules()
        self._load_image_metadata()
        self._process_class_texts()
        self._build_categories()
        self._coco_temp_file = self._create_coco_format_file()

        # === 父类初始化 ===
        super().__init__(
            ann_file=self._coco_temp_file,
            data_prefix={'img': ''},  # 路径已包含在COCO数据中
            data_root=self.data_root,
            test_mode=test_mode,
         ** kwargs
        )

        # === 清理临时文件 ===
        os.remove(self._coco_temp_file)

    def _init_synonym_rules(self):
        """初始化同义词映射规则"""
        synonym_rules = {
            'person': ['man', 'woman', 'boy', 'girl', 'human'],
            'car': ['auto', 'automobile', 'sedan', 'truck', 'van'],
            'dog': ['puppy', 'doggy'],
            'tree': ['palm', 'oak', 'pine'],
            'building': ['house', 'skyscraper']
        }
        self.reverse_synonym = {
            alias: main for main, aliases in synonym_rules.items()
            for alias in aliases
        }

    def _load_image_metadata(self):
        """加载图像元数据"""
        with open(self.img_info_file, 'r', encoding='utf-8') as f:
            self.img_metas = {img['id']: img for img in json.load(f)}
        print(f"已加载 {len(self.img_metas)} 张图像元数据")

    def _process_class_texts(self):
        """处理文本数据"""
        if self.class_text_path:
            with open(self.class_text_path, 'r') as f:
                data = json.load(f)
                assert isinstance(data, list), "文本文件格式错误"
                self.class_texts = [item[0].strip() for item in data]
            assert len(self.class_texts) == self.max_classes, "类别数量不匹配"
        else:
            self.class_texts = None

    def _build_categories(self):
        """构建类别系统"""
        if self.class_text_path:
            # 静态加载模式
            self.cat2id = {name: idx + 1 for idx, name in enumerate(self.class_texts)}
        else:
            # 动态统计模式
            category_counter = defaultdict(int)
            with open(self._raw_ann_file, 'r') as f:
                for img_info in json.load(f):
                    for obj in img_info.get('objects', []):
                        raw_name = obj['names'][0].strip().lower()
                        mapped_name = self.reverse_synonym.get(raw_name, raw_name)
                        category_counter[mapped_name] += 1

            sorted_cats = sorted(category_counter.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:self.max_classes]
            self.class_texts = [k for k, _ in sorted_cats]
            self.cat2id = {k: idx + 1 for idx, (k, _) in enumerate(sorted_cats)}

        self.id2cat = {v: k for k, v in self.cat2id.items()}

    def _find_image_path(self, filename: str) -> Optional[str]:
        """多路径查找图片文件"""
        # 统一处理路径格式
        img_prefixes = []
        for prefix in self.data_prefix['img']:
            full_path = osp.join(self.data_root, prefix)
            if osp.isdir(full_path):
                img_prefixes.append(full_path)

        # 尝试所有可能路径
        for prefix in img_prefixes:
            candidate = osp.join(prefix, filename)
            if osp.exists(candidate):
                return candidate
        return None

    def _create_coco_format_file(self) -> str:
        """生成COCO格式文件（适配多路径）"""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": cid, "name": name}
                for name, cid in self.cat2id.items()
            ]
        }

        ann_id = 0
        with open(self._raw_ann_file, 'r') as f:
            for img_info in json.load(f):
                img_id = img_info['id']
                img_meta = self.img_metas.get(img_id)
                if not img_meta:
                    continue

                # 查找实际图片路径
                filename = osp.basename(img_meta['url'])
                img_path = self._find_image_path(filename)
                if not img_path:
                    continue

                # 添加图片信息（保存相对路径）
                coco_data["images"].append({
                    "id": img_id,
                    "width": img_meta['width'],
                    "height": img_meta['height'],
                    "file_name": osp.relpath(img_path, self.data_root)
                })

                # 处理标注
                for obj in img_info.get('objects', []):
                    instance = self._parse_instance(obj, img_meta)
                    if not instance:
                        continue

                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": self.cat2id[instance['mapped_name']],
                        "bbox": instance['bbox'],
                        "area": instance['bbox'][2] * instance['bbox'][3],
                        "iscrowd": 0
                    })
                    ann_id += 1

        # 写入临时文件
        fd, temp_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(coco_data, f, indent=2)
        return temp_path

    def _parse_instance(self, obj: Dict, img_meta: Dict) -> Optional[Dict]:
        """解析单个实例"""
        try:
            # 坐标处理
            x = max(0, int(obj['x']))
            y = max(0, int(obj['y']))
            w = max(0, int(obj['w']))
            h = max(0, int(obj['h']))

            # 边界修正
            img_w, img_h = img_meta['width'], img_meta['height']
            x = min(x, img_w - 1)
            y = min(y, img_h - 1)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w * h < 10:
                return None

            # 类名处理
            raw_name = obj['names'][0].strip().lower()
            mapped_name = self.reverse_synonym.get(raw_name, raw_name)
            if mapped_name not in self.cat2id:
                return None

            return {
                "bbox": [x, y, w, h],
                "mapped_name": mapped_name
            }
        except Exception as e:
            print(f"解析异常: {str(e)}")
            return None

    def get_data_info(self, idx: int) -> dict:
        """注入文本信息"""
        data_info = super().get_data_info(idx)
        data_info['texts'] = self.class_texts
        return data_info

    def extra_repr(self) -> str:
        return (f"模式: {'静态' if self.class_text_path else '动态'}\n"
                f"类别数: {len(self.class_texts)}\n"
                f"有效图像: {len(self.data_list)}")


import os.path as osp
from typing import List, Dict
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.coco import CocoDataset
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class DynamicCocoDataset(CocoDataset):
    """支持动态类别映射的COCO数据集，继承自CocoDataset但覆盖类别加载逻辑"""

    def __init__(self, *args,  ** kwargs):
        # 确保在初始化时加载metainfo
        if 'metainfo' not in kwargs:
            kwargs['metainfo'] = {}
        super().__init__(*args,  ** kwargs)

    def _load_metainfo(self, metainfo: Optional[dict] = None) -> dict:
            """动态加载元信息，覆盖父类的硬编码类别"""
            # 确保metainfo是字典
            metainfo = metainfo or {}

            # 从标注文件加载实际类别
            with get_local_path(self.ann_file) as local_path:
                with open(local_path) as f:
                    data = json.load(f)

            # 按ID排序确保顺序一致
            categories = sorted(data['categories'], key=lambda x: x['id'])
            custom_classes = [cat['name'] for cat in categories]

            # 合并用户指定的metainfo和自动生成的metainfo
            result_metainfo = {
                'classes': metainfo.get('classes', custom_classes),
                'palette': metainfo.get('palette', self._generate_palette(len(custom_classes))),
            }

            # 保留其他用户指定的元信息
            for k, v in metainfo.items():
                if k not in ['classes', 'palette']:
                    result_metainfo[k] = v

            return result_metainfo

    def _generate_palette(self, num_classes: int) -> list:
            """动态生成调色板"""
            base_palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
            if num_classes <= len(base_palette):
                return base_palette[:num_classes]

            # 对于更多类别，使用HSV色彩空间生成
            import colorsys
            return base_palette + [
                tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / (num_classes - len(base_palette)), 0.8, 0.8))
                for i in range(num_classes - len(base_palette))
            ]

    def load_data_list(self) -> List[dict]:
            """重写数据加载，确保metainfo已初始化"""
            if not hasattr(self, '_metainfo') or self._metainfo is None:
                self._metainfo = self._load_metainfo()

            return super().load_data_list()

@DATASETS.register_module()
class MultiDirCocoDataset(DynamicCocoDataset):
        """支持多目录搜索的动态COCO数据集"""

        def __init__(self, img_dirs: List[str] = None,  ** kwargs):
            self.img_dirs = img_dirs or []
            super().__init__(**kwargs)

        def parse_data_info(self, raw_data_info: dict) -> dict:
            """重写路径解析逻辑，支持多目录搜索"""
            data_info = super().parse_data_info(raw_data_info)
            original_path = data_info['img_path']

            # 如果原始路径存在，直接返回
            if osp.exists(original_path):
                return data_info

            # 在指定目录中搜索
            filename = osp.basename(original_path)
            for dir_name in self.img_dirs:
                candidate_path = osp.join(self.data_prefix['img'], dir_name, filename)
                if osp.exists(candidate_path):
                    data_info['img_path'] = candidate_path
                    return data_info

            # 检查是否直接放在data_root下
            default_path = osp.join(self.data_prefix['img'], filename)
            if osp.exists(default_path):
                data_info['img_path'] = default_path
                return data_info

            raise FileNotFoundError(
                f"无法找到图片文件: {filename}\n"
                f"搜索路径:\n- {original_path}\n"
                + "\n- ".join([osp.join(self.data_prefix['img'], d, filename) for d in self.img_dirs])
                + f"\n- {default_path}"
            )


