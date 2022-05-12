ONNX

##### [Tutorial 8: Pytorch to ONNX (Experimental) — MMDetection 2.24.1 documentation](https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html)

- 完成onnx模型转换
- 实现后处理Topk以及NMS
- 可视化结果

!(https://github.com/fenfenfenfan/mmdet2onnx/blob/master/no_topk_nms_thr1.jpg?raw=true)

# 1.生成onnx文件（ATSS）

- [点击下载](https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth)，放在mmdetection/checkpoint中

- mmdetection/configs/atss/atss.py

#### 测试环境

```Python
# traffic
# 测试环境
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env tools/analysis_tools/benchmark.py configs/atss/atss.py checkpoint/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --max-iter 10 --log-interval 1 --launcher pytorch
```

#### atss2onnx

```Python
# 安装依赖
pip install onnx
pip install onnxruntime

# atss2onnx
# with postprocess
python tools/deployment/pytorch2onnx.py configs/atss/atss.py checkpoint/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --output-file checkpoint/atss_coco.onnx --input-img /mmdetection/coco/COCO_train2014_000000000009.jpg --test-img /mmdetection/coco/COCO_train2014_000000000009.jpg --shape 608 608 --verify --dynamic-export --cfg-options model.test_cfg.deploy_nms_pre=-1
# 去除post process
python tools/deployment/pytorch2onnx.py configs/atss/atss_coco.py checkpoint/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --output-file checkpoint/atss_r50_fpn_1x_coco_no_postprocess408.onnx --input-img data/traffic/task11_origin/data/train_images/13021_1647589563476.png --test-img data/traffic/task11_origin/data/train_images/13021_1647589563476.png --shape 408 408 --verify --dynamic-export --cfg-options model.test_cfg.deploy_nms_pre=-1 --skip-postprocess
# 只去除topk和nms
# 需要修改目标文件
# 运行指令去掉了--verify
python tools/deployment/pytorch2onnx.py configs/atss/atss.py checkpoint/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --output-file checkpoint/atss_coco_no_topk_nms.onnx --input-img /mmdetection/coco/COCO_train2014_000000000009.jpg --test-img /mmdetection/coco/COCO_train2014_000000000009.jpg --shape 608 608 --dynamic-export --cfg-options model.test_cfg.deploy_nms_pre=-1
```

#### no_topk_nms

- 为了生成no topk and nms的onnx模型，但是保留一部分后处理过程，需要修改源码(对应注释掉455-490，在514添加一句with_nms = False
- mmdetection/mmdet/models/dense_heads/base_dense_head.py  
- 与no postprocess区别在于：no postprocess.onnx直接输出head的所有每层输出，包括scores、pred boxes、centerness，并且没有经过box decoder以及sigmoid
- 如果不去掉--verify会报错，原因是验证时由于模型去掉了nms，onnx的输出label会变成[N,80]，而pytorch模型输出的label为[N,]

```Python
# 使用no postprocess.onnx进行推理,输出结果的shape，默认图片尺寸为608*608
len(raw_results) 15
0: (1, 80, 76, 76)
1: (1, 80, 38, 38)
2: (1, 80, 19, 19)
3: (1, 80, 10, 10)
4: (1, 80, 5, 5)
5: (1, 4, 76, 76)
6: (1, 4, 38, 38)
7: (1, 4, 19, 19)
8: (1, 4, 10, 10)
9: (1, 4, 5, 5)
10: (1, 1, 76, 76)
11: (1, 1, 38, 38)
12: (1, 1, 19, 19)
13: (1, 1, 10, 10)
14: (1, 1, 5, 5)
```

# 2.ONNX模型进行推理



```Python
# origin
python tools/onnx/onnx_with_postprocess.py
# manu-nms-topk
python tools/onnx/onnx_no_topk_nms.py
```