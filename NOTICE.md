# NOTICE

## 许可

Copyright (c) 2026 HLC. All rights reserved.

本项目为课题一交付件。**未经著作权人书面授权，不得复制、修改、
分发、公开传播或用于任何商业用途。**

代码中不包含任何第三方数据集、模型权重或私有标注文件。

## 第三方依赖

| 依赖 | 用途 | 许可 |
|---|---|---|
| NumPy | 决策层数值计算（必需） | BSD-3-Clause |
| FastAPI / Uvicorn / Pydantic | HTTP 服务（可选） | MIT / BSD-3-Clause / MIT |
| RapidOCR (onnxruntime) | 机械 OCR 通道（可选） | Apache-2.0 |
| PyTorch / Transformers | LLaVA 证据提供方（可选） | BSD-3-Clause / Apache-2.0 |

被评估的 LVLM 检查点（LLaVA-1.5-13B）及其许可条款由使用方自行确认，
本项目不再分发。

## 数据与资产

以下资产**不在**本仓库中，需由使用方自行获取：

- TextVQA / OK-VQA / VizWiz / MMStar 数据集原始图像
- LLaVA-1.5-13B 模型权重
- 冻结打分器二进制（joblib）与研究期 OCR / 生成缓存

研究期产出位于研究服务器的大容量数据分区下，体积为 GB 级；绝对路径已省略。
实验数值属于项目交付物，不在本仓库公开。

## 免责声明

本项目按"现状"提供。使用方应自行在实际数据分布上评估适用性后再投入生产。
