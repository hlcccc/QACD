# QACD — 问题条件化原子主张分解

**面向 LVLM 视觉问答的回答级错误风险后处理评分**

[![tests](https://img.shields.io/badge/tests-62%20passed-brightgreen)](#快速开始)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-proprietary-lightgrey)](NOTICE.md)

---

QACD（Question-Conditioned Atomic Claim Decomposition）是一条**后处理**的回答错误风险
评分流程：不重训、不改动被评估的视觉语言模型，把冻结模型的回答连同问题与图像作为输入，
输出 `[0,1]` 的**回答级错误风险分数**与逐主张风险明细。适用于预警、人工复核分流与拒答决策。

```python
from qacd import QACDPipeline, QACDConfig

pipeline = QACDPipeline(config=QACDConfig(k=3))   # k>0 启用 MVR 通道
result = pipeline.score(
    question="What brand is the camera?",
    answer="Dakota digital",
    image="demo.jpg",
)
print(result.risk_score, result.is_high_risk, result.num_claims)
```

```
QACD offline demo
  question : What brand is the camera?
  answer   : Dakota digital
  ocr      : ['Dakota Digital', 'open 24 days']
  risk     : 0.2798  (high_risk=False)
  channels : {'evidence_score': 0.279777, 'mvr_unsupported_rate': 0.333333, ...}
```

## 主要结果

冻结 TextVQA 测试集：**1,999 条回答 / 1,278 张开发未见图像 / 836 条失败（41.8%）**

![主结果与配对对比](docs/assets/fig_main_forest.png)

| 方法 | 回答级 AUROC | 图像级 AUROC | 备注 |
|---|---:|---:|---|
| UMPIRE K=5 | 0.8564 | 0.8561 | 白盒，需要模型内部量 |
| **QACD（LM + 机械仪器 + MVR K=5）** | **0.8526** | — | **黑盒** |
| SelfCheckGPT-NLI K=5 | 0.8319 | 0.8477 | 公开基线 |
| Multi-sample Consistency K=5 | 0.8258 | 0.8363 | 公开基线 |
| QACD（原始 95 维） | 0.7991 | — | 早期配置 |

配对图像级 bootstrap（5,000 次）：

| 对比 | ΔAUROC | 95% CI | 判读 |
|---|---:|---|---|
| vs UMPIRE K=5（白盒） | −0.0039 | [−0.0195, +0.0124] | **统计持平** |
| vs SelfCheckGPT-NLI K=5 | +0.0207 | [+0.0028, +0.0391] | 领先 |
| vs Multi-sample Consistency K=5 | +0.0267 | [+0.0102, +0.0436] | 领先 |

**黑盒 QACD 与需要模型内部量的白盒方法统计持平，并优于两个公开采样类基线。**

评估设置：冻结 TextVQA 测试集（1,999 条回答 / 1,278 张开发未见图像 / 836 条失败），
配对图像级 bootstrap 5,000 次，随机种子固定。全部校准器仅用开发集拟合。

## 方法一览

| 阶段 | 内容 | 代码 |
|---|---|---|
| ① | 问题条件化原子主张分解（受检 LLM + 确定性规则回退） | `qacd/decompose.py` |
| ② | 信念一致性视图（4 视图） | `qacd/features.py` |
| ③ | 类型化直接验证（按主张类型路由证据探针） | `qacd/features.py` |
| ④ | 证据-主张对齐（**声明的无独立增益组件**） | `qacd/align.py` |
| ⑤ | 机械 OCR 仪器通道（确定性文字核对，19 项特征） | `qacd/mechanical.py` |
| ⑥ | BIC-lite 风险校准（L2 逻辑回归，λ=0.05，仅 NumPy） | `qacd/calibrate.py` |
| ⑦ | 最大值算子聚合为主张→回答风险 | `qacd/aggregate.py` |
| ⑧ | MVR 多视图重采样一致性（K=3 性价比拐点） | `qacd/mechanical.py` |

详见 **[docs/01-方法说明.md](docs/01-方法说明.md)**。

## 快速开始

决策层**只依赖 NumPy**，CPU 可跑，无需 GPU 与模型权重：

```bash
git clone https://github.com/hlcccc/QACD.git && cd QACD
pip install -r requirements.txt

python -m pytest -q                  # 62 项测试，全部离线
python scripts/demo_offline.py       # 端到端离线演示（合成开发集）
qacd demo                            # 同上的 CLI 版本

# 用你自己的开发集拟合打分器（dev.jsonl: question/answer/image/failed/group）
qacd fit --data dev.jsonl --out scorer.json --k 3

# 起 HTTP 服务（四个对接端点）
pip install "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2.0"
qacd serve --scorer scorer.json --port 8080
```

`qacd fit` / `qacd serve` 默认使用 `MockProvider`（确定性、无模型）以便打通链路。
生产环境必须注入 `LLaVAProvider` + `RapidOCRProvider`，见 `qacd/providers.py`。

## 对接接口

| 端点 | 技术点 | 状态 |
|---|---|---|
| `POST /v1/qacd/risk` | ① 核心回答风险评分器 | ✅ 可交付 |
| `POST /v1/qacd/mvr` | ② MVR 多视图重采样一致性 | ✅ 可交付 |
| `POST /v1/qacd/mechanical` | ③ 机械 OCR 仪器通道 | ✅ |
| `POST /v1/qacd/select` | ④ 保形选择性预测与弃权 | ✅ |
| `GET /health` | 健康检查与打分器状态 | ✅ |

字段定义、告警语义与版本约定见 **[docs/02-接口文档.md](docs/02-接口文档.md)**。

## 文档

| 文档 | 内容 |
|---|---|
| [01-方法说明](docs/01-方法说明.md) | 方法、流程与结果 |
| [02-接口文档](docs/02-接口文档.md) | 四个对接端点的入参/出参契约 |
| [03-部署与资源需求](docs/03-部署与资源需求.md) | 依赖、显存、调用次数、开发集规模、部署步骤 |
| [04-平台对接说明](docs/04-平台对接说明.md) | 对接边界、验收标准、监控与 FAQ |
| [05-课题一技术对接表](docs/05-课题一技术对接表.md) | 技术统计表（Word 版镜像） |
| [results/](results/README.md) | 冻结实验数值与溯源 |

## 项目结构

```
QACD/
├── qacd/                  决策层（NumPy only）
│   ├── types.py           数据容器与主张类型词表
│   ├── decompose.py       ① 问题条件化原子主张分解
│   ├── features.py        特征装配（顺序即冻结系数绑定）
│   ├── align.py           ④ 证据-主张对齐
│   ├── mechanical.py      ⑤ 机械 OCR 通道 + ⑧ MVR
│   ├── calibrate.py       ⑥ BIC-lite / RidgeLogistic（阻尼牛顿法）
│   ├── aggregate.py       ⑦ 最大值算子
│   ├── conformal.py       保形选择性预测（BH / BY）
│   ├── pipeline.py        端到端编排 + JSON 打分器导出
│   ├── providers.py       证据提供方（Mock / RapidOCR / LLaVA）
│   ├── service.py         HTTP 服务
│   └── cli.py             命令行入口
├── configs/               参考配置与接口 schema
├── docs/                  方法、接口、部署、对接说明
├── results/               冻结实验数值与溯源
├── scripts/               离线演示
└── tests/                 62 项测试
```

## 引用

见 [CITATION.cff](CITATION.cff)。论文尚未投稿，暂无公开链接。

## 许可

专有许可，见 [NOTICE.md](NOTICE.md)。本项目为课题一交付件，
未经授权不得对外分发。
