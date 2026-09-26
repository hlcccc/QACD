# QACD — 问题条件化原子主张分解

**面向 LVLM 视觉问答的回答级错误风险后处理评分**

[![tests](https://img.shields.io/badge/tests-176%20passed-brightgreen)](#快速开始)
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

## 特征工程

`frozen/` 是研究流水线特征工程的逐行移植：

```
frozen/constants.py   特征族名称表（逐字转录，顺序即冻结系数所绑定顺序）
frozen/features.py    派生规则：方向对齐 → 矛盾感知 v2 → QACD 感知 → 直接验证器
frozen/build.py       划分、特征选择、中位数填补、矩阵装配
frozen/assemble.py    逐条 claim 装配 112 维特征向量
```

所有派生规则都是逐行的，因此单条 claim 可以独立装配 —— 这是该特征集能用于在线打分、
而不只是批处理的原因。

> 移植中唯一的有意偏离：研究代码里有两个**同名但行为不同**的 `_num` ——
> `run_bew_bcm_v2._num` 不填补不截断（`add_contradiction_aware_features` 依赖
> 它在缺失时返回 NaN），`run_qacd_direct_verifier_benchmark._num` 会填补并截断到
> `[0,1]`。合并到同一模块后必须分开，本仓库保存为 `_num_bew` / `_num_dvb`，
> `tests/test_frozen_features.py` 有专门测试锁住这个区别。

## 分解版本 v1 与 v2

产出报告数值的是 **v1**；`qacd/decompose.py`（仓库默认）实现的是 **v2**。
两者不只是标签不同：

| | v1（产出结果的版本） | v2（当前默认） |
|---|---|---|
| 回退切分 | 仅句子 / 小句 | 增加二级小句、逗号、22 词分块、去重 |
| 类型判定 | 用 `question + 条件化 claim`，数字优先 | 用 source span，claim 优先 |
| `is_atomic` 阈值 | 0.75 | 0.9 |
| 引号问句 | 计入长度惩罚 | 剥离后再算 |
| claim 集合校验 | 无 | 覆盖率 / 原子性 / 重复校验 |

**两个版本都在仓库里**：`qacd/decompose.py`（v2，默认）与 `qacd/decompose_v1.py`
（v1，逐字移植自升级提交 `f03b27a^`）。每一项差异都有测试锁住
（`tests/test_decompose_v1.py`）。

## 实验数据与结果

**本仓库不包含实验数据，也不包含结果数值。**

冻结特征矩阵、原始证据表与结果表属于项目交付物，按要求不在本仓库公开。这里保留的是
**方法与可运行的代码**。要运行下面这些检查，需要先从研究服务器导出数据：

```bash
python tools/export_raw_evidence.py    # 原始证据表        -> artifacts/raw/
python tools/export_bundle.py          # 特征矩阵与参考分数 -> artifacts/
```

| 脚本 | 验证内容 |
|---|---|
| `scripts/run_frozen_evaluation.py --from-raw` | 从原始证据重建特征 → 校准 → 聚合 → 指标 |
| `scripts/verify_feature_port.py` | 112 维特征与冻结矩阵逐元素对齐 |
| `scripts/verify_decomposition_v1.py` | v1 分解复现冻结 claim table |
| `scripts/verify_pipeline_frozen.py` | `QACDPipeline` 端到端闭环 |

**缺少数据时的行为**：以上脚本以退出码 **2** 明确报出缺失项与恢复步骤，不会抛栈、
也不会在零输入上报告成功。测试中依赖数据的 14 项显示为 `skipped`
（`pytest -rs` 会打印原因），其余 162 项不需要任何数据。

数据到位后，`QACDPipeline` 可以直接用冻结特征集打分：

```python
from frozen.assemble import FrozenFeatureAssembler
from qacd.pipeline import QACDPipeline

pipeline = QACDPipeline(assembler=FrozenFeatureAssembler.from_matrices(matrices))
pipeline.fit_matrix(matrices.dev_matrix, labels, matrices.train_mask)
claim_risk = pipeline.score_evidence_frame(test_frame)   # 每条 claim 一个风险分
```

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

python -m pytest -q                  # 162 passed + 14 skipped（跳过项需实验数据）
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

## 代码完成度

| 组件 | 状态 |
|---|---|
| 决策层（分解 / 校准 / 聚合 / MVR / 保形） | 完整，162 项离线测试覆盖 |
| 特征层 `frozen/`（112 维冻结特征工程） | 完整；逐元素对齐检查在数据到位后可运行 |
| 分解 v1（`qacd/decompose_v1.py`） | 完整；冻结 claim table 复现检查同上 |
| 分解 v2（`qacd/decompose.py`） | 完整，仓库默认；**不是**产出报告数值的版本 |
| 评估层（证据包校验 / 指标 / 配对 bootstrap） | 完整 |
| `qacd/features.py`（48 维轻量参考路径） | 仅供离线演示，**不是**产出报告数值的特征集 |
| `MockProvider`（离线确定性桩） | 完整，用于链路自测 |
| `RapidOCRProvider` | 完整，需 `rapidocr-onnxruntime` |
| `LLaVAProvider`（LLaVA-1.5-13B 证据提供方） | **完整实现**：四类提示、类型化探针、yes/no 解析、logprob 置信度、K 次采样。模型调用点 `_generate_with_scores` 可注入，整套逻辑在 `tests/test_provider.py` 中无权重跑通 |
| HTTP 服务层 | 端点实现完整，需 `fastapi` 额外依赖 |

**本仓库尚未验证的一件事**：用真实 LLaVA-1.5-13B 权重把评分链路端到端跑一遍。
`LLaVAProvider` 的代码是完整的，但开发环境没有 GPU 与权重，因此那一次真实推理尚未发生。

## 文档

| 文档 | 内容 |
|---|---|
| [01-方法说明](docs/01-方法说明.md) | 方法、流程与结果 |
| [02-接口文档](docs/02-接口文档.md) | 四个对接端点的入参/出参契约 |
| [03-部署与资源需求](docs/03-部署与资源需求.md) | 依赖、显存、调用次数、开发集规模、部署步骤 |
| [04-平台对接说明](docs/04-平台对接说明.md) | 对接边界、验收标准、监控与 FAQ |
| [05-课题一技术对接表](docs/05-课题一技术对接表.md) | 技术统计表（Word 版镜像） |

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
├── frozen/                冻结特征工程（112 维）
├── evaluation/            评估层：证据包校验、指标、配对 bootstrap
├── tools/                 服务器侧导出脚本（证据包 / 原始证据表）
├── configs/               参考配置与接口 schema
├── docs/                  方法、接口、部署、对接说明
├── scripts/               离线演示 + 数据到齐后的验证脚本
└── tests/                 176 项测试（162 离线 + 14 需数据）
```

## 引用

见 [CITATION.cff](CITATION.cff)。论文尚未投稿，暂无公开链接。

## 许可

专有许可，见 [NOTICE.md](NOTICE.md)。本项目为课题一交付件，
未经授权不得对外分发。
