# QACD — 问题条件化原子主张分解

**面向 LVLM 视觉问答的回答级错误风险后处理评分**

[![CI](https://github.com/hlcccc/QACD/actions/workflows/ci.yml/badge.svg)](https://github.com/hlcccc/QACD/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

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
print(result.risk_score, result.calibrated_confidence, result.is_high_risk, result.num_claims)
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

## 两条打分路径，以及报告数值出自哪一条

这是本仓库最容易踩错的一点：

| | 路径 A（实时） | 路径 B（冻结） |
|---|---|---|
| 入口 | `QACDPipeline.score(question, answer, image)` | `QACDPipeline.score_evidence_frame(evidence)` |
| 特征 | `qacd/features.py`，**48 列** | `frozen/`，**112 列** |
| HTTP 接口 | `POST /v1/qacd/risk`（响应中 `feature_dim: 48`） | 不对外提供 |
| 是否产出报告数值 | **不是** | **是** |

**平台调 HTTP 接口拿到的分，与交付文档里报的分不是同一条路径算出来的。** 两者都正确，
但不可混用：给 `QACDPipeline(assembler=...)` 传入冻结装配器只会把**列名**换成冻结的
112 个，列**值**仍然来自 48 维实时路径，缺的 93 列会被静默填 0。字段契约见
**[docs/02-接口文档.md](docs/02-接口文档.md)**。

## 实验数据与结果

**本仓库不包含实验数据**（冻结特征矩阵、原始证据表与结果表）。这里保留的是方法与可运行的代码。

要运行下面这些检查，需要一份数据包——由作者提供，取得方式见 **[REPRODUCING.md](REPRODUCING.md)**。
把 `evidence_bundle.npz` 与 `raw/` 放到 `artifacts/` 下即可：

```
artifacts/
├── evidence_bundle.npz      # 冻结特征矩阵、标签、划分、参考分数
└── raw/                     # 开发集与测试集的原始证据表
```

| 脚本 | 验证内容 |
|---|---|
| `scripts/run_frozen_evaluation.py --from-raw` | 从原始证据重建特征 → 校准 → 聚合 → 指标 |
| `scripts/verify_feature_port.py` | 112 维特征与冻结矩阵逐元素对齐 |
| `scripts/verify_decomposition_v1.py` | v1 分解复现冻结 claim table |
| `scripts/verify_pipeline_frozen.py` | `QACDPipeline` 端到端闭环 |
| `scripts/measure_calibration_gain.py` | 校准性能（ECE）的校准前后对比 |

`measure_calibration_gain.py` 算的是**指标 2.2** 需要的那个数：ECE 的相对改进。它同时
给出 Brier 与分箱敏感性，**引用时必须连同 `n_bins` 与 `strategy` 一起写**——ECE 是
分箱相关的，换设置数值就变。

> **关于 ECE 的一个陷阱。** ECE 只衡量"说出的概率与实际发生的频率是否一致"，不衡量
> "有没有把两类区分开"。一个永远输出基准失败率的**常数预测器**，ECE 恰好是 0，
> 判别能力也是 0（AUROC 0.5）。所以报指标 2.2 时**务必同时给出 Brier 或 AUROC**，
> 否则第三方用一个常数预测器就能在任何真实方法面前"赢"下 ECE。
> `measure_calibration_gain.py` 因此把三个指标一并打印。

**缺少数据时的行为**：以上脚本以退出码 **2** 明确报出缺失项与恢复步骤，不会抛栈、
也不会在零输入上报告成功。测试中依赖数据的 14 项显示为 `skipped`
（`pytest -rs` 会打印原因），其余 215 项不需要任何数据。另有 1 项校验对接表生成器的测试
在缺少 `python-docx` 时同样跳过（该依赖只用于生成交付 Word，未列入 `requirements.txt`）。

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

决策层**只依赖 NumPy**，CPU 可跑，无需 GPU 与模型权重。

```bash
git clone https://github.com/hlcccc/QACD.git && cd QACD

# 安装：包本体 + 命令行入口 qacd + 测试依赖
pip install -e ".[test]"
```

> 只装运行时依赖（`pip install -r requirements.txt`）**不会**创建 `qacd` 命令，也不会装
> pytest —— 那种装法适合把 QACD 当库嵌进你自己的代码，此时请用 `python -m qacd.cli` 代替 `qacd`。

```bash
python -m pytest -q                  # 215 passed + 15 skipped（14 项需数据，1 项需 python-docx）
python scripts/demo_offline.py       # 端到端离线演示（合成开发集）
qacd demo                            # 同上的 CLI 版本
```

上面这一步**不需要模型、不需要显卡**，它验证的是安装和链路。但它用的是 `MockProvider`，
**一行 `LLaVAProvider` 的代码都没跑到** —— 所以它不能告诉你接线是否正确。

想在拿到 26 GB 权重之前先验证真实 provider 的接线：

```bash
python examples/smoke_test_stub_model.py
```

它用一个确定性桩模型驱动**真实的** `LLaVAProvider`（prompt 构造、回答解析、视角路由、
类型化验证探针全部执行），并冒烟五个端点。它证明接线是通的，**不证明方法有效** ——
桩模型的回答是写死的，分数没有意义。

要拿到**有意义的分数**，必须接上真实的证据提供方：

```bash
# 1. 装模型依赖 + 取权重（约 26 GB，不在本仓库内）
pip install -e ".[llava,ocr]"
huggingface-cli download llava-hf/llava-1.5-13b-hf --local-dir /models/llava-1.5-13b-hf

# 2. 用你自己的开发集拟合打分器（dev.jsonl: question/answer/image/failed/group）
qacd fit --data dev.jsonl --out scorer.json --k 3 \
         --provider llava --model-path /models/llava-1.5-13b-hf

# 3. 起 HTTP 服务（四个对接端点）
pip install -e ".[service]"
qacd serve --scorer scorer.json --port 8080 \
           --provider llava --model-path /models/llava-1.5-13b-hf
```

完整可复制的一页脚本见 **[`examples/run_real_provider.py`](examples/run_real_provider.py)**。

> ### ⚠️ `score` 与 `fit` 必须显式指定 `--provider`
>
> | 取值 | 证据来源 | 用途 |
> |---|---|---|
> | `llava` | 真实 LLaVA 检查点 + RapidOCR | **唯一能产出有意义分数的配置** |
> | `mock` | 无模型、无图像，读数由词面重叠推得 | 只用于打通链路 |
>
> mock **不再是默认值**。它会打印醒目横幅，所生成的打分器文件里也记有
> `provenance.provider_kind = "mock"`，此后每次加载都会带出"该打分器不含模型证据、
> 分数无意义"的告警。这样设计是因为：一个静默的假默认值，会让第一次使用的人拿到
> 一个看起来很确定、方向却完全相反的分数，而且没有任何提示。

## 准备模型与数据

### 模型

```bash
pip install -e ".[llava,ocr]"
huggingface-cli download llava-hf/llava-1.5-13b-hf --local-dir /models/llava-1.5-13b-hf
```

fp16 约 26 GB，建议单卡 ≥40 GB。**不必先下完再验证接线**——
`python examples/smoke_test_stub_model.py` 用桩模型跑通真实 provider 的全部代码路径。

换成别的 LVLM 也可以，但**冻结系数不复用**，必须在自己的留出集上重新拟合与评估
（见 `docs/04` 第 9 节）。

### 数据

QACD 的打分器**在你自己带标签的开发集上拟合**，所以这一步必须由使用方完成。
开发集是一份 JSONL，每行一条回答：

```json
{"question": "What brand is the camera?", "answer": "Dakota Digital",
 "image": "/data/textvqa/val/eb38600d8a5ade9a.jpg", "failed": 0,
 "group": "/data/textvqa/val/eb38600d8a5ade9a.jpg"}
```

| 字段 | 含义 |
|---|---|
| `question` / `answer` | 问题，以及**被评估模型给出的回答** |
| `image` | 图像路径，供证据通道读取 |
| `failed` | `1` 表示这个回答是错的。**判定规则由你定义**——见下 |
| `group` | **必须是图像**。校准集按图像切分，同一张图的多个问题不能跨切分，否则校准集里会有训练集的近重复样本，报出的置信度会偏乐观 |

**图像从哪来**：用你自己的 VQA 数据即可，不必是 TextVQA。若要用 TextVQA，
从数据集官方渠道获取（本仓库不分发）。

**回答与标签从哪来**：跑你的被评估模型得到 `answer`，再与参考答案比对得到 `failed`。
这一步有现成工具：

```bash
python examples/make_dev_set.py --data raw.jsonl --out dev.jsonl
```

`raw.jsonl` 每行在开发集字段之外多带答案依据，二选一：

```json
{"question": "...", "answer": "...", "image": "...", "gold": "Dakota Digital"}
{"question": "...", "answer": "...", "image": "...", "answers": ["Dakota Digital", "Dakota Digital", "..."]}
```

- `gold` 单个参考答案 → 归一化后精确匹配（大小写、标点、多余空格不影响）；
- `answers` 十个标注者答案（VQA 惯例）→ 用标准的 `min(匹配数 / 3, 1)` 打分，
  再由 `--accuracy-threshold` 决定二值标签。

**标签定义是平台决策，不是 QACD 的属性。** 如果你的产品更在意品牌名错误而不是计数错误，
改 `examples/make_dev_set.py` 里的 `decide_label` 换成自己的规则即可——**关键是把它写下来**，
校准器的上限就是标签的质量。

**要多少条**：300 条是下限而非够用线；增益的大部分在约 1,200 条时实现。
见 `docs/03` 第 4 节的学习曲线。脚本会在条数或标签分布异常时给出告警。

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
| 决策层（分解 / 校准 / 聚合 / MVR / 保形） | 完整，215 项离线测试覆盖 |
| 特征层 `frozen/`（112 维冻结特征工程） | 完整；逐元素对齐检查在数据到位后可运行 |
| 分解 v1（`qacd/decompose_v1.py`） | 完整；冻结 claim table 复现检查同上 |
| 分解 v2（`qacd/decompose.py`） | 完整，仓库默认；**不是**产出报告数值的版本 |
| 评估层（证据包校验 / 指标 / 配对 bootstrap） | 完整 |
| `qacd/features.py`（48 维轻量参考路径） | 仅供离线演示，**不是**产出报告数值的特征集 |
| `MockProvider`（离线确定性桩） | 完整，用于链路自测 |
| `RapidOCRProvider` | 完整，需 `rapidocr-onnxruntime` |
| `LLaVAProvider`（LLaVA-1.5-13B 证据提供方） | **完整实现**：四类提示、类型化探针、yes/no 解析、logprob 置信度、K 次采样。模型调用点 `_generate_with_scores` 可注入，整套逻辑在 `tests/test_provider.py` 中无权重跑通 |
| HTTP 服务层 | 端点实现完整，需 `fastapi` 额外依赖 |

## 文档

| 文档 | 内容 |
|---|---|
| **[复现说明](REPRODUCING.md)** | **面向验收人：什么能验证、什么不能、各怎么做** |
| [01-方法说明](docs/01-方法说明.md) | 方法、流程与结果 |
| [02-接口文档](docs/02-接口文档.md) | 四个对接端点的入参/出参契约 |
| [03-部署与资源需求](docs/03-部署与资源需求.md) | 依赖、显存、调用次数、开发集规模、部署步骤 |
| [04-平台对接说明](docs/04-平台对接说明.md) | 对接边界、验收标准、部署安全边界、监控与 FAQ |
| [05-课题一技术对接表](docs/05-课题一技术对接表.md) | 技术统计表（正式交付件为 Word 版） |

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
└── tests/                 230 项测试（215 离线 + 15 需数据/依赖）
```
