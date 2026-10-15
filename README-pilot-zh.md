# DUET 子指令感知融合：R2R 低成本验证（修订版）

基于官方 cshizhe/VLN-DUET，基准提交 `93e8b233164bc079a6db48b8a0a78d123ec8de41`。独立副本，不包含 AFv1，不需要重新预训练。原始 README 保留官方数据和环境说明。

## 方法

完整指令经过原有语言编码器。规则按标点和连接词定位子句，对同一份 contextual token embeddings 做子句均值池化。用 global/local summary 拼接作为 query，注意力聚合子句，得到语言状态。

`alpha = sigmoid(original_gate(global, local) + residual(global, local, language))`

原有动作映射、global/local logits 和导航损失保留。冻结编码器及动作预测头，仅训练原有融合门控和新增残差模块。冻结主干保持 eval 模式，关闭其 dropout；训练采用原有 teacher-forcing imitation loss，验证采用 argmax 自主导航。

残差末层在原始 checkpoint 加载后再次置零，初始门控与原始模型一致。新增注意力和 MLP 使用 128 维中间层；hidden_size=768 时该模块约 60.7 万参数。冻结主干仍需前向计算；时间与显存收益需实测。

这是子指令表示辅助融合，没有显式方向/地标分类或进度跟踪。无额外分块标签和在线 LLM 调用，不声称分块本身具有新颖性。

| mode | 作用 |
|---|---|
| off | 原始门控；iters=0 为不训练基线，iters>0 为继续训练门控对照 |
| mean | 有效语言 token 均值，经 value 投影后进入残差 |
| token_attention | 对有效 token 做状态查询注意力 |
| semantic_chunks | 对语义子句均值做状态查询注意力 |
| equal_chunks | 每条指令匹配语义块数，以等长 token 分组替代语义边界 |

所有语言模式共享有效 token 范围：排除特殊 token 和分块连接边界。mean 不使用 query/key，不能作为严格参数量匹配对照；关键对照为 token_attention 与 equal_chunks。规则仅近似子句划分。

## 环境和检查

使用已经能运行官方 DUET 的 Linux/CUDA/MatterSim 环境。需要官方 R2R 数据、图像特征、connectivity、BERT tokenizer 和已有 R2R 微调 checkpoint。模型层数与特征维度必须匹配 checkpoint。代码包不含数据或权重。

仓库根目录执行：

```bash
python tools/test_semantic_pilot.py --require-torch
```

无需 GPU/MatterSim/数据：检查截断对齐、分块公平性、零残差、梯度、padding 不变性及 checkpoint 保存/加载。缺少 PyTorch 时强制失败，避免把跳过数值检查当成通过。

本地验证：3 项纯 Python 测试通过；2 项 PyTorch 数值测试因本机缺少 PyTorch 跳过。安装 PyTorch 时 C 盘空间不足，未完成安装。Python 语法和入口帮助检查通过；完整 MatterSim 导航、真实 BERT annotation 对齐及 GPU 训练尚未验证。

## 最小运行

仓库根目录设置服务器路径：

```bash
export DATA_ROOT=/path/to/datasets
export DUET_CKPT=/path/to/best_val_unseen
export PILOT_ROOT=/path/to/pilot_results
bash tools/run_r2r_semantic_pilot.sh
```

脚本先运行 CPU 检查、评估原始 checkpoint，再对五个 mode 各训练 100 次。默认 64 条训练/64 条 val_unseen 指令，batch_size=4。此预算用于验证流程与梯度，不足以判断方法有效。

单独训练方法，从仓库根目录执行：

```bash
cd map_nav_src
python ../tools/run_semantic_pilot.py --mode semantic_chunks \
  --root_dir /path/to/datasets --resume_file /path/to/best_val_unseen \
  --output_dir /path/to/pilot/semantic_chunks \
  --dataset r2r --tokenizer bert --fusion dynamic \
  --enc_full_graph --graph_sprels \
  --features vitbase --image_feat_size 768 --angle_feat_size 4 \
  --batch_size 4 --max_instr_len 200 --max_action_len 15 \
  --optim adamW --lr 1e-4 --train_alg imitation \
  --iters 100 --log_every 100 --eval_first
```

各方法使用同一 checkpoint、训练预算与 seed。训练前语义模型的指标应与 off/iters=0 一致；若不一致，先排查配置与加载。

入口默认 100 次更新，不继承原始 checkpoint 的步数。只支持单 GPU R2R imitation、无增广数据，不恢复原始 optimizer。

## 加载适配器

复制单方法命令，保持原始 checkpoint 和模型配置一致，将 iters 改为 0、output_dir 改为新目录，并添加：

```bash
--pilot-checkpoint /path/to/pilot/semantic_chunks/ckpts/latest_dict.pilot.pt
```

mode 必须匹配适配器。需要原始 checkpoint 加适配器两个文件。若 iters>0，则用新 optimizer 开始新一轮适配；不是精确恢复优化器和随机状态的断点续训。

适配器格式为 version 2，与之前未验证原型不兼容，请重新训练。

## 扩大验证与输出

通过最小检查后，可用 `--limit-train-examples 1000 --limit-val-examples 300` 扩大筛选；0 表示完整划分。子集成员固定，不随训练 seed/mode 改变，记录在 `pilot_subsets.json`。指令数量不等于独立路径数量。

- `logs/train.txt`：损失和 SR/SPL/NE/TL 等官方指标。
- `ckpts/latest_dict.pilot.pt`：最终适配器；best_val_unseen 按验证 SPL 选择。
- `pilot_subsets.json`：训练/验证指令 ID。
- `pilot_resources.json`：整个运行含评估的耗时与峰值 GPU allocated memory。
- 控制台：可训练参数数量、加载状态。

先比较相同最终步数结果，再扩大到完整 val_unseen 和多个 seed。不要在验证数据上训练。只有胜过 token 与等长分块对照，才能支持语义边界有帮助。

## 关键文件

- `map_nav_src/models/semantic_fusion.py`：独立残差模块。
- `map_nav_src/models/vilmodel.py`：向原有 gate logit 添加残差。
- `map_nav_src/models/model.py`：冻结主干 train/eval 控制。
- `tools/run_semantic_pilot.py`：分块、对齐、冻结设置和入口。
- `tools/pilot_checkpoint.py`：原始模型与适配器加载。
- `map_nav_src/r2r/main_nav.py`：续训区间、保存步数和子集记录。
- `tools/test_semantic_pilot.py`：纯 Python / CPU PyTorch 测试。

原始 DUET 默认 semantic mode 为 off；新语言模式应通过 pilot 入口运行，以提供 chunk IDs。
