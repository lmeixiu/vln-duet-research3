# 诊断语义融合是否影响决策

在现有 DUET 服务器环境使用。不会修改训练后的权重，无需重新训练。

## 一键诊断

保持 DATA_ROOT、DUET_CKPT、PILOT_ROOT 环境变量与实验一致，从仓库根目录执行：

```bash
bash tools/run_semantic_diagnosis.sh
```

结果保存到 PILOT_ROOT 下带时间戳的 diagnosis_semantic 目录。默认原有 64 条验证指令；模型配置读取适配器中保存的 args。需要与训练时相同的原始 checkpoint。

## 只检查参数（无需 GPU/模拟器）

```bash
python tools/diagnose_semantic_pilot.py \
  --base "$DUET_CKPT" \
  --adapter "$PILOT_ROOT/semantic_chunks/ckpts/latest_dict.pilot.pt" \
  --output "$PILOT_ROOT/parameter_check"
```

output 必须是空目录或新目录。命令需要 PyTorch。

## 输出与解释

- parameter_report.json：原有门控相对原始 checkpoint 的变化；新增 delta.3 末层相对零初始化的变化；有限值检查。其他随机初始化层没有保存初始值，因此不能从当前 checkpoint 精确计算更新量。
- summary.json：残差绝对值、alpha 变化、alpha 接近 0/1 的比例、同状态 argmax 动作变化比例。
- same_state_steps.json：每次前向、每个 batch 槽位的详细对比，包括有限动作分数的最大变化。
- trajectories.json：开启残差时逐指令实际导航轨迹。
- evaluation/logs/train.txt：开启残差时的官方导航指标。

同状态对照运行两次前向，第二次把 adapter 输出置零，保留训练后的原始融合门控。实际导航始终跟随开启残差的模型。该诊断用于隔离残差贡献，不是与原始完整 checkpoint 的对照，也不是两条独立轨迹的最终 SR 对比。约增加一次导航模型前向开销，不进行反向传播。

summary 的统计范围是模型前向 batch 槽位，可能包含已经结束的样本和评估最后重复的样本，不是独立导航步/指令的无偏统计；不能作为论文性能指标。

判读顺序：

1. delta.3 的变化为零：残差末层未偏离初始化，需要检查训练更新。
2. 参数变化非零，但残差输出接近零：权重更新没有产生明显输出影响。
3. 残差非零，但 alpha 变化很小：可能因门控饱和，需要结合 saturated_rate 和逐步数据。
4. alpha/动作分数变化，但动作选择不变：当前小验证集上的 argmax 对这些变化不敏感。
5. 动作发生变化：模块已经影响决策；是否改善结果需要正式评估。

optimizer 的非零动量是历史更新的间接证据，不是实时梯度。当前脚本不计算真实 imitation loss 的梯度，不宣称完成梯度诊断。不同实验的训练 loss 不能单独证明语义有用。

本地已执行语法与帮助检查；新增 PyTorch 数值测试因本机没有 PyTorch 而未执行。服务器运行前可执行 `python tools/test_semantic_pilot.py --require-torch`，当前共 6 项测试。
