# Jaka MF (Multi-Frame Future) 策略迁移

## 修改的文件


| 文件                                            | 操作  | 说明                                                                          |
| --------------------------------------------- | --- | --------------------------------------------------------------------------- |
| `sim2real/rl_policy/observations/jaka_mf.py`  | 新建  | MF 观测类 `jaka_frame_stack_mf`，输出 620 维                                       |
| `sim2real/rl_policy/observations/__init__.py` | 修改  | 添加 `from .jaka_mf import *`                                                 |
| `sim2real/config/robots/Jaka.py`              | 修改  | `mjcf_path` 改为 `checkpoints/jaka-mini/Khan_mini_simplified_new_bigfeet.xml` |
| `checkpoints/jaka_mf/latest_mf40k.yaml`       | 新建  | MF 策略配置，`future_steps: [0,1,2,3,4]`                                         |
| `checkpoints/jaka_mf/latest_mf40k.onnx`       | 已有  | MF 策略模型                                                                     |


## 与旧版关键差异

- **Command**: 33→155 维（单帧→5 帧未来 root_pos_diff + z + joint_pos）
- **Anchor ori**: 6→30 维（单帧→5 帧未来 rot6d）
- **History obs**: 126→87 维/帧（command/anchor_ori 移出 history）
- **总维度**: 630→620
- **Gravity**: root quat → waist_yaw_Link body quat
- **Neck/Wrist yaw**: command 中置零

## 运行命令

```bash
# Terminal 1
python sim2real/sim_env/base_sim.py --robot jaka

# Terminal 1
python -m sim2real.rl_policy.tracking \
  --robot jaka \
  --policy_config checkpoints/jaka_mf/latest_mf40k.yaml
```

