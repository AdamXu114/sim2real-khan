# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/teleop`。

English version: [README.md](./README.md)

Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)

如果你在找 HDMI 的部署栈，请看 [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi)。

## 快速开始

```bash
uv sync
```

运行离线动作跟踪（sim2sim）：

```bash
python sim2real/sim_env/base_sim.py --robot g1
python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml

python sim2real/sim_env/base_sim.py --robot khan
python sim2real/rl_policy/tracking.py \
  --robot khan \
  --policy_config checkpoints/khan/policy_config.yaml
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后在 MuJoCo viewer 里按 `9` 关闭虚拟 gantry。

## 下一步

- [文档首页](https://egalahad.github.io/sim2real/zh-Hans/)
- [快速上手](https://egalahad.github.io/sim2real/zh-Hans/getting-started/overview)
- [Root Project Setup](https://egalahad.github.io/sim2real/zh-Hans/getting-started/root-project)
- [离线动作跟踪教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/offline-motion-tracking)
- [Pico Teleoperation 教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/pico-teleoperation)
- [Motion Recording 教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/motion-recording)

## 1. Start the Pico retarget publisher

```bash
python sim2real/teleop/pico_retarget_pub.py --bind tcp://*:28701 --publish_hz 50 --actual_human_height 1.80
```

## 2. Inspect the retarget in realtime

```bash
python sim2real/teleop/realtime_viewer.py --connect tcp://127.0.0.1:28701 --viewer_hz 50
```

