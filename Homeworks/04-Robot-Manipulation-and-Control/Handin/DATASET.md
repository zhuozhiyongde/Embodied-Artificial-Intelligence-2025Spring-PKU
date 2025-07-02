# Dataset

生成供训练的点云及观测数据。

1. 使用 `tools/wrapper_env.py` 替换 `src/sim/wrapper_env.py`，以获得随机化以及分割图。
2. 执行 `mv tools/* .` 来切换工作目录。
3. 启动 `produce_data.py` 脚本，生成数据。

```bash
python produce_data.py
```

3. 生成得到的数据，移动到 `tools/model/data` 目录下，即可使用 `tools/model` 中的 Assignment2 同款框架训练模型，注意训练 `est_coord` 而非 `est_pose`，前者估计更加准确。

注意，为了获取更开阔的视野、更完整的点云，我调整了相机初始位姿（`produce_data.py/generate_data()`，所以你的 `main.py` 中也需要相应修改）：

```python
observing_qpos = humanoid_init_qpos + np.array([0.01, 0, 0.25, 0, 0, 0, 0.15])
init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos, steps=20)
execute_plan(env, init_plan)
```

一些 Tricks/踩坑：

1. 分割不需要上 SAM 模型，直接用 bbox 分出来即可；可以通过启发式算法滤掉桌面点，详见 `extra/utils.py/get_workspace_mask()`。
2. 善用可视化工具 tools/vis.py 来检查点云、位姿，从而 debug。
3. 你可以使用 `parallel_produce_data.py` 来并行生成数据，作为参考，4090+14900K，4 进程生成 5000 样本，耗时约 15 分钟。
4. 各个辅助脚本都有参数，自行探索，在此不多赘述。