# FP4 paged MQA logits 性能复现

这是**现有算子的优化**。基线直接使用已发布/已构建的旧版本及其实测耗时，不复制或编写另一套基线算子。

脚本 `test/benchmark/bench_paged_mqa_logits_fp4.py` 只调用 `lightop.attention.get_paged_mqa_logits_fp4_metadata` 和 `lightop.attention.paged_mqa_logits_fp4`。用相同脚本测量不同版本，再比较保存的结果即可。它不依赖模型、SGLang、HumanEval 或 8 卡服务。

正确性继续由两个已有文件覆盖，不新增 CI 测试文件：

- `test/ci_tests/test_paged_mqa_logits_fp4.py`
- `test/ci_tests/test_paged_mqa_logits_fp4_metadata.py`

## 基线版本与测量范围

本 PR 的主对照是第一父提交 **`5c866bec33c3385053c61b2869751c79ae8a4875`** 中的既有实现，避免用中间优化版本代替 PR 基线。

当前候选内核对应已测 wheel 的源码提交 `ade65b856ea8d7d5f480b1fc9d930741ea6279d5`。提交合并后的内核、Python 运行代码和构建代码相同，后续仅整理测试、benchmark 和文档，wheel 保留原来的版本标识。

主要抽样 128K，per-DP batch 为 1/4/7/8/16/32，每请求 1 行 decode 和 6 行 VERIFY，生成偏移 0/1023，共24形状。偏移表示生成1024 token期间的KV长度变化，不表示实际运行完整模型生成。

## 运行当前版本

在已安装目标 LightOp 的 HIP PyTorch 环境中，从本分支仓库根目录运行：

```bash
python3 test/benchmark/bench_paged_mqa_logits_fp4.py \
  --gpu 0 --batches 1,4,7,8,16,32 --rows 1,6 \
  --input-lengths 131072 --offsets 0,1023 \
  --mode graph --warmup 10 --samples 9 --repeats 8 \
  --output /tmp/fp4-candidate-$(date +%Y%m%d_%H%M%S)
```

脚本记录实际加载包的路径、版本、扩展SHA、脚本SHA、GPU PCI/架构/CU、DTK、PyTorch和Python版本。可加 `--expect-commit ade65b85` 核对当前wheel内的提交；不是用仓库HEAD猜测安装包身份。

测试完成自动保存 `result.csv`，对比时自动保存 `comparison.csv`，可直接用 Excel/WPS 打开。CSV 使用 Python 标准库生成、UTF-8 BOM 编码，无需额外安装包；耗时列单位为 µs，`latency_reduction_percent` 的数值单位为百分比，正值表示候选更快。原有 JSON 和 Markdown 输出保留。

输出目录必须为新目录；不覆盖历史结果。失败时已生成的原始样本仍保留。默认只用物理GPU0，PyTorch分配上限6GiB，可用 `--memory-gib` 修改。运行前自行确认该卡可用于测量，脚本不会停止其他服务。

## 基线复测与对比

已有可靠基线结果时，可以直接引用版本和耗时表。需要复测时，在基线版本环境执行**同一个脚本、相同参数**，另存输出目录即可；无需专门生成基线代码。

两个版本可分别安装在独立环境，也可将wheel解压后使用 `PYTHONPATH` 隔离加载，避免覆盖当前环境。例如：

```bash
# 使用各自已解压的包目录。两个命令的其余参数保持一致。
PYTHONPATH=/path/to/baseline-package python3 test/benchmark/bench_paged_mqa_logits_fp4.py \
  --expect-commit 5c866bec --output /tmp/fp4-A1

PYTHONPATH=/path/to/candidate-package python3 test/benchmark/bench_paged_mqa_logits_fp4.py \
  --expect-commit ade65b85 --output /tmp/fp4-B1

# 比较保存的结果，不加载torch或GPU。
python3 test/benchmark/bench_paged_mqa_logits_fp4.py \
  --compare /tmp/fp4-A1/result.json /tmp/fp4-B1/result.json \
  --output /tmp/fp4-compare-A1-B1
```

如需检查次序影响，继续用同一脚本按候选B2、基线A2的次序各跑一次，然后以 `--compare A2/result.json B2/result.json` 比较。不要覆盖A1/B1，不选择性删除慢样本。

两份已测wheel在本任务服务器的容器路径如下（宿主前缀为 `/root/luoyao/projects/deepseekv41flash`）：

```text
基线：
/home/projects/deepseekv41flash/fp4_logits_20261031/wheels/baseline/lightop-0.6.0+das1.1git5c866be.abi1.dtk26041.torch2.11-cp310-cp310-linux_x86_64.whl

候选：
/home/projects/deepseekv41flash/fp4-logits-general-gpu0-ade65b85/wheels/lightop-0.6.0+das1.1gitade65b8.abi1.dtk26041.torch2.11-cp310-cp310-linux_x86_64.whl
```

| 角色 | wheel内提交 | wheel SHA256 |
|---|---|---|
| 基线 | `5c866be` | `6355ac9af2d7b1998a585319bb48851051350b1c1fd55600bb65c8f3708e3c8f` |
| 候选 | `ade65b8` | `2f82f8d6080c4e7ca3986ed7c59db1c21616a4432b25bbef68bac8163de1637d` |

如自行构建，在相同DTK/PyTorch环境分别检出基线和当前分支，运行 `python3 setup.py bdist_wheel` 并保留各自产物。基线的setup.py会查询GPU0，因此其构建环境需要可访问GPU。

## 可选控制点

在同一命令中调整形状参数和新的输出目录即可，不要求遍历所有组合：

```text
# 4K/50K/64K，小/大批量抽样
--batches 2,16 --rows 1,6 --input-lengths 4096,51200,65536 --offsets 512

# 混合长短请求，较宽scale
--batches 3,16,32 --rows 5,6 --input-lengths 131072 --offsets 1023 --layout mixed --scale wide

# sparse表块8；多行请求复用不适用于此配置
--batches 1,16,32 --rows 1 --input-lengths 4096 --offsets 0 --table-block-size 8

# 快速检查脚本与安装包
--batches 1 --rows 1 --input-lengths 64 --offsets 0 --warmup 2 --samples 3 --repeats 2
```

也支持 `--mode eager`。不要将eager与graph结果混入同一比较。

## 计时与结果判定

- 使用CPU RNG生成非零随机FP4 payload、Q、scale、权重、页表和长度。比较时要求形状、种子、全部输入SHA一致。
- 调用范围为**设备端metadata规划 + logits**。输入准备、host copy和校验不计时；graph模式排除捕图与CPU分配开销。
- eager预热10次；graph模式捕图后再预热10次。每形状采9个时间样本，每样本执行8次，用GPU event并同步后换算每次调用的µs。
- 输出必须有限，不可见尾部保留固定哨兵。比较时检查两个版本完整输出SHA相同；独立数学参考仍由已有CI测试负责。
- `result.json` 保存完整身份与结果，`samples.jsonl` 逐项保存原始时间、min/max、中位数、输入/输出SHA及分配峰值。
- `--compare` 生成 `comparison.json` 和 `comparison.md`，列出每个形状的基线耗时、当前耗时与变化。下降比例为 `(baseline - candidate) / baseline × 100%`，正值更快。
- 退出0只表示执行和校验完成，不自动表示性能提升。以目标128K形状相对基线的实际耗时评估收益，保留退化与噪声；不增加“所有形状必须提升”的要求。
- 算子级优化的PR验证依据为已有正确性测试和同条件前后对照。框架联调、模型吞吐和HumanEval另行验证，不作为本benchmark的前置条件。


## 本次基线性能与候选结果（2026-11-04）

环境：物理GPU0，gfx938、64 CU，Python3.10.12，DTK26.04.1，PyTorch2.11.0+dtk2604.2609291046.ge12687。基线为5c866bec，候选为ade65b85对应的同一运行代码。使用上述默认128K矩阵与同一脚本，按A1/B1/B2/A2完成96条测量；全部输入和输出SHA一致。

24个形状中20个在两个次序下都更快。单行decode为11/12个，多行VERIFY为9/12个。VERIFY的两序跨形状耗时降幅中位数为34.82%/35.02%，decode为14.25%/13.09%；这两个汇总值不是统计置信区间。

GPU0有其他进程常驻，179条资源采样中13条记录到其他进程计算活动，最大48 CU。因此下表是本次共卡实测，不能当作独占环境下的稳定收益；不将某个异常值直接归因于外部干扰。尤其保留bs8/32多行场景和bs1单行场景的不利样本。

所有行均input_len=131072、H=32；单位为每次metadata+logits调用的微秒，正值表示候选耗时降低。

| bs | rows | offset | A1 us | B1 us | A/B下降 | A2 us | B2 us | B/A下降 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 47.94 | 43.18 | +9.93% | 47.72 | 154.08 | -222.88% |
| 1 | 1 | 1023 | 55.68 | 44.60 | +19.90% | 48.40 | 44.74 | +7.56% |
| 1 | 6 | 0 | 189.88 | 128.48 | +32.34% | 189.44 | 128.64 | +32.09% |
| 1 | 6 | 1023 | 191.60 | 130.40 | +31.94% | 381.20 | 130.56 | +65.75% |
| 4 | 1 | 0 | 135.62 | 118.32 | +12.76% | 135.66 | 118.16 | +12.90% |
| 4 | 1 | 1023 | 139.38 | 123.38 | +11.48% | 139.26 | 123.44 | +11.36% |
| 4 | 6 | 0 | 708.12 | 435.46 | +38.50% | 707.36 | 435.90 | +38.38% |
| 4 | 6 | 1023 | 713.22 | 443.28 | +37.85% | 713.96 | 442.98 | +37.95% |
| 7 | 1 | 0 | 224.70 | 196.18 | +12.69% | 224.36 | 196.06 | +12.61% |
| 7 | 1 | 1023 | 225.56 | 197.26 | +12.55% | 225.80 | 197.02 | +12.75% |
| 7 | 6 | 0 | 1231.76 | 663.28 | +46.15% | 1230.36 | 663.34 | +46.09% |
| 7 | 6 | 1023 | 1243.82 | 672.10 | +45.96% | 1241.28 | 672.04 | +45.86% |
| 8 | 1 | 0 | 251.14 | 213.48 | +15.00% | 251.34 | 213.90 | +14.90% |
| 8 | 1 | 1023 | 255.92 | 221.34 | +13.51% | 255.62 | 221.66 | +13.29% |
| 8 | 6 | 0 | 1048.58 | 1321.00 | -25.98% | 1048.26 | 799.64 | +23.72% |
| 8 | 6 | 1023 | 1289.06 | 808.16 | +37.31% | 1065.24 | 1220.90 | -14.61% |
| 16 | 1 | 0 | 483.74 | 403.88 | +16.51% | 483.88 | 404.04 | +16.50% |
| 16 | 1 | 1023 | 489.36 | 414.12 | +15.38% | 489.58 | 413.62 | +15.52% |
| 16 | 6 | 0 | 2070.90 | 1563.64 | +24.49% | 2072.24 | 1564.64 | +24.50% |
| 16 | 6 | 1023 | 2618.52 | 1580.88 | +39.63% | 2769.12 | 1579.26 | +42.97% |
| 32 | 1 | 0 | 948.58 | 707.26 | +25.44% | 948.38 | 707.42 | +25.41% |
| 32 | 1 | 1023 | 955.44 | 726.42 | +23.97% | 955.88 | 726.66 | +23.98% |
| 32 | 6 | 0 | 4123.80 | 3115.28 | +24.46% | 4123.74 | 4206.52 | -2.01% |
| 32 | 6 | 1023 | 4153.46 | 3158.76 | +23.95% | 4153.74 | 3504.18 | +15.64% |

原始96条记录、每例9个时间样本、版本/扩展身份、审计与脚本保存在服务器：
`/root/luoyao/projects/deepseekv41flash/fp4-logits-benchmark-20261104/benchmark_evidence.tar.gz`。

- 证据包SHA256：`fd990dd97e4d74c9b852fbbf7878b37b0dcbdabe7060f56a7335f6c46da2abef`。
- 本地逐文件核验39个成员通过。
- 实测benchmark脚本SHA256：`f8dd4545c058fb464dede8a3174534c7c130254ed721139bf3d6af95d03c1457`。
- 离线对比入口在CPU环境重新执行通过；故意改变输入SHA后能拒绝不一致的比较。

实测副本使用CRLF换行；规范为LF后的脚本SHA256为 `7422fc4e2b6a246d4569dc259f8a32a8081ffabbc9421fda2bbbafb8034c4176`，与仓库脚本一致，执行逻辑未改动。
