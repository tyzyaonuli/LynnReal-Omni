# 泰国 B300 推理交接

批量 T2V eval 的天鹏交接入口、JSONL schema 和结果契约见
[`aliyun-thailand-b300-eval-handoff.md`](aliyun-thailand-b300-eval-handoff.md)。

## 已完成的官方对齐速度基线

2026-09-16（北京时间）已在泰国 PAI-DLC 用单卡完成 Standard 和 Flash 的 10 秒 T2V 官方 speed-test 流程。成功任务：

- Job ID：`dlcfchvd6ou2zz4v`
- Display name：`lynnreal-b300-official-10s-97fdcb8-kfix`
- Repo revision：`97fdcb871b28b3a2b2316b5d19a9ea8e61185570`
- Kernel hash：`44e0d680bad9991a`
- PAI 状态：`Succeeded`，总占卡时间 5175 秒
- 控制台：`https://pai.console.aliyun.com/?regionId=ap-southeast-7&workspaceId=384#/dlc/jobs/dlcfchvd6ou2zz4v/overview`
- CPFS 结果：`/cpfs/world-model/lynnreal-omni/results/dlcfchvd6ou2zz4v/`
- OSS 结果：`oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/dlcfchvd6ou2zz4v/`
- 本地结果：`output/aliyun_thailand_b300/dlcfchvd6ou2zz4v/`

测试直接调用仓库的 `script/sample/speed_test/{standard,flash}.sh`：相同茶壶 prompt、seed 77、1344×768、240 个交付帧、24 fps；H3 原生时间对齐实际计算 243 帧，padding 计算计入时延。每个候选配置做 2 次不计时 warmup 和 5 次 measured，按“每次 DiT + decoder CUDA 时间之和的中位数”选最佳项。Standard 为 W8A8、4 steps，Flash 为训练好的 W8A8、3 steps，均使用 light VAE，不做 refinement。

B300 不能使用仓库 H100 `fixed` 路径写死的 FA3，因此使用官方 `--search fast` 流程，在 B300 可用的 `_native_cudnn`/native SDPA 和 adaptive/native decoder tiles 中搜索；INT8 GEMM 始终固定为修复后的 Triton 实现。

| 模型 | 路径 | 最佳配置 | DiT | Decoder | DiT + Decoder | Generate wall | 峰值显存 | SHA256 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Standard | `standard/video.mp4` | adaptive tiles，native SDPA，Triton INT8 | 96.180 s | 1.788 s | 97.969 s | 98.904 s | 63.846 GiB | `3d2737ac1d8bdd82987e79166b66f887410b64da2635cc2adadd25c20696ebd9` |
| Flash | `flash/video.mp4` | adaptive tiles，native SDPA，Triton INT8 | 33.016 s | 1.765 s | 34.781 s | 35.495 s | 55.104 GiB | `f47a6abf26f5af50112994637f05a27245e54668433c4fde324b623a97130e30` |

同卡同口径下，Flash 的 DiT + decoder 比 Standard 快 2.817 倍，generate wall 快 2.787 倍。云端 PyAV 与本地 `ffprobe` 独立校验均确认两份最佳视频为 240 帧、10.000 秒、24 fps、1344×768、H.264，且包含 10.000 秒、32 kHz、双声道 AAC。下载文件 SHA256 与云端一致；两份视频各有 240 个唯一解码帧哈希，音频整体 RMS 分别为 -49.19 dB 和 -58.39 dB，均非静音。首/中/尾帧接触图人工抽查无黑帧或损坏。

## INT8 kernel 修复与门禁

原始实现创建了 `tl.int32` accumulator，却调用 `tl.dot(a, b, acc)` 而未显式指定 `out_dtype`。Triton 3.7 的 `tl.dot` Python 默认值是 `float32`，编译期要求传入 accumulator dtype 与 `out_dtype` 一致，因此触发空消息的断言并回退到 `torch._int_mm`。修复只在 generic 与 Hopper TMA 两条路径为 `tl.dot` 增加 `out_dtype=tl.int32`，没有改变 INT8 输入、scale、bias 或 BF16 epilogue。

完整模型测试前先在同一张 B300 上执行 `script/validate_int8_gemm.py`。四类真实投影维度、每类四个 Blackwell tile 均与 `torch._int_mm` 的 BF16 输出逐元素完全一致；autotune 输出完全一致，连续 5 次确定性检查通过，fallback shape 数为 0。257 行 microbenchmark 的 Triton/`torch._int_mm` 中位数如下：

| M×N×K | Triton | torch._int_mm | Triton speedup |
| --- | ---: | ---: | ---: |
| 257×21504×5376 | 0.564 ms | 0.619 ms | 1.097× |
| 257×5376×7168 | 0.265 ms | 0.309 ms | 1.167× |
| 257×28672×5376 | 0.740 ms | 0.816 ms | 1.103× |
| 257×5376×14336 | 0.499 ms | 0.554 ms | 1.111× |

## 固定资源方案

- 区域：`ap-southeast-7`
- Workspace：`384`
- B300 quota：`quota1betfyi1ilu`
- CPFS dataset：`d-ak7t158lhzaq7kdwp3`，容器内挂载 `/cpfs`
- 默认资源：`1 GPU / 23 CPU / 225 GiB / 64 GiB shared memory`
- 镜像：`zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/zing-training-env:h3-fast-ulysses-torch2121-fa34-f7351c75-r3`
- 实测设备枚举：`NVIDIA L20D`，275040 MiB，compute capability 10.3，driver 580.105.08

两个模型在同一任务内顺序运行，因此始终只占一张卡。当前入口不是分布式实现，申请两张卡不会自动提速；只有算法代码明确实现多卡时才设置 `PAI_RESOURCE_GPU=2`。

## 可复用环境和模型

基础运行时直接复用 Zing 镜像里的 Python 3.12、CUDA 13.0、PyTorch 2.12.1+cu130。LynnReal 所需 Diffusers commit 和上层依赖安装在独立 overlay，不修改镜像：

```text
/cpfs/world-model/lynnreal-omni/
├── cache/env/py312-torch2121-cu130-lynnreal-0384eca3d5d7482d.tar.gz
├── cache/shared/{huggingface,pip}/
├── cache/compiled/b300-torch2121-97fdcb871b28b3a2b2316b5d19a9ea8e61185570-44e0d680bad9991a/
├── models/
│   ├── standard-f1d6990e23496bef6c7d55bdcfe2925df2508325/
│   ├── flash-9950cfc882b0e6b8600fbc49b58dc1d82d7a435b/
│   └── light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a/
└── results/<DLC_JOB_ID>/
```

固定模型清单为：

- Standard：50 个对象，77,301,658,214 bytes
- Flash：32 个对象，39,742,516,911 bytes
- Light VAE：7 个对象，7,729,847,284 bytes

每次任务只需从 CPFS 解压约束哈希对应的 Python overlay 到本地盘，并为权重建立轻量只读视图。权重不会再次下载，Triton/Inductor 编译缓存按 repo revision 和 kernel hash 持久化。Standard 的 `audio_scheduler`、`processor`、`scheduler`、`text_encoder`、`tokenizer` 与现有 H3 checkpoint 哈希相同，可硬链接复用；`audio_vae` 和 `vae` 使用本模型 revision 的官方文件。

B300 启动探针选择 `_native_cudnn`，但完整 speed-test 搜索显示 native SDPA 略快，两模型最佳项均为 native SDPA。当前仓库 FA3 只支持 Hopper，不能在 Blackwell 上强制使用；BF16/FP16、head dim 64/96/128 的启动数值探针均已通过。

## 再跑官方速度测试

默认命令只做 PAI 请求 dry-run；真正创建任务必须显式传 `--submit`：

```bash
bash script/aliyun_thailand_b300_submit.sh --dry-run
bash script/aliyun_thailand_b300_submit.sh --submit
```

若要完全复现本次已经上传并验证过的 bundle，可跳过本地重新打包：

```bash
JOB_NAME=lynnreal-b300-official-10s-97fdcb8-rerun \
LYNNREAL_REUSE_BUNDLE=true \
bash script/aliyun_thailand_b300_submit.sh --submit
```

对应 bundle：

```text
oss://leap-worldmodel-thailand/world-model/code/lynnreal-omni/97fdcb871b28b3a2b2316b5d19a9ea8e61185570-ffc0296066e4.tar.gz
```

算法同事接手时，建议保留 `script/aliyun_thailand_b300_inference_bootstrap.sh` 中的 GPU/版本/哈希预检、kernel 门禁、overlay 解压、缓存变量以及权重视图创建，只替换最后两条 `script/sample/...` 命令为评测入口。`script/aliyun_thailand_b300_submit.sh` 会把固定 HEAD、kernel 修改和本地 bootstrap 一起打成新 bundle；新任务仍应保持单卡并在完成后主动退出。

## 模型缓存刷新

泰国训练 VPC 不能稳定访问 Hugging Face/ModelScope API，因此模型刷新走 CPU 任务，不消耗 B300：

```text
ModelScope OSS 源对象
  -> 杭州临时 staging OSS
  -> OSS 跨区域复制到泰国
  -> 泰国 CPU PAI-DLC 逐文件 SHA256 校验并写入 CPFS
```

杭州 bucket `codex-minmaxh3-cache-97318276` 已配置仅 PUT 的加速跨区域复制规则 `ac710c16-7df8-4af7-a41b-aa5ff7f37b15`，目标为 `leap-worldmodel-thailand/world-model/lynnreal-omni/staged-models-crr/`；删除不会传播到泰国副本。

模型 revision 变化时，先更新三个脚本中的固定 revision/清单，再依次提交：

```bash
bash script/aliyun_thailand_model_cache_submit.sh --stage --submit
bash script/aliyun_thailand_model_cache_submit.sh --sync --submit
```

必须等待 stage 成功且跨区域对象完整到达后再提交 sync。两个任务都使用 CPU ECS 规格；泰国 sync 固定走 ops workspace `385` 和其 RW CPFS dataset `d-qs9w70tf7ss5yw78vc`，不要把已停止新增任务的旧 CPU dataset 或 GPU workspace `384` 用于模型搬运，更不要占用 B300。

## 结果契约

```text
results/<DLC_JOB_ID>/
├── standard/{video.mp4,latency.json,progress.json,run.log,...}
├── flash/{video.mp4,latency.json,progress.json,run.log,...}
├── int8-gemm-validation.json
├── validation.json
├── SHA256SUMS
├── runtime.env
├── bootstrap.log
├── setup/attention.json
└── READY
```

以 `READY` 为唯一成功标记；存在 `FAILED`、缺视频，或 `validation.json` 不满足 240 帧/10 秒/24 fps/1344×768 时，都不能交接评测。
