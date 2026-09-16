# 阿里云上海单卡 H100 推理与测速

本文沉淀 2026-09-15 在阿里云 PAI 华东 2（上海）验证 LynnReal-Omni
Standard/Flash 的可复用方法。它覆盖固定 H100 的资源关系、单卡任务、共享缓存、官方
测速口径、浏览器视频发布和释放门禁。泰国 B300 的批量 eval 入口见
[`aliyun-thailand-b300-eval-handoff.md`](aliyun-thailand-b300-eval-handoff.md)。

## 资源关系与单卡约束

固定 H100 分为三层：

1. 灵骏资源池中的按量使用订单负责实际机器和整机计费；
2. 资源配额把订单、PAI workspace 和网络绑定起来；
3. DLC Job 从配额中申请 CPU、内存和 GPU。

本次验证的实例为 `ml.gx8ef.8.46xlarge`，一台机器包含 8 张 80 GB H100；PAI
内部设备名显示为 `NVIDIA L20Z`。LynnReal-Omni 推理是单进程单卡实现，单个 Job 只应
申请 1 GPU。多申请 GPU 不会加速单个样本，只会占用同事可用的卡。

要特别区分“停止 DLC”与“释放整机”：停止单卡 Job 只归还 quota 中的卡，不会取消
资源池按量订单，也不会停止整机费用。创建顺序是按量订单 → quota 绑定 → DLC Job，
释放顺序必须反过来：

1. 停止或等待所有 DLC Job 结束；
2. 确认 quota 内没有其他人的活跃 Job/Pod；
3. 解绑并删除 quota；
4. 释放按量订单，重新查询资源池/订单，确认实例不再存在或状态已终止。

自动释放必须 fail closed：只要无法完成全量、支持分页的 Job/Pod 归属检查，就不得释放
整机。不要依赖不存在的 `paistudio list-node-pods` API，也不要仅凭当前 Job 已结束就
推断整机空闲。

## 环境与共享存储

Hopper 上验证过的基础环境为 Torch 2.7.1 + CUDA 12.8 + FA3。PAI 官方镜像：

```text
dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/pytorch:2.7.0-gpu-py312-cu128-ubuntu24.04-ngc25.02-4e622ce0-1765438200
```

任务把持久化内容放到同地域共享挂载，节点本地盘只负责解压、权重热拷贝和运行：

```text
/mnt/lynnreal-omni/
├── code/LynnReal-Omni-<git-sha>.tar.gz
├── cache/env/<environment-key>.tar.gz
├── cache/compiled/<gpu>-<runtime>-<source-sha>.tar.gz
├── models/
│   ├── standard-<revision>/
│   ├── flash-<revision>/
│   └── light-vae-<revision>/
└── results/<job-id>/
```

代码、模型和环境都必须使用完整 SHA/revision，不能使用 `latest`。下载先写 staging，
校验文件清单、总字节数和关键 SHA256 后再发布完成 marker。消费者同时检查 marker 和
关键权重 index，不能只判断目录存在。

为缩短 cold start：

- venv、pip/Hugging Face 缓存按依赖和模型 revision 共享；
- Triton/TorchInductor 缓存按 GPU 架构、Torch/CUDA、模型源码和仓库 SHA 隔离；
- 模型从共享盘复制到节点本地盘后再加载，避免大量随机读持续打到 FUSE；
- Standard 只同步推理需要的根 metadata 和组件，不能无差别下载仓库中的
  `comfyui/**` 大制品；
- 源码扩展在本地盘构建，成功后归档整个 venv；不要在 OSS FUSE 中构建 wheel。

## 官方 540p 测速口径

仓库 `script/sample/speed_test/README.md` 是指标定义的真源。默认 540p 口径是
960×540、22 帧、24 fps，原生 960×544 canvas 中心裁切；Standard 为 W8A8 4 steps，
Flash 为训练版 W8A8 3 steps，二者都使用 light VAE。

测量要求：

- 单张空闲 H100；
- 固定 prompt 和 seed；
- 2 次 warmup + 5 次正式测量；
- 固定已选中的 FA3、Triton INT8 GEMM 和 adaptive light-VAE tiles；
- 以每次调用的 `DiT CUDA + decoder CUDA` 之和的中位数选取结果；
- 排除 conditioning、权重加载、首次编译和文件编码；
- 保留 `latency.json`、源码快照、硬件信息和 `source_changed_during_run`。

上海共享挂载准备好代码归档、环境归档和三个模型 revision 后，在单卡 DLC 中分别执行：

```bash
bash script/aliyun_h100_official_speed_test.sh
bash script/aliyun_h100_official_speed_test_flash.sh
```

脚本默认锁定本次已验证的源码和模型 revision。复测新 revision 时显式设置
`LYNNREAL_RELEASE_SHA`、`LYNNREAL_STANDARD_REV`、`LYNNREAL_FLASH_REV` 和
`LYNNREAL_LIGHT_VAE_REV`，并确保共享盘存在完全对应的归档和完成 marker。不要只改变量
却复用旧目录或旧编译缓存。

本次单卡 H100 的已验证 22 帧结果为：

| Model | DiT CUDA | Decoder CUDA | DiT + decoder | Generation + decode | Peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 703.4 ms | 115.9 ms | 819.3 ms | 887.1 ms | 54.0 GiB |
| Flash | 251.8 ms | 113.8 ms | 366.0 ms | 432.2 ms | 46.5 GiB |

这些值不能与完整 Job wall time混用；完整 wall 还包含环境解压、权重复制/加载、conditioning、
首次编译、音视频编码和上传。10 秒视频是功能样例，也不能直接与默认 22 帧官方测速表比较。

## 浏览器视频发布

模型的归档输出可能使用 H.264 High 4:4:4 RGB（`gbrp`/full range）。部分浏览器在开始
硬件解码后会把它误按 YUV 处理，表现为首帧正常、播放后变成绿色/洋红色。不要把网页
CSS 或模型输出误判为根因。

保留原始文件用于质量复核和 SHA256；对外网页另生成 H.264 High、`yuv420p`、BT.709、
limited range、faststart 的派生文件：

```bash
bash script/export_web_video.sh input.mp4 output.web.mp4
```

上传 OSS 时为 MP4 设置 `Content-Type: video/mp4`，页面使用派生文件。发布前用
`ffprobe` 确认 `pix_fmt=yuv420p`、`color_range=tv`、三项色彩元数据均为 `bt709`，并做
一次全量解码检查。网页更新后使用新 object key 或 query cache-buster，避免 CDN/浏览器
继续命中旧视频。

## Job 交付门禁

每次运行至少保留：

- PAI Job ID 和可点击控制台链接；
- region、workspace、quota、请求的 GPU/CPU/内存；
- 镜像 URI、代码 SHA、模型 revision、环境/编译缓存 key；
- `latency.json`、结果 SHA256、`READY`/`FAILED`；
- 释放时对 DLC、quota 和按量订单的最终状态查询。

生成视频、完整日志、模型权重、临时签名 URL、STS/ACR token 和个人 CLI profile 不进入
Git。大产物保留在共享存储/OSS，仓库只保存可复现它们的脚本、口径和 provenance。
