# 泰国 B300 Eval 交接（天鹏）

## 交付边界

这套入口面向 LynnReal-Omni 的 10 秒、768p、24 fps T2V eval：

- 每个 PAI-DLC Job 固定一个模型：`standard` 或 `flash`。
- 每个 Job 只申请一张 B300；模型在 Job 内只加载一次，再顺序生成 manifest 中的所有样本。
- Standard 固定 W8A8、4 steps；Flash 固定训练版 W8A8、3 steps。
- 固定使用已在 B300 实测胜出的 native SDPA、adaptive light-VAE tiles 和 Triton INT8 GEMM。
- 输入是 JSONL；输出是稳定的 `samples/`、`metadata/`、`results.jsonl` 和 `READY` 契约。

当前入口只覆盖 T2V。TI2V、V2V、Ref2V 的 reference media 语义不同，不应悄悄塞进同一 schema；需要时单独扩展并做真机验证。

## 1. 准备输入 manifest

每行必须包含唯一 `id`、非空 `prompt`；`seed` 可省略，默认 0：

```json
{"id":"case_000001","prompt":"integrated_multimodal_description: ...","seed":77}
{"id":"case_000002","prompt":"integrated_multimodal_description: ...","seed":78}
```

约束：

- `id` 长度 1–128，只能使用字母、数字、点、下划线和横线，并以字母或数字开头。
- `id` 直接成为视频文件名，因此必须唯一。
- `seed` 必须是非负整数。
- prompt 原样进入模型并写回结果，eval 侧可以用 `id` join，不要依赖输出顺序或文件修改时间。

仓库内的 `test/eval_t2v.example.jsonl` 可以直接用于 dry-run。

如果天鹏现有数据是 `sample_id/caption` 字段，只需在提交前转换一次，例如：

```bash
jq -c '{id:(.sample_id|tostring), prompt:.caption, seed:(.seed // 0)}' \
  existing_eval.jsonl > lynnreal_eval.jsonl
```

## 2. 本地只做 dry-run

本地不需要安装模型环境，也不会下载权重。只需要已登录的阿里云 CLI、`jq`、`git` 和可访问的 PAI profile：

```bash
ALIYUN_PROFILE=<your-profile> \
bash script/aliyun_thailand_b300_submit.sh --dry-run \
  --eval-manifest test/eval_t2v.example.jsonl \
  --variant flash
```

dry-run 会在申请 GPU 前完成以下检查：

- JSONL 可解析，字段、ID 和 seed 合法；
- 资源固定为泰国 `ap-southeast-7`、workspace `384`、quota `quota1betfyi1ilu`；
- CPFS dataset 固定挂载为 `d-ak7t158lhzaq7kdwp3`；
- 请求只包含一张 GPU；
- manifest SHA256 已进入 bundle ID，同名 bundle 不会误复用另一份输入。

## 3. 提交 eval

Flash：

```bash
ALIYUN_PROFILE=<your-profile> \
JOB_NAME=lynnreal-flash-eval-<tag> \
PAI_MAX_MINUTES=360 \
bash script/aliyun_thailand_b300_submit.sh --submit \
  --eval-manifest /absolute/path/lynnreal_eval.jsonl \
  --variant flash
```

Standard：

```bash
ALIYUN_PROFILE=<your-profile> \
JOB_NAME=lynnreal-standard-eval-<tag> \
PAI_MAX_MINUTES=720 \
bash script/aliyun_thailand_b300_submit.sh --submit \
  --eval-manifest /absolute/path/lynnreal_eval.jsonl \
  --variant standard
```

默认推荐顺序提交两个 Job，全程只占一张卡。如果 eval 时效优先，可以同时提交 Standard 和 Flash 两个 Job，总共占两张卡；不要给一个 Job 申请两张卡，因为当前入口不是分布式推理，多申请的卡不会提速。

按已经测得的 10 秒样本速度估算，单条 Flash 核心生成约 35 秒，Standard 约 98 秒；总时长还需加首次加载、一次 warmup、prompt conditioning 和视频编码。`PAI_MAX_MINUTES` 要根据样本数留余量，避免完成大部分样本后被平台超时终止。

不同 prompt token 数会形成不同的 INT8 行数。某个行数第一次出现时可能包含 Triton autotune/编译，单条 wall 会暂时高于热态基线；缓存会跨 Job 持久化。正式速度比较仍使用固定 shape 的官方 speed-test，不要拿 eval 中首个新 shape 的 wall 做性能回归判断。

## 4. 结果与 eval 接口

提交响应中的 `JobId` 是唯一 run ID。结果同时写到：

```text
CPFS: /cpfs/world-model/lynnreal-omni/results/<JobId>/
OSS:  oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/<JobId>/
```

成功目录：

```text
<JobId>/
├── eval/
│   ├── requests.jsonl
│   ├── samples/<id>.mp4
│   ├── metadata/<id>.json
│   ├── results.jsonl
│   ├── run.json
│   ├── summary.json
│   └── READY
├── int8-gemm-validation.json
├── runtime.env
├── bootstrap.log
└── READY
```

eval 侧只需要读取 `eval/results.jsonl`。每行包含：

- `id`、原 prompt 和 seed；
- `video` 的绝对 CPFS 路径与 SHA256；
- variant、W8A8、steps、宽高、帧数、fps 和时长；
- 单样本 DiT、decoder、wall 和峰值显存 timing。

推荐适配方式：

1. 等待 Job 状态为 `Succeeded`。
2. 同时确认根目录和 `eval/` 都有 `READY`，且不存在 `FAILED`。
3. 逐行读取 `results.jsonl`，用 `id` 与天鹏现有 eval dataset join。
4. 把 `video` 传给现有 scorer；不要从目录顺序猜 sample ID。
5. 保存 `summary.json`、`runtime.env` 和 Job ID，保证分数可追溯到模型、kernel、输入 manifest 和运行环境。

Job 运行中只有 `progress.jsonl`，成功收尾时才原子改名为 `results.jsonl` 并写 `READY`。因此 `progress.jsonl` 只能用于观察进度，不能作为正式完整 eval 输入。

## 5. 已持久化内容与 cold start

天鹏不需要再次下载 125 GB 左右的模型，也不需要在每个 Job 重新 pip 安装：

```text
/cpfs/world-model/lynnreal-omni/
├── models/
│   ├── standard-f1d6990e23496bef6c7d55bdcfe2925df2508325/
│   ├── flash-9950cfc882b0e6b8600fbc49b58dc1d82d7a435b/
│   └── light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a/
├── cache/env/py312-torch2121-cu130-lynnreal-0384eca3d5d7482d.tar.gz
├── cache/shared/{huggingface,pip}/
└── cache/compiled/
```

启动流程会把小型 Python overlay 解压到节点本地盘，权重通过只读 symlink 使用 CPFS 文件，Triton/Inductor 缓存继续留在 CPFS。缓存 key 由模型 Python 源码、`requirements.txt` 和固定 Diffusers 内容哈希生成；只修改文档或提交器不会让缓存失效。模型运行时代码或依赖内容变化时会使用新缓存，避免错误复用旧二进制。

## 6. 已验证的批量 smoke

fork commit `351f41e16ec29f9ae51b62ac5d531107e270e688` 已完成两条 Flash 样本的端到端 smoke：

- PAI Job：`dlc1qjld6yy3gu92`，状态 `Succeeded`，总时长 853 秒，单卡。
- OSS：`oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/dlc1qjld6yy3gu92/`。
- `completed_samples=2`，两个唯一 ID、两个视频 SHA256，均为 1344×768、240 帧、24 fps、10 秒。
- `source_changed_during_run=[]`；INT8 门禁 `fallback_shapes=0`；日志中无 PyTorch GEMM fallback 或 traceback。
- 热态第一条样本 DiT 32.990 秒、decoder 1.762 秒、generation wall 35.440 秒，与官方 Flash 基线一致。
- 第二条使用此前未见的 prompt row shape，首次 autotune 后 wall 为 69.940 秒；该开销进入持久化编译缓存，不代表模型重新加载。

日志只出现一次 checkpoint shard 加载；当前批量脚本还会在 `summary.json` 显式记录 `model_loads=1`。后续相同运行时源码和相同 shape 会直接复用本次缓存。

## 7. 不要踩的坑

- 不要启用仓库 `_flash_3`：当前 FA3 是 Hopper 路径，B300 已验证的最佳配置是 native SDPA。
- 不要绕过 `validate_int8_gemm.py`：它会阻止 Triton 编译失败后静默回退 `torch._int_mm`，否则速度和精度口径都会变化。
- 不要把 PAI UI/API 的内部设备名 `NVIDIA L20D` 当成 L20；本次交付平台的实测设备是 275040 MiB、compute capability 10.3，资源运维口径为 B300。
- 不要把 speed-test timing 与 eval 总 wall 混用。官方速度基线排除 conditioning、加载和文件编码；eval `results.jsonl` 同时保留 kernel timing，Job wall 还包含这些外围成本。
- 不要改模型 revision 后继续复用旧模型目录。模型、overlay 和编译缓存都是内容/版本寻址；刷新权重必须重新生成 manifest 并做 SHA256 校验。
- 不要只看 PAI `Succeeded`。正式交付还必须检查 `READY`、样本数以及 `summary.json` 中 `source_changed_during_run=[]`。
- 不要把 OSS 临时签名 URL、STS、ACR token 或个人 profile 写入仓库；提交器只在内存里取得临时凭证。

## 8. 本次应该长期沉淀什么

应进入 Git 的内容：

- Blackwell/Triton INT8 dtype 修复和真实投影 shape 的正确性门禁；
- 单卡 PAI 提交器、环境/模型/编译缓存的版本化规则；
- eval JSONL 输入与结果输出契约；
- 官方 speed-test 与业务 eval 的口径区别、已验证 backend 和故障门禁；
- 模型 revision、依赖版本和结果 provenance。

不应进入 Git 的内容：

- 生成视频、完整运行日志和几百 MB 的 benchmark 产物；
- 临时下载 URL、云端临时凭证、个人 CLI profile；
- 某次 Job 的临时工作目录。

这些大产物继续以 CPFS 为热存储、OSS 为长期归档；Git 只保留可复现它们的代码、清单和文档。
