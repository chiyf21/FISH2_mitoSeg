# mito_video_repair_fast：计算优化版

**当前生产停止阈值已按用户新要求改为0.5（`inference_policy.json`，version=`1.1-score05`）。各方向首帧score<0.5停止并排除该帧；=0.5保留。原始v1及下面历史0.4性能报告不改写，不能把旧面板误认为0.5结果。输出目录若存在不同停止策略将拒绝混用。历史0.4金标准测试入口会明确拒绝在0.5下复用旧参考。**

基于已冻结`mito_video_repair_v1`。不改prompt、score阈值、模型、分辨率、追踪方向、crop自适应、原mask并集及最终26连通选择。基准代码快照位于`baseline/`，原版目录保持不变。

## 已实现

1. 原图空间块共享缓存：固定原生128×128列、32层磁盘块，CloudVolume底层chunk缓存；遇到黑层跨32层缓存边界查找真实有效邻层，插值与全列版本一致。8路并行源块读取。
2. CPU异步读取和预处理：当前SAM计算时，准备后续帧；生产worker还提前准备下一个实例的prefill和原图。预取不进行条件记忆更新，不改变停止点。
3. GPU图像特征缓存：每worker上限2GiB；key含原图身份（worker固定）、crop坐标、边长、Z和实际编码批次起点/大小。只有完全相同的输入及批次才复用；不复用decoder或Video条件记忆。
4. 编码批处理：保留原版8帧＋末批的形状及顺序，用并行CPU RGB转换、pinned memory、非阻塞上传，保留已编码特征在GPU以减少往返搬运。没有为了“合批”修改crop、Z或batch shape。
5. 核分割共享缓存：全局800窗口/400步长、原全图row-major8窗口批次、原float32概率均值融合与0.5阈值保持不变。精确窗口缓存与CPU预取；两台服务器核权重及代码SHA256相同。
6. 动态队列：空间顺序、SQLite原子领取，每卡一份长期驻留模型，任务不重复。不会仅凭超时判断worker死亡后盲目重跑；异常明确记failed，由检查进程后再恢复。
7. 并集/26连通/保存由CPU线程处理，与下一个实例GPU推理重叠。

服务只监听127.0.0.1，使用访问令牌；跨主机仅经SSH隧道。不得开放无认证原图接口。`io_access.token`、`queue_access.json`和隧道文件不能放入分发压缩包。

无损gzip缓存压缩也做了实测：节省约4.07倍存储，但冷核窗口计算由0.645秒变成1.207秒，已退出默认路径。可选环境变量`MITO_NUCLEAR_LOSSLESS_ARCHIVE=1`仅用于空间优先的归档，不作为本次默认提速方案。

## 冒烟结果（2026-11-03）

- 局部数据：12例原版＋24例优化版完整修复；24个优化结果全部与冻结参考逐体素一致。12个配对样本的推理耗时中位加速1.833倍。原版12例307秒，优化版24例248秒；样本数不同，不把两组wall之比称为加速倍数。
- 真实全局数据：同一64步SAM输入序列，原始logits哈希和score逐步一致。热缓存耗时8.197→4.114秒，1.992倍。这个测试`limited_run=true`，不是完整实例修复，不能用于声称全局实例已完成。
- 冷核窗口：24个全局窗口的float32概率与mask完全一致；最终无压缩默认路径0.801→0.541秒，约1.479倍。该计时的源原图块已有缓存，不能当成网络冷启动耗时。
- 特征复用：与原始CPU往返版本的特征逐比特一致；再次请求相同10帧命中缓存，不额外运行编码batch。随机24例crop不同，不能声称都能命中缓存。
- 插值跨块等价、batch-specific缓存淘汰和32并发领取无重复测试通过。

原mask只有5层的真实样本，原规则试跑已访问187个Z层仍未自然结束；此未完成试跑被明确中断，记录于`global_full_probe_incomplete.json`，没有生成正式修复结果，也没有偷偷给正式pipeline加追踪上限。

**全量ETA限制：** 局部1.8–2倍计算收益不能直接把全量预算减半。此前14–20小时没有充分覆盖全局长追踪、原图冷缓存及核模型成本，不能再作为可靠承诺。必须用真实完整全局实例的自然收尾分布和跨实例核缓存命中率校准。现有64步剖析只用于计算吞吐，不能证明平均实例步数。

## 使用入口

- `optimized_worker.py`：本地或全局生产worker；各自`CUDA_VISIBLE_DEVICES`选择GPU。
- `queue_service.py`：共享空间排序任务清单；默认本机18084、令牌认证。
- `build_global_queue.py`：从20k预填后清单构建两台服务器可读取的任务。只准备队列，不启动全量。
- `run_local_smoke.py`：局部配对完整冒烟。
- `profile_global_prefix.py`：有界全局性能剖析，明确不视为完成。
- `profile_cold_nucleus.py`：核概率/mask与冷核计算对照。
- `test_optimization_rules.py`：插值、缓存key及动态队列核验。

生产示例（先启动queue_service，各worker使用不同输出目录，结果由队列汇总）:

```bash
python queue_service.py --manifest jobs.json --out queue --port 18084
CUDA_VISIBLE_DEVICES=2 python optimized_worker.py \
  --queue http://127.0.0.1:18084 --token-file queue/queue_access.json \
  --worker 113-gpu2 --out run/113-gpu2 --source global \
  --raw-url http://127.0.0.1:18085 --raw-token-file io_access.token \
  --nuclear-code /path/to/JaneliaEM/organelle_detect/scripts/train_nuclear_unet2d.py \
  --nuclear-weight /path/to/JaneliaEM/outputs/train_nuclear_unet2d/best.pt
```

global worker的输入可带`original_mask_path112`、`prefill_lt200`及`expected_prepared_voxels`，在112按相同规则预填并核对结果。无GT参与生产修复。当前版本未启动14734例全量任务。
