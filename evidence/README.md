# 证据阅读入口

1. **[root合并判断](v40/root_verdict/合并判断.txt)**、[结构化verdict](v40/root_verdict/verdict.json)：明确停止扩大当前配方及下一项未注册诊断。
2. **[48条逐回合读数](v40/analysis/episode_rows.csv)**、[冻结完整汇总](v40/analysis/coverage_readout.json)、[48份原始小result](v40/raw48_result_json/)。
3. [预先注册](v40/registration.json)、[冻结分析标记](v40/analysis_frozen_before_outcomes.json)、[固定未来条件](v40/fixed_future_conditions.json)、训练成员/抽样文件。
4. [原始独立复核](v40/root_completed_review/review.json)、[成本增量](v40/root_completed_review/cost_increment.json)、[视觉签收](v40/visual/root_manual_visual_review.json)、[12张视频关键帧拼图](v40/visual/)。
5. [两名agent独立判断](v40/agent_verdicts/)、[失败原件](v40/training_failure.json)、[恢复附录及四阶段收据](v40/recovery/)。
6. [v22历史完整迁移验证](v22/)、[v38历史固定BC](v38/)。它们与v40是不同实验，不混作同一分母。

这些是已闭合原文件的逐字节复制；发布组装没有运行训练/模拟或再次执行冻结analyzer。范围说明见[证据范围说明](证据范围说明.txt)，完整快照清单见[full_snapshot_manifest.json](full_snapshot_manifest.json)。

48原始result引用的轨迹、视频、逐query输入、初始传感器包并未全部随仓库上传。这些文件在本地保存；公开root review记录相应哈希与原始核验。12张拼图覆盖全部48视频的固定帧位，不代表人工逐帧看完全视频。没有缺省补造未测值：旧A/source的900步值仍为null。

仅CPU核验本审阅快照：在仓库根目录执行`python tools/verify_review_bundle.py`。它核对文件SHA、源码注册身份、48条小结果与已发布读数，不能代替未附完整资产的端到端复现。
