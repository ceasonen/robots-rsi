# 冻结实现快照

这里按原相对路径收录53份v40注册原创源、3份冻结恢复编排源，以及已有独立复核/CPU抽帧源码。见[manifest.json](manifest.json)与[运行边界](代码与运行边界.txt)。原注册170项实现中的117项外部/环境依赖仅列在[excluded_dependencies.json](excluded_dependencies.json)，未分发其字节。

文件按原字节复制，SHA与原注册对应，保留历史F:/robot和工作站路径。`code/`是公开审阅布局，不是原实验ROOT；本仓库缺模型、模拟资产、环境及轨迹，不能直接端到端复跑。不要直接运行注册器、历史owner或恢复器以“检查”代码；本地实验已经闭合。

当前核心入口供阅读：

- [冻结规格](rsi_group_meeting/v40/code/state_coverage_study_v40.py)
- [训练实现](rsi_group_meeting/v40/code/train_state_coverage_v40.py)
- [评估实现](rsi_group_meeting/v40/code/evaluate_state_coverage_v40.py)
- [冻结分析](rsi_group_meeting/v40/code/analyze_state_coverage_v40.py)
- [恢复编排](rsi_group_meeting/v40/recovery_code/run_namespace_recovery_sequence_v40.py)

原owner在第二次后端初始化时失败；恢复只补做当时尚未执行的第二臂，未重训首臂。原失败及完整恢复收据都在evidence。旧v21注册从未运行且已被v22更正，不能启动；旧v11/v14也不应重跑。文件名包含旧版本不代表它仍是活动实验。

依赖DMPEL、LIBERO等工作的归属和许可仍属于原作者。仓库不分发外部论文PDF、第三方源码树、权重或数据。
