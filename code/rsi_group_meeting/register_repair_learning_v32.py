"""Freeze a small, matched nonprivileged-student feasibility experiment after teacher audit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev';T=ROOT/'rsi_group_meeting/v31/handoff_factorial_dev'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    assert not W.exists(),'Preserve any previous registration/attempt; do not overwrite.'
    diagnosis=read(T/'analysis/diagnostic_readout.json');assert diagnosis['new_physical_episodes']==12 and diagnosis['independent_event_language_history_action_state_goal_initial_audit_pass']
    assert diagnosis['all_A_prefixes_exact'] and all(p['exact'] for p in diagnosis['home_learned_vs_home_low_common_physical_path'])
    teachers=[];tp=read(T/'registration.json')
    for e in tp['episodes']:
        f=T/'results'/f"{e['key']}.json";r=read(f)
        if not r['success_by900'] or r['instability_terminated']:continue
        files=[f,f.with_suffix('.npz'),T/r['events'],T/r['video'],T/r['initial_sensors']]
        teachers.append(dict(key=e['key'],context=e['context'],result_file=str(f.relative_to(ROOT)),files_sha256={str(x.relative_to(ROOT)):sha(x) for x in files},privileges='Goal0 trigger and declared home/noise factors; actual actions recorded. Not a deployable learned teacher.'))
    assert teachers,'No complete success exists: do not train successful-BC on a partial goal.'
    canonical_files=[dict(source=str(f.relative_to(ROOT)),copy=str((W/'teacher_canonical'/f.name).relative_to(ROOT)),sha256=sha(f)) for f in sorted((T/'canonical').iterdir()) if f.is_file()]
    assert canonical_files
    source=ROOT/'rsi_group_meeting/v7/dmpel_onpolicy_source';sourceplan=read(source/'registration.json');retention=[]
    for f in sorted((source/'results').glob('*.json')):
        r=read(f);assert r['registration_sha256']==sha(source/'registration.json')
        if not r['success'] or r['instability_terminated'] or r['task_id'] in [9,34]:continue
        trace=f.with_suffix('.npz');assert sha(trace)==r['trace_sha256']
        retention.append(dict(result_file=str(f.relative_to(ROOT)),result_sha256=sha(f),trace=str(trace.relative_to(ROOT)),trace_sha256=sha(trace),task_id=r['task_id'],behavior_actor_sha256=r['actor_sha256'],official_complete_task_success=True))
    assert retention and sourceplan['frozen_pretrained_parameters_sha256']==tp['frozen_pretrained_parameters_sha256']
    old=read(ROOT/'rsi_group_meeting/v22/fresh_transfer/registration.json');contexts=[]
    for task in [0,7]:
        setting=next(e['context']['setting'] for e in tp['episodes'] if e['context']['setting']['task_id']==task)
        for state in [11,12]:contexts.append(dict(context_id=f'libero_10_task{task}_original_init{state}_noise20265400',partition='Long_development',state_index=state,setting=setting))
    for task in [9,34]:
        setting=next(c['setting'] for c in old['contexts'] if c['partition']=='calibration' and c['task_id']==task and c['setting']['perturbation']=='original')
        for state in [7,8]:contexts.append(dict(context_id=f'libero_90_task{task}_original_init{state}_noise20265400',partition='source_regression',state_index=state,setting=setting))
    assert len(contexts)==8 and {r['task_id'] for r in retention}.isdisjoint({9,34})
    episodes=[]
    for context in contexts:
        actors=sorted(['parent','uniform_bc','local_bc'],key=lambda a:hashlib.sha256(f"v32-order|{context['context_id']}|{a}".encode()).hexdigest())
        episodes.extend(dict(key=context['context_id']+'__'+actor,context=context,actor=actor) for actor in actors)
    code=['register_repair_learning_v32.py','cache_repair_teachers_v32.py','repair_learning_worker_v32.py','select_repair_training_targets_v32.py','train_repair_bc_v32.py','evaluate_repair_learning_v32.py','analyze_repair_learning_v32.py','dmpel_actor_runtime_v7.py','dmpel_policy_ports_v6.py','dmpel_motor_core_v6.py','dmpel_onpolicy_objective_v7.py','policy_pipe_client_v7.py','evaluate_dmpel_source_probe_v6.py','paired_observation_v4.py','xvla_benchmark_v4.py','benchmark_v4.py','local_run_lock_v4.py']
    paths=['rsi_group_meeting/'+name for name in code]+['scripts/dmpel_native_backend_v6.py','scripts/check_dmpel_actor_runtime_v7.py','rsi_group_meeting/repair_learning_awake_v32.py']
    plan=dict(registered_utc=datetime.now(timezone.utc).isoformat(),scope='Two BC operations as a learning-signal feasibility check, not novelty, matched-RL superiority, R1/R2 or independent confirmation.',parent=tp['parent'],frozen_pretrained_parameters_sha256=tp['frozen_pretrained_parameters_sha256'],teacher_diagnosis_sha256=sha(T/'analysis/diagnostic_readout.json'),teacher_registration_sha256=sha(T/'registration.json'),teachers=teachers,all_successful_registered_teachers_included=True,teacher_failures_remain_charged=True,retention_sources=retention,source_registration_sha256=sha(source/'registration.json'),public_sensors=['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos'],cache_rule='Same teacher initial state, exact all public sensor hashes and executed action replay, full state/robot/goals/warnings equal every step. Encode original unmodified joint instruction with frozen spatial encoder, no head sampling/GT/stage features. Abort and preserve on mismatch; no silent rerender substitution. Cache replay costs are not independent samples.',target_selection='N=min(128,all uniform targets,local targets,retention targets),N>=16. Same N entries per operation and same source pool. Targets start at index9; local starts max(9,firstBindex-10),ends fulltasksuccess. Equal trajectory shares then deterministic SHA ordering, no endpoint or best epoch selection. Entry means episode/index, not independent physical state; correlated teacher prefixes disclosed.',training=dict(operations=['uniform_bc','local_bc'],initial_actor='parent',optimizer='AdamW',learning_rate=.0001,weight_decay=0,betas=[.9,.999],epsilon=1e-8,optimizer_steps=20,batch_target_count=32,teacher_count_per_step=16,retention_count_per_step=16,implementation='Batch1 gradient accumulation over the same32 drawn entries, fixed order teacher then retention, no sample-length normalization or epoch selection.',training_sampling_seed=20265450,rank=4,alpha=8,trainable_parameters=73728,trainable_surface='Six temporal qkv LoRA only; frozen spatial encoder/head/original tensors.',loss='Mean action NLL+.001 exact auxiliary component/action joint-KL to frozen original initializer.',max_grad_norm=1,checkpoint='Exactly step20, no selection',author_low_eval_noise=False),evaluation_contexts=contexts,evaluation_episodes=episodes,evaluation_policy_seed=20265400,physical_evaluation_episode_budget=24,maximum_evaluation_controls=21600,evaluation_settling_controls=240,evaluation='Original joint language, native public sensors/history, author learned Normal scales/canonical float32 GMM and random mixture. No stage trigger, home servo, GT input or privileged reset. Full original check_success by900 primary,by520 sensitivity; numerical failures kept.',stopping='Collect all24 after both declared twenty-step runs, then frozen analysis once. Never stop or change settings from partial outcomes. If neither BC makes any full Long repair, do not sweep LR/epochs/windows on this branch. If uniform is equal/better, do not claim local selection advantage.',exposure='Long task0/7 init11/12 and source cal9/34 init7/8 already exposed in v22. New noise seed is only a development replication, not held-out task confirmation. No inference/p value in tiny feasibility check.',known_preparation_exposure='This training recipe was drafted by research_agent before the factorial run. Root later saw first3 factorial completion log lines while checking infrastructure (two successes); all12 collection and its frozen rules continued unchanged. This new study is explicitly development after audited teachers, not blinded confirmation.',costs='All20 earlier task episodes +12 factorial +40preflight/10settling +10CPUqueries remain charged; cache all successful teachers, count all physical replays/spatial encodings and time. Two training runs20 steps each,320teacher+320source presentations each;24 fresh eval episodes up to21600+240 controls. Historical v22/v7 ledgers preserved; unknown end-to-end/load times null, not0.',implementation_sha256={path:sha(ROOT/path) for path in paths},new_method_work_demonstrated=False)
    plan['teacher_canonical_files']=canonical_files
    (W/'teacher_canonical').mkdir(parents=True)
    for item in canonical_files:shutil.copy2(ROOT/item['source'],ROOT/item['copy']);assert sha(ROOT/item['copy'])==item['sha256']
    save(W/'registration.json',plan)
    save(W/'analysis_frozen_before_learner_outcomes.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),analysis_implementation_sha256=sha(ROOT/'rsi_group_meeting/analyze_repair_learning_v32.py'),complete_learner_outcomes_at_freeze=0,new_learner_training_runs_at_freeze=0))
    print('Registered all successful teachers,2 matched BC operations,24 original-instruction development evaluations. No execution started.')
if __name__=='__main__':main()
