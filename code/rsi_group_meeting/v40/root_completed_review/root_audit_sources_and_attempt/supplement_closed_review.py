"""Finite states, complete process identity, cost fields and truthful status pointers."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,hashlib,shutil
import numpy as np
ROOT=Path('F:/robot');W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control';R=W/'recovery_namespace_guard_20261102';OUT=ROOT/'rsi_group_meeting/v40/root_completed_review';C=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    began=time.perf_counter();review=read(OUT/'review.json');assert review['all48_raw_success_initial_three_actor_inputs_event_query_action_hashes_checked'] and len(review['raw_rows'])==48
    observed=read(C/'root_review_external_completion.json');assert observed['return_code']==0
    command="$ErrorActionPreference='Stop';$v40Live=@(Get-CimInstance Win32_Process -ErrorAction Stop -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -match 'run_namespace_recovery_sequence_v40.py|run_remaining_coverage_namespace_v40.py|aggregate_recovered_trainings_v40.py|train_state_coverage_v40.py|run_state_coverage_sequence_v40.py|evaluate_state_coverage_v40.py|worker_state_coverage_v40.py|analyze_state_coverage_v40.py' } | Select-Object ProcessId,CommandLine);[PSCustomObject]@{query_succeeded=$true;processes=$v40Live}|ConvertTo-Json -Depth 4"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True);identity=json.loads(result.stdout);assert identity['query_succeeded'] and identity['processes']==[]
    counts={k:0 for k in ['privileged_sim_states_for_replay_only','robot_states']}
    for row in review['raw_rows']:
        with np.load(W/'evaluation/results'/(row['key']+'.npz'),allow_pickle=False) as z:
            for k in counts:assert np.isfinite(z[k]).all();counts[k]+=len(z[k])
    assert counts['privileged_sim_states_for_replay_only']==counts['robot_states']==18069
    refresh=read(W/'pre_evaluation_exposure_refresh/receipt.json');cost=read(OUT/'cost_increment.json')
    supplementary=dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),main_raw_review_sha256=sha(OUT/'review.json'),all_v40_scientific_and_recovery_python_processes=identity['processes'],all_actual_query_attempts=review['query_attempts_and_samples'],all_action_samples_completed=review['query_attempts_and_samples'],all_saved_sim_and_robot_states_finite_rows=counts,exposure_refresh_CPU_seconds_already_inside_evaluation_external_interval=refresh['observed_cpu_io_seconds'],root_raw_review_external_CPU_seconds=observed['external_CPU_review_command_seconds'],root_raw_review_internal_CPU_seconds_already_inside_external=review['CPU_observed_seconds'],CPU_observed_supplementary_seconds=time.perf_counter()-began,new_model_gradient_encoder_physics_calls=0,whole_project_or_human_elapsed_seconds=None)
    save(OUT/'supplementary_audit.json',supplementary)
    originals=[]
    for f in C.iterdir():
        if f.is_file() and f.suffix in ['.py','.json','.log']:
            dest=OUT/'root_audit_sources_and_attempt'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(f,dest);assert sha(f)==sha(dest);originals.append({'file':str(dest),'sha256':sha(dest)})
    save(OUT/'root_audit_archive.json',originals)
    for rel in ['reports/current-research-status.json','rsi_group_meeting/v30/研究进度.json']:
        p=ROOT/rel;x=read(p);x.update(updated_utc=datetime.now(timezone.utc).isoformat(),stage='v40_all48_frozen_analysis_and_raw_cost_review_complete_visual_and_verdict_pending',active_gpu_experiment=False,active_robot_task_collection=False,current_pipeline_running=False,new_training_started=False,new_training_runs_completed=6,new_training_runs=6,current_phase_completed_training_runs=2,current_phase_completed_episodes=48,active_phase_completed_cases=48,completed_development_episodes=140,completed_new_development_task_episodes=140,new_robot_task_episodes=140,current_actual_python_pid=None,actual_python_pid=None,current_actual_python_creation_local=None,actual_python_creation_local=None,current_pipeline_sequence_pid=None,current_pipeline_sequence_creation_local=None,actual_worker_pid=None,current_actual_worker_pid=None,current_worker_pid=None,actual_worker_creation_local=None,current_process_reverification=[],current_root_closed_review=str(OUT/'review.json'),latest_cost_increment=str(OUT/'cost_increment.json'),current_scoped_completed_trial_rows=cost['cumulative_scoped_completed_trial_rows'],current_scoped_task_controls=cost['cumulative_scoped_task_controls'],current_scoped_settling_controls=cost['cumulative_scoped_settling_controls'],current_visual_review_complete=False,current_verdict_pending=True,current_phase_complete=True,work_demonstrated=False,new_method_work_demonstrated=False,original_owner_failed=True)
        temp=p.with_name(p.name+'.v40_closed_review_working');assert not temp.exists();temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8');temp.replace(p)
    print(json.dumps({'status':'RAW_REVIEW_AND_SUPPLEMENT_COMPLETE_VISUAL_PENDING','new_rows':48,'new_task_controls':18069,'cumulative_rows':cost['cumulative_scoped_completed_trial_rows'],'new_model_physics_calls':0,'supplement_CPU_seconds':supplementary['CPU_observed_supplementary_seconds']}))
if __name__=='__main__':main()
