"""Audited orchestration only: one unstarted arm, truthful aggregation, frozen eval/analysis."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path('F:/robot').resolve();CODE=ROOT/'rsi_group_meeting/v40/code'
RCODE=ROOT/'rsi_group_meeting/v40/recovery_code'
W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control'
R=W/'recovery_namespace_guard_20261102'
sys.path[:0]=[str(CODE),str(ROOT/'rsi_group_meeting')]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def check_addendum():
    a=read(R/'recovery_addendum_frozen_before_remaining_arm.json')
    assert sha(W/'registration.json')==a['registration_sha256']
    for name,h in a['recovery_sources_sha256'].items():assert sha(RCODE/name)==h,name
    for path,h in a['preserved_original_evidence_sha256'].items():assert sha(ROOT/path)==h,path
    assert a['scientific_changes']==[] and a['remaining_arm']=='coverage_KL'
    return a
def status(stage,**extra):
    p=R/'owner_status.working.json';p.write_text(json.dumps(dict(updated_utc=now(),actual_owner_pid=os.getpid(),stage=stage,original_owner_remains_failed=True,work_demonstrated=False,**extra),ensure_ascii=False,indent=2),'utf8');p.replace(R/'owner_status.json')
def main():
    assert Path(__file__).resolve()==RCODE/'run_namespace_recovery_sequence_v40.py','Preparation only; root review/install first.'
    from state_coverage_study_v40 import verify
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
    began=time.perf_counter();phase='preflight'
    with exclusive('state_coverage_sequence_owner_v40'),awake():
        a=check_addendum();verify()
        assert not (R/'owner_attempt.json').exists() and not (R/'remaining_arm').exists()
        assert not (W/'arms/coverage_KL').exists() and not (W/'owner_complete.json').exists()
        assert not (W/'training_complete.json').exists() and not (W/'evaluation_started.json').exists()
        assert not (W/'analysis_attempt.json').exists() and not (W/'analysis').exists()
        assert read(W/'owner_phase_timings/two_fixed_trainings.json')['return_code']==1
        save(R/'owner_attempt.json',dict(started_utc=now(),actual_python_pid=os.getpid(),implementation_sha256=sha(__file__),addendum_sha256=sha(R/'recovery_addendum_frozen_before_remaining_arm.json'),automatic_retry=False,scope='One previously unstarted coverage arm only; no oldA retraining. Then CPU aggregation and unchanged fixed48 evaluation/frozen analysis once.'))
        commands=[('remaining_coverage_training','envs/dmpel/Scripts/python.exe',RCODE/'run_remaining_coverage_namespace_v40.py'),('truthful_two_arm_aggregation','envs/robot/Scripts/python.exe',RCODE/'aggregate_recovered_trainings_v40.py'),('fixed48_evaluation','envs/xvla/Scripts/python.exe',CODE/'evaluate_state_coverage_v40.py'),('frozen_analysis_once','envs/robot/Scripts/python.exe',CODE/'analyze_state_coverage_v40.py')]
        returns={}
        try:
            for phase,python,script in commands:
                check_addendum();verify();command=[str(ROOT/python),'-u',str(script)]
                log=R/'logs'/f'{phase}.log';log.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
                with log.open('x',encoding='utf8') as f:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    status(phase,command=command,launcher_pid=child.pid,launcher_is_not_actual_python_identity=True,log=str(log));code=child.wait()
                elapsed=time.perf_counter()-started
                save(R/'owner_phase_timings'/f'{phase}.json',dict(phase=phase,command=command,return_code=code,external_command_wall_seconds=elapsed,definition='Spawn through childexit; contains internal timers. Do not add contained timers.'))
                returns[phase]=code;assert code==0,f'{phase}: preserve outputs and audit, no automatic retry'
                verify();check_addendum()
                if phase=='remaining_coverage_training':
                    c=read(R/'remaining_arm/one_remaining_arm_complete.json');assert c['arm']=='coverage_KL' and c['optimizer_steps']==2000
                    t=read(R/'remaining_arm/arms/coverage_KL/training_complete.json');assert t['steps']==2000 and t['parity_queries']==8
                elif phase=='truthful_two_arm_aggregation':
                    c=read(W/'training_complete.json');assert c['training_runs']==2 and c['total_optimizer_steps']==4000 and c['within_attempt_elapsed_seconds'] is None
                    c=read(R/'aggregation_complete.json');assert c['registration_sha256']==sha(W/'registration.json') and c['addendum_sha256']==sha(R/'recovery_addendum_frozen_before_remaining_arm.json')
                elif phase=='fixed48_evaluation':
                    c=read(W/'evaluation_complete.json');assert c['physical_episodes']==48 and c['settling_controls']==480
                    assert read(W/'evaluation_worker_close.json')['closed']
                else:assert len(read(W/'analysis/coverage_readout.json')['rows'])==48
            elapsed=time.perf_counter()-began
            save(R/'owner_complete.json',dict(completed_utc=now(),all_four_recovery_commands_returned_zero=True,return_codes=returns,registration_sha256=sha(W/'registration.json'),addendum_sha256=sha(R/'recovery_addendum_frozen_before_remaining_arm.json'),original_owner_still_failed=True,original_failed_training_command_external_seconds=read(W/'owner_phase_timings/two_fixed_trainings.json')['external_command_wall_seconds'],recovery_owner_observed_sequence_wall_seconds=elapsed,definition='Recovery owner internal interval includes its four phase command timers, excludes original failed attempt and audit gap. Original plus recovery timers are disjoint; none is wholeproject time.',whole_project_wall_seconds=None,independent_raw_cost_visual_review_still_required=True,work_demonstrated=False))
            status('recovered_finite_state_coverage_control_complete_requires_root_review')
        except BaseException as error:
            save(R/'owner_failure_requires_audit.json',dict(failed_utc=now(),phase=phase,error=repr(error),traceback=traceback.format_exc(),completed_return_codes=returns,recovery_owner_elapsed_seconds=time.perf_counter()-began,automatic_retry=False))
            status('stopped_requires_audit_no_automatic_retry',failed_phase=phase,error=repr(error));raise
if __name__=='__main__':main()
