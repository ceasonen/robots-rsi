"""One owner, exactly two fixed trainings then48 evaluations then one frozen analysis."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,time,subprocess,traceback
ROOT=Path('F:/robot').resolve();sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/v40/code/run_state_coverage_sequence_v40.py','Preparation only; root must review/copy.'
    began=time.perf_counter()
    from state_coverage_study_v40 import W,verify
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
    def status(stage,**extra):
        f=W/'owner_status.working.json';f.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),owner_actual_pid=os.getpid(),stage=stage,work_demonstrated=False,**extra),ensure_ascii=False,indent=2),'utf8');f.replace(W/'owner_status.json')
    with exclusive('state_coverage_sequence_owner_v40'),awake():
        p=verify();assert len(p['cases'])==48 and len(p['conditions'])==16
        assert not (W/'owner_attempt.json').exists() and not (W/'training_started.json').exists()
        assert not list((W/'evaluation/results').glob('*')) and not (W/'evaluation_started.json').exists()
        assert not (W/'analysis').exists() and not (W/'analysis_attempt.json').exists()
        regsha=sha(W/'registration.json')
        save(W/'owner_attempt.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),actual_python_pid=os.getpid(),registration_sha256=regsha,implementation_sha256=sha(__file__),scope='Exactly2x2000 fixed standardBC trainings,48paired rollouts,1 frozen descriptive analysis; no oldscience restart, no extrahead scoring, no automatic retry.',automatic_retry=False))
        commands=[('two_fixed_trainings','envs/dmpel/Scripts/python.exe','train_state_coverage_v40.py'),('fixed48_evaluation','envs/xvla/Scripts/python.exe','evaluate_state_coverage_v40.py'),('frozen_analysis_once','envs/robot/Scripts/python.exe','analyze_state_coverage_v40.py')]
        phase='initializing'
        try:
            for phase,python,script in commands:
                verify();log=ROOT/'reports'/('state-coverage-v40-'+phase+'.log');assert not log.exists()
                command=[str(ROOT/python),'-u',str(ROOT/'rsi_group_meeting/v40/code'/script)];started=time.perf_counter()
                with log.open('x',encoding='utf8') as f:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    status(phase,launcher_pid=child.pid,launcher_pid_is_not_actual_python_identity=True,command=command,log=str(log));code=child.wait()
                save(W/'owner_phase_timings'/f'{phase}.json',dict(phase=phase,command=command,return_code=code,external_command_wall_seconds=time.perf_counter()-started,definition='Before childspawn through childexit, contains internal timers. Do not add contained timers.'))
                assert code==0,phase+' failed: preserve all files and audit before recovery'
                verify()
                if phase=='two_fixed_trainings':
                    t=json.loads((W/'training_complete.json').read_text('utf8'));assert t['training_runs']==2 and t['total_optimizer_steps']==4000
                    actors=json.loads((W/'trained_actors.json').read_text('utf8'));assert set(actors)=={'oldA_KL','coverage_KL'} and all(x['steps']==2000 for x in actors.values())
                elif phase=='fixed48_evaluation':
                    t=json.loads((W/'evaluation_complete.json').read_text('utf8'));assert t['physical_episodes']==48 and t['settling_controls']==480
                    assert json.loads((W/'evaluation_worker_close.json').read_text('utf8'))['closed']
                elif phase=='frozen_analysis_once':
                    assert len(json.loads((W/'analysis/coverage_readout.json').read_text('utf8'))['rows'])==48
            elapsed=time.perf_counter()-began
            save(W/'owner_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=regsha,all_three_commands_returned_zero=True,owner_observed_sequence_wall_seconds=elapsed,definition='Internal main interval contains all three command timers and verification/status intervals, excludes initial Pythonstartup and final receiptwrites; not wholeproject time.',independent_root_raw_cost_visual_review_still_required=True,work_demonstrated=False))
            status('finite_state_coverage_control_complete_requires_root_review',owner_observed_sequence_wall_seconds=elapsed)
        except BaseException as error:
            status('stopped_requires_audit_no_automatic_retry',failed_phase=phase,error=repr(error),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
