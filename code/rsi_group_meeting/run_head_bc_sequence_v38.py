"""One external owner for the finite full-teacher-head-BC baseline. No automatic retries."""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, time, subprocess, traceback

ROOT = Path('F:/robot').resolve()
sys.path.insert(0, str(ROOT / 'rsi_group_meeting'))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    with Path(p).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)

def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/run_head_bc_sequence_v38.py','Preparation only; root must review/copy before execution.'
    owner_started = time.perf_counter()
    from head_bc_study_v38 import W, verify
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    def status(stage, **extra):
        working = W / 'owner_status.working.json'
        working.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), owner_actual_pid=os.getpid(), stage=stage, work_demonstrated=False, **extra), ensure_ascii=False, indent=2), 'utf8')
        working.replace(W / 'owner_status.json')
    with exclusive('head_bc_sequence_owner_v38'), awake():
        verify()
        assert not (W / 'owner_attempt.json').exists()
        assert not (W / 'training_started.json').exists()
        assert not list((W / 'evaluation/results').glob('*'))
        assert not (W / 'analysis').exists() and not (W / 'analysis_attempt.json').exists() and not (W / 'offline_fit').exists()
        reg_sha = sha(W / 'registration.json')
        save(W / 'owner_attempt.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), owner_actual_pid=os.getpid(), implementation_sha256=sha(__file__), registration_sha256=reg_sha, scope='Own exactly one fixed2000-step training,12 evaluations, frozen analysis and3291 new-head offline forwards. Owner is orchestration, not a scientific-method amendment.', automatic_retry=False))
        (W / 'owner_phase_timings').mkdir()
        phase = 'initializing'
        commands = [
            ('single_fixed_training', 'envs/dmpel/Scripts/python.exe', 'train_head_bc_v38.py'),
            ('fixed12_evaluation', 'envs/xvla/Scripts/python.exe', 'evaluate_head_bc_v38.py'),
            ('frozen_analysis_once', 'envs/robot/Scripts/python.exe', 'analyze_head_bc_v38.py'),
            ('new_head_offline_once', 'envs/dmpel/Scripts/python.exe', 'score_head_bc_offline_v38.py'),
        ]
        try:
            for phase, runtime, script in commands:
                verify()
                log = ROOT / 'reports' / ('full-teacher-head-bc-v38-' + phase + '.log')
                assert not log.exists()
                status(phase, log=str(log))
                command = [str(ROOT / runtime), '-u', str(ROOT / 'rsi_group_meeting' / script)]
                started = time.perf_counter()
                with log.open('x', encoding='utf8') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    status(phase, launcher_pid=child.pid, launcher_pid_is_not_a_claim_of_actual_python_pid=True, log=str(log))
                    code = child.wait()
                timing = dict(phase=phase, command=command, return_code=code, external_command_wall_seconds=time.perf_counter()-started, definition='Before spawn through child exit. Includes this command internal timers/startup/close. Never add internal timers to this value.')
                save(W / 'owner_phase_timings' / (phase + '.json'), timing)
                assert code == 0, phase + ' failed; preserve files/logs and audit before any recovery'
                verify()
                def result(path): return json.loads((W / path).read_text('utf8'))
                if phase == 'single_fixed_training':
                    assert result('training_complete.json')['optimizer_steps'] == 2000
                    assert result('zero_B_functional_parity_complete.json')['total_queries'] == 8
                elif phase == 'fixed12_evaluation':
                    assert result('evaluation_complete.json')['physical_episodes'] == 12
                    assert result('evaluation_worker_close.json')['closed']
                elif phase == 'frozen_analysis_once':
                    assert len(result('analysis/head_readout.json')['rows']) == 12
                elif phase == 'new_head_offline_once':
                    offline = result('offline_fit/summary.json')
                    assert offline['new_head_density_forwards'] == 3291 and sum(offline['old_actor_rows_reused'].values()) == 13164
            elapsed = time.perf_counter()-owner_started
            save(W / 'owner_complete.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), owner_observed_sequence_wall_seconds=elapsed, registration_sha256=reg_sha, definition='Measured inside main before imports/initial verification through completion of all four child commands. Contains command/verification/status intervals. Excludes owner Python startup and final receipt writes; not a complete whole-project timer. Do not add contained phase/internal timers.', all_four_commands_returned_zero=True, root_outcome_review_still_required=True, work_demonstrated=False))
            status('finite_head_BC_sequence_complete_requires_root_review', owner_observed_sequence_wall_seconds=elapsed)
        except BaseException as error:
            status('stopped_requires_audit_no_automatic_retry', failed_phase=phase, error=repr(error), traceback=traceback.format_exc())
            raise

if __name__ == '__main__': main()
