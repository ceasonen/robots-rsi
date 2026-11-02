"""One external CPU raw-review observation; preserve first logs even on failure."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,time
C=Path(__file__).resolve().parent
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    source=C/'root_review_closed_v40.py'
    with source.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    save(C/'root_review_attempt.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=h,science_job=False,model_or_physics_calls=0))
    began=time.perf_counter()
    with (C/'root_review_stdout.log').open('x',encoding='utf8') as out,(C/'root_review_stderr.log').open('x',encoding='utf8') as err:
        completed=subprocess.run(['F:/robot/envs/robot/Scripts/python.exe','-B',str(source)],cwd='F:/robot',stdout=out,stderr=err)
    receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),return_code=completed.returncode,external_CPU_review_command_seconds=time.perf_counter()-began,internal_CPU_timer_is_contained=True,source_sha256=h,model_or_physics_calls=0)
    save(C/'root_review_external_completion.json',receipt);print(json.dumps(receipt));raise SystemExit(completed.returncode)
if __name__=='__main__':main()
