"""Deterministic matched target selection, frozen before any optimizer or new learner outcome."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def key(row):return hashlib.sha256(f"v32-target|{row['file']}|{row['action_index']}".encode()).hexdigest()
def balanced(pools,n):
    # Equal trajectory shares first; deterministic hash order fills shortfalls.
    queues=[sorted(pool,key=key) for pool in pools if pool]
    queues.sort(key=lambda pool:key(pool[0]));out=[];i=0
    while len(out)<n:
        progressed=False
        for pool in queues:
            if i<len(pool):out.append(pool[i]);progressed=True
            if len(out)==n:break
        assert progressed;i+=1
    return out
def main():
    assert not (W/'training_membership.json').exists() and not (W/'training_started.json').exists()
    p=read(W/'registration.json');cache=read(W/'cache_complete.json');assert cache['registration_sha256']==sha(W/'registration.json')
    uniform=[];local=[];files={};retention=[]
    for teacher in cache['teachers']:
        f=ROOT/teacher['file'];assert sha(f)==teacher['sha256'];files[teacher['file']]=sha(f)
        with np.load(f) as z:assert len(z['spatial_features'])==len(z['actions'])==teacher['controls']
        rows=[dict(file=teacher['file'],feature_field='spatial_features',action_index=i,teacher_key=teacher['key']) for i in range(9,teacher['controls'])]
        uniform.append(rows);local.append([r for r in rows if r['action_index']>=teacher['first_local_target_index']])
    for source in p['retention_sources']:
        f=ROOT/source['trace'];assert sha(f)==source['trace_sha256'];files[source['trace']]=sha(f)
        with np.load(f) as z:
            assert z['frozen_spatial_features'].ndim==3 and z['frozen_spatial_features'].shape[1:]==(4,768)
            assert z['frozen_spatial_features'].dtype==np.float32 and len(z['frozen_spatial_features'])==len(z['actions'])
            rows=[dict(file=source['trace'],feature_field='frozen_spatial_features',action_index=i,source_task=source['task_id']) for i in range(9,len(z['actions']))]
        retention.append(rows)
    n=min(128,sum(map(len,uniform)),sum(map(len,local)),sum(map(len,retention)));assert n>=16
    selected=dict(uniform_bc=balanced(uniform,n),local_bc=balanced(local,n),retention=balanced(retention,n))
    assert all(len({(r['file'],r['action_index']) for r in rows})==n for rows in selected.values())
    out=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),N=n,selected_targets=selected,files_sha256=files,uniform_available=sum(map(len,uniform)),local_available=sum(map(len,local)),retention_available=sum(map(len,retention)),teacher_episode_count=len(cache['teachers']),overlap_teacher_target_entries=len({(r['file'],r['action_index']) for r in selected['uniform_bc']} & {(r['file'],r['action_index']) for r in selected['local_bc']}),independent_unit='Teacher episodes/contexts, never feature frames; correlated shared prefixes remain correlated. Entries are distinct episode/index pairs, not claimed unique physical states.',selector_implementation_sha256=sha(__file__),no_new_student_task_outcomes=True)
    with (W/'training_membership.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('Frozen matched target memberships; N=',n,'teacher trajectories=',len(cache['teachers']))
if __name__=='__main__':main()
