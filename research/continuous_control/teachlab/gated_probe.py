"""Development-only frozen-parent local control diagnostic.

A CPU-only inertial point mass, not a robot or RSS experiment.
Thresholds here are fixed exploratory controls, not an optimized method.
Usage: from research/continuous_control, python -m teachlab.gated_probe
"""
import json
from pathlib import Path
import numpy as np
from .env import TASKS, PointMassEnv
from .experiment import collect_exact, eval_policy
from .teachers import parent
from .student import Student

SEEDS=(503,509,521,523,541)
STYLES=("aggressive","damped")
THRESHOLDS=(0.35,0.60)


def run():
 results=[]
 for task_name,task in TASKS.items():
  for seed in SEEDS:
   rng=np.random.default_rng(seed)
   samples=rng.choice(np.arange(1,2**20),size=500,replace=False)
   train,source,target=samples[:300],samples[300:360],samples[360:420]
   frozen=parent(PointMassEnv(task))
   policies={"parent":frozen}
   for style in STYLES:
    rows,_=collect_exact(task,style,train,200)
    student=Student(task.max_accel).fit(rows)
    policies[f"global_{style}"]=student
    for threshold in THRESHOLDS:
     def intervention(obs,st=student,th=threshold,prior=frozen):
      return st(obs) if (obs[4]>=0.5 and np.linalg.norm(obs[2:4])>th) else prior(obs)
     policies[f"phase_velocity_{style}_{threshold}"]=intervention
   ps=eval_policy(task,policies["parent"],source,split="source")
   pt=eval_policy(task,policies["parent"],target,split="target")
   records={}
   for name,policy in policies.items():
    s=eval_policy(task,policy,source,split="source")
    t=eval_policy(task,policy,target,split="target")
    records[name]={
     "source_success":int(s.sum()),"target_success":int(t.sum()),
     "source_repair":int(((ps==0)&(s==1)).sum()),
     "source_harm":int(((ps==1)&(s==0)).sum()),
     "target_repair":int(((pt==0)&(t==1)).sum()),
     "target_harm":int(((pt==1)&(t==0)).sum())
    }
   results.append(dict(task=task_name,seed=seed,budget=200,panel_size=60,records=records))
 out=Path("results/gated_probe_exploratory.json")
 out.parent.mkdir(exist_ok=True)
 out.write_text(json.dumps({"scope":"CPU-only point mass, not a robot experiment","results":results},indent=2)+"\n",encoding="utf-8")
 print(out)
 for task_name in TASKS:
  rs=[r for r in results if r["task"]==task_name]
  print(task_name)
  for method in rs[0]["records"]:
   print(method,"target",sum(r["records"][method]["target_success"] for r in rs),
         "source",sum(r["records"][method]["source_success"] for r in rs))


if __name__=="__main__":
 run()
