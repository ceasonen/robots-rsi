import numpy as np
from .env import TASKS,PointMassEnv
from .teachers import teacher,parent
from .student import Student
STYLES=("aggressive","damped")
def collect_exact(task,style,seeds,budget):
 env=PointMassEnv(task);policy=teacher(style,env);rows=[];episodes=successes=steps=si=0
 while steps<budget:
  seed=int(seeds[si%len(seeds)]+episodes*7919);obs=env.reset(seed);done=False
  while not done and steps<budget:
   a=policy(obs);rows.append((obs.copy(),np.asarray(a).copy()))
   obs,_,done,info=env.step(a);steps+=1
  successes+=int(info["success"]);episodes+=1;si+=1
 return rows,{"steps":steps,"episodes":episodes,"teacher_successes":successes}
def eval_policy(task,policy,seeds):
 env=PointMassEnv(task);out=[]
 for seed in seeds:
  obs=env.reset(int(seed));done=False
  while not done: obs,_,done,info=env.step(policy(obs))
  out.append(int(info["success"]))
 return np.asarray(out,dtype=int)
def run_one(task_name,seed,budget=200,n_eval=30):
 task=TASKS[task_name];rng=np.random.default_rng(seed);seeds=rng.integers(1,2**31-1,size=1000)
 train=seeds[:300];val=seeds[300:300+n_eval];src=seeds[300+n_eval:300+2*n_eval];tgt=seeds[300+2*n_eval:300+3*n_eval]
 pilot_budget=budget//5;rem=budget-pilot_budget;pilot={};receipts={};scores={}
 for style in STYLES:
  pilot[style],receipts[style]=collect_exact(task,style,train,pilot_budget//2)
  scores[style]=float(eval_policy(task,Student(task.max_accel).fit(pilot[style]),val).mean())
 selected=max(STYLES,key=lambda s:(scores[s],-STYLES.index(s)));opposite=next(s for s in STYLES if s!=selected)
 def adaptive(style):
  more,receipt=collect_exact(task,style,train,rem)
  return pilot["aggressive"]+pilot["damped"]+more,receipt
 selected_rows,selected_receipt=adaptive(selected);inverted_rows,inverted_receipt=adaptive(opposite)
 fixed={s:collect_exact(task,s,train,budget) for s in STYLES}
 pp=parent(PointMassEnv(task));ps=eval_policy(task,pp,src);pt=eval_policy(task,pp,tgt)
 arms={"fixed_aggressive":fixed["aggressive"][0],"fixed_damped":fixed["damped"][0],
       "feedback_selected":selected_rows,"feedback_inverted":inverted_rows}
 results={}
 for name,rows in arms.items():
  st=Student(task.max_accel).fit(rows);s=eval_policy(task,st,src);t=eval_policy(task,st,tgt)
  results[name]={"source_success":int(s.sum()),"target_success":int(t.sum()),
   "source_repair":int(((ps==0)&(s==1)).sum()),"source_harm":int(((ps==1)&(s==0)).sum()),
   "target_repair":int(((pt==0)&(t==1)).sum()),"target_harm":int(((pt==1)&(t==0)).sum()),
   "training_transitions":len(rows)}
 return {"task":task_name,"seed":seed,"budget":budget,"selected":selected,"pilot_scores":scores,
  "parent_source":int(ps.sum()),"parent_target":int(pt.sum()),"results":results,
  "receipts":{"pilot":receipts,"selected":selected_receipt,"inverted":inverted_receipt,
   "fixed_aggressive":fixed["aggressive"][1],"fixed_damped":fixed["damped"][1]},
  "scope":"continuous-state inertial point-mass; not MuJoCo or robot evidence"}
def run_matrix(seeds=(3,7,11,17,23),budget=200,n_eval=30):
 return [run_one(t,s,budget,n_eval) for t in TASKS for s in seeds]
