import numpy as np
def teacher(style,env):
 if style=="aggressive": kp,kd=8.0,4.5
 elif style=="damped": kp,kd=5.5,5.0
 else: raise ValueError(style)
 def act(obs):
  return np.clip(kp*(env.goal-obs[:2])-kd*obs[2:4],-env.task.max_accel,env.task.max_accel)
 return act
def parent(env):
 def act(obs): return np.clip(2.2*(env.goal-obs[:2])-.8*obs[2:4],-env.task.max_accel,env.task.max_accel)
 return act
