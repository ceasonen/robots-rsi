import unittest
import numpy as np
from teachlab.env import PointMassEnv,TASKS
from teachlab.teachers import parent
from teachlab.experiment import run_one
class TestContinuous(unittest.TestCase):
 def test_dynamics(self):
  e=PointMassEnv(TASKS["impulsive"]);o=e.reset(5);self.assertEqual(o.shape,(7,));o,r,d,i=e.step([.2,-.1]);self.assertEqual(o.shape,(7,))
 def test_split_is_real_and_replayable(self):
  e=PointMassEnv(TASKS["impulsive"]);e.reset(123,split="target")
  target_shock=e.shock.copy();start=e.p.copy()
  e.reset(123,split="source")
  self.assertTrue(np.array_equal(e.shock,np.zeros(2)))
  self.assertTrue(np.array_equal(e.p,start))
  self.assertFalse(np.array_equal(target_shock,e.shock))
 def test_parent_uses_its_observed_phase(self):
  e=PointMassEnv(TASKS["precision"]);p=parent(e)
  o=np.array([0.,0.,0.,0.,0.,0.,1.]);later=o.copy();later[4]=1.
  self.assertNotEqual(p(o).tolist(),p(later).tolist())
 def test_exact_budget_and_accounting(self):
  r=run_one("precision",13,budget=100,n_eval=10)
  for v in r["results"].values():
   self.assertEqual(v["training_transitions"],100)
   self.assertEqual(v["source_success"]-r["parent_source"],v["source_repair"]-v["source_harm"])
   self.assertEqual(v["target_success"]-r["parent_target"],v["target_repair"]-v["target_harm"])
  self.assertEqual(r["receipts"]["fixed_aggressive"]["steps"],100)
  self.assertEqual(r["receipts"]["fixed_damped"]["steps"],100)
if __name__=="__main__":unittest.main()
