import argparse,json
from pathlib import Path
from .experiment import run_matrix
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",default="3,7,11,17,23");p.add_argument("--budget",type=int,default=200);p.add_argument("--eval",type=int,default=30);p.add_argument("--out",default="results/continuous_matrix.json");a=p.parse_args()
 r=run_matrix(tuple(int(x) for x in a.seeds.split(",")),a.budget,a.eval);f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(f)
if __name__=="__main__":main()
