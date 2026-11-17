"""Historical parameter scan (requires prepared output/domains.npz and clusters.npz)."""
import json, numpy as np
from pathlib import Path
from hard_hiwa_adapted import hiwa_fit,align_eval,OUT
z=np.load(OUT/"domains.npz",allow_pickle=True);q=np.load(OUT/"clusters.npz")
X,Y,cx,cy=z["Xtrain"],z["Ytrain"],q["cx"],q["cy"]
records=[]
for mu in [.05,.5,2.]:
    R,P,h=hiwa_fit(X,Y,cx,cy,mu=mu,iterations=22,sa_iterations=10)
    tr=align_eval(X,Y,cx,cy,z["tr_pair"],z["tr_y_pair"],R,P)
    te=align_eval(z["Xtest"],z["Ytest"],q["cx_test"],q["cy_test"],z["te_pair"],z["te_y_pair"],R,P)
    records.append(dict(mu=mu,primal_last=h[-1]["primal_residual"],deltaR_last=h[-1]["rotation_change"],
       train=tr,test=te,P=P.tolist(),R=R.tolist(),history=h))
(OUT/"hiwa_mu_scan.json").write_text(json.dumps(records,indent=2))
