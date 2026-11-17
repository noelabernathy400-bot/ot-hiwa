"""Historical extended mu=2 reference-style run (not original HiWA source)."""
import json,numpy as np
from hard_hiwa_adapted import hiwa_fit,align_eval,OUT
z=np.load(OUT/"domains.npz",allow_pickle=True);q=np.load(OUT/"clusters.npz")
X,Y,cx,cy=z["Xtrain"],z["Ytrain"],q["cx"],q["cy"]
R,P,h=hiwa_fit(X,Y,cx,cy,mu=2.,iterations=110,sa_iterations=10)
train=align_eval(X,Y,cx,cy,z["tr_pair"],z["tr_y_pair"],R,P)
test=align_eval(z["Xtest"],z["Ytest"],q["cx_test"],q["cy_test"],z["te_pair"],z["te_y_pair"],R,P)
result=dict(mu=2.,iter=len(h),converged=h[-1]["primal_residual"]<.05 and h[-1]["rotation_change"]<.001,
 last_primal=h[-1]["primal_residual"],last_rotation_change=h[-1]["rotation_change"],
 train=train,test=test,R=R.tolist(),P=P.tolist(),history=h)
(OUT/"hiwa_mu2_extended.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ("iter","converged","last_primal","last_rotation_change","train","test")},indent=2))
