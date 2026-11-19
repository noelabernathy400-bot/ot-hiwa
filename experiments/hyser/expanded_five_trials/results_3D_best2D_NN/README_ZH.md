# 3D Hard/Soft nearest-neighbor and best training-only 2D projections
3D features X/Y fitted only on four training trials. These are new three-dimensional reductions of full five-finger target, NOT old hand-picked thumb/middle/index three-dimensional datasets.
Threshold calibrated from training target domain unrelated trial pair distances (5% quantile); score both mere NN distance coverage and true synchronized physical-neighborhood retrieval.
Cannot compare absolute tolerance or W2 across 3D versus 5D representation.
|Config|Method|train radius|cloud coverage|correct physical-neighborhood retrieval|random null|exact time index|
|---|---|---:|---:|---:|---:|---:|
|PCA3_PCA3|hard|0.3193|63.120%|5.680%|8.992%|0.600%|
|PCA3_PCA3|soft05|0.3193|63.280%|5.800%|8.992%|0.640%|
|FA3_PCA3|hard|0.3193|63.840%|4.960%|8.992%|0.560%|
|FA3_PCA3|soft05|0.3193|63.840%|4.880%|8.992%|0.560%|
|PCA3_FA3|hard|0.2983|59.040%|4.280%|9.155%|0.440%|
|PCA3_FA3|soft05|0.2983|59.680%|4.360%|9.155%|0.480%|
|FA3_FA3|hard|0.2983|62.120%|6.280%|9.155%|0.520%|
|FA3_FA3|soft05|0.2983|62.680%|6.680%|9.155%|0.400%|
