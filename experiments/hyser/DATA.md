# 数据来源与本地目录

- 官方：<https://physionet.org/content/hd-semg/2.0.0/>
- 子目录：`random_dataset/subject01_session1/`
- 本次使用 `random_preprocess_sample1.dat` + `.hea`、`random_force_sample1.dat` + `.hea`。
- 官方另外有 `random_raw_sample1.dat/.hea`，本次没有使用。
- 肌电：256 通道×51,200 帧，2048 Hz；力量：5 通道×2,500 帧，100 Hz；同一25秒实验。两个采样率不同，切窗使用物理时间。
- `data/` 与 `output/` 由 `.gitignore` 排除。请勿将原始 25MB 级 `.dat` 直接写入 Git 历史。
- 若系统下载成 `random_force_sample1(1).dat`，务必重命名 `random_force_sample1.dat`，否则 WFDB 头文件指向的文件名不匹配。
- 单个 sample1 不能用于独立试次测试；下载同一受试者同一 session 的 sample2–5 才能进行完整试次留出。
- 所有同步标签和试次信息只进入评价与分割，不作为模型的监督匹配信号。
