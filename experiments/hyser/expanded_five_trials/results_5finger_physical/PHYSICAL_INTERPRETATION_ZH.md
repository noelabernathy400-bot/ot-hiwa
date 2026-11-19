# Hyser 真正五根手指：四簇与五簇物理意义诊断

Random Task没有五个单指动作真标签。以下使用原始五指力量测量对无监督统计簇作训练后描述。不把源域簇编号直接对到目标域簇编号。

- EMG_source_PCA5 K=4: silhouette=0.3415; size=[233, 666, 179, 172]; trial-NMI=0.0535
- EMG_source_PCA5 K=5: silhouette=0.3783; size=[131, 627, 152, 165, 175]; trial-NMI=0.0551
- five_finger_force K=4: silhouette=0.3161; size=[656, 161, 332, 101]; trial-NMI=0.0767
  - 最大稳健化力量偏差手指：['little', 'middle', 'thumb', 'ring']，其中不同手指总数=4
- five_finger_force K=5: silhouette=0.3506; size=[228, 602, 168, 101, 151]; trial-NMI=0.0724
  - 最大稳健化力量偏差手指：['thumb', 'little', 'little', 'ring', 'middle']，其中不同手指总数=4

**判定**：同一个手指可能参与多个簇，多个手指也可共同决定一个簇；这里的最大力量偏差是描述性标签，不能当作经实验标注的真实动作。
