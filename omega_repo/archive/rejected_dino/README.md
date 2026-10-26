# 判废：DINOv2 掩码

**不复制原始掩码进本仓**（避免被误用），只登记位置与判废依据。

| 目录 | 面积 | 与权威掩码 IoU |
|---|---|---|
| `data/hole_masks`（DINOv1，用户格子标注 + 精修） | 65.6% | 0.242 |
| `data/hole_masks_v2`（DINOv2） | 45.8% | 0.314 |

生成脚本：`scripts/refine_hole_dino.py` / `refine_hole_dino_v2.py`（已进 hole_exploration 归档族）
判废理由：把 2/3 画面标成"洞"，与权威掩码几乎无关。
