import numpy as np
import json
import supervision as sv

KPTS_OKS_SIGMAS_COCO = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

MHR_TO_COCO = {
    0: 0,   # nose
    1: 1,   # left_eye
    2: 2,   # right_eye
    3: 3,   # left_ear
    4: 4,   # right_ear
    5: 5,   # left_shoulder
    6: 6,   # right_shoulder
    7: 7,   # left_elbow
    8: 8,   # right_elbow
    62: 9,  # left_wrist
    41: 10, # right_wrist
    9: 11,  # left_hip
    10: 12, # right_hip
    11: 13, # left_knee
    12: 14, # right_knee
    13: 15, # left_ankle
    14: 16, # right_ankle
}


class KeypointConverter:
    def __init__(self):
        pass

    def convert_to_coco17(self, pose_data_path: str):
        data = self.load_json(pose_data_path)[0]
        pred_keypoints_2d = data["pred_keypoints_2d"]
        scale = data["scale_x"]
        width = data["original_width"]
        height = data["original_height"]
        coco_xy = self.mhr_to_coco(pred_keypoints_2d)
        scale = 1 / scale
        coco_xy *= scale
        visibility = self.compute_visibility(coco_xy, width, height)
        # supervision expects batch dimension
        xy = coco_xy[None, ...]                    # (1, 17, 2)
        confidence = visibility[None, ...].astype(np.float32)  # (1, 17)
        return sv.KeyPoints(xy=xy, confidence=confidence)


    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def mhr_to_coco(mhr_kpts):
        coco_kpts = np.zeros((17, 2), dtype=np.float32)
        for mhr_idx, coco_idx in MHR_TO_COCO.items():
            coco_kpts[coco_idx] = mhr_kpts[mhr_idx]
        return coco_kpts

    @staticmethod
    def compute_visibility(xy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        visible = np.all(xy > 0, axis=1)

        if img_w is not None and img_h is not None:
            visible &= (xy[:, 0] < img_w) & (xy[:, 1] < img_h)
        return visible.astype(np.uint8)

    @staticmethod
    def OKS(kp1: sv.KeyPoints, kp2: sv.KeyPoints) -> float:
        k1 = kp1.xy[0]  # (17, 2)
        k2 = kp2.xy[0]
        v1 = kp1.confidence[0]
        v2 = kp2.confidence[0]

        visible = (v1 > 0) & (v2 > 0)
        if visible.sum() == 0:
            return 0.0

        k1 = k1[visible]
        k2 = k2[visible]
        sigmas = KPTS_OKS_SIGMAS_COCO[visible]

        # --- object scale (bbox diagonal) ---
        min_xy = np.minimum(k1.min(axis=0), k2.min(axis=0))
        max_xy = np.maximum(k1.max(axis=0), k2.max(axis=0))
        s = np.linalg.norm(max_xy - min_xy) + 1e-6

        d2 = np.sum((k1 - k2) ** 2, axis=1)
        vars = (2 * sigmas * s) ** 2

        oks = np.mean(np.exp(-d2 / vars))
        return float(oks)
