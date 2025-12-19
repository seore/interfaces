import numpy as np

def landmarks_to_feature(lm) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)

    # translate so wrist (0) is origin
    pts = pts - pts[0:1, :]

    # scale normalize by distance wrist -> middle MCP (9)
    scale = np.linalg.norm(pts[9, :2]) + 1e-6
    pts = pts / scale

    return pts.reshape(-1).astype(np.float32)
