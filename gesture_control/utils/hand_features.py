import cv2
import numpy as np

def segment_hand_ycrcb(frame_bgr):
    """
    Basic skin segmentation in YCrCb.
    Returns: mask (uint8), debug dict
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    # common skin thresholds (tweakable)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask

def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def contour_features(cnt):
    """
    Feature vector from hand contour:
    - area, perimeter
    - bounding box ratio
    - extent, solidity
    - Hu moments (7)
    - convexity defects count / depth summary
    Total: ~20-ish features
    """
    area = cv2.contourArea(cnt)
    if area < 1000:
        return None

    peri = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = (w / (h + 1e-6))

    rect_area = w * h + 1e-6
    extent = area / rect_area

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    # Hu moments (log scale)
    moms = cv2.moments(cnt)
    hu = cv2.HuMoments(moms).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    # convexity defects (finger gaps proxy)
    defects_count = 0
    defects_depth_mean = 0.0
    defects_depth_max = 0.0

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3 and len(cnt) > 3:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None and len(defects) > 0:
            depths = defects[:, 0, 3].astype(np.float32) / 256.0
            # keep only meaningful defects
            depths = depths[depths > 5.0]
            defects_count = int(len(depths))
            if defects_count > 0:
                defects_depth_mean = float(np.mean(depths))
                defects_depth_max = float(np.max(depths))

    # centroid (for swipe tracking)
    cx = float(moms["m10"] / (moms["m00"] + 1e-6))
    cy = float(moms["m01"] / (moms["m00"] + 1e-6))

    feat = np.concatenate([
        np.array([
            area, peri, aspect, extent, solidity,
            defects_count, defects_depth_mean, defects_depth_max,
            cx, cy
        ], dtype=np.float32),
        hu.astype(np.float32)
    ])
    return feat

def draw_debug(frame, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
