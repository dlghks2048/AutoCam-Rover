def box_center(box):
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = box[:4]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def box_area(box):
    if len(box) < 4:
        return 0
    x1, y1, x2, y2 = box[:4]
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a, box_b):
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def clamp(value, low, high):
    return max(low, min(high, value))
