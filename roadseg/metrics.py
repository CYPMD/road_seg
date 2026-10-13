import numpy as np


class BinaryMetrics:
    """Pixel-pooled road metrics plus image-mean IoU; each update is one image."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.image_ious = []

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
        if prediction.shape != target.shape or target.ndim != 2:
            raise ValueError("Metrics require matching 2D masks")
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        tn = int(np.count_nonzero(~prediction & ~target))
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.image_ious.append(tp / (tp + fp + fn) if tp + fp + fn else 1.0)

    def compute(self):
        if not self.image_ious:
            raise ValueError("Cannot compute metrics on an empty dataset")
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else float(fn == 0),
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn),
            "mean_image_iou": float(np.mean(self.image_ious)),
            "num_images": len(self.image_ious),
        }
