import numpy as np
import keras.backend as K
from keras.models import load_model


class YOLO:
    def __init__(self, class_threshold, nms_threshold, algo = 'tiny_yolo.h5'):
        """Init.

        # Arguments
            class_threshold: Integer, threshold for object.
            nms_threshold: Integer, threshold for box.
        """
        self._t1 = class_threshold
        self._t2 = nms_threshold
        #K.clear_session()
        self._yolo = load_model(algo)
    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def _process_feats(self, out, anchors, mask):
        """process output features.

        # Arguments
            out: Tensor (N, N, 3, 4 + 1 +80), output feature map of yolo.
            anchors: List, anchors for box.
            mask: List, mask for anchors.

        # Returns
            boxes: ndarray (N, N, 3, 4), x,y,w,h for per box.
            box_confidence: ndarray (N, N, 3, 1), confidence for per box.
            box_class_probs: ndarray (N, N, 3, 80), class probs for per box.
        """
        grid_h, grid_w, num_boxes = map(int, out.shape[1: 4]) # out shape (height width, column width, number of anchors)

        anchors = [anchors[i] for i in mask]
        # Reshape to batch, height, width, num_anchors, box_params.
        out = out[0]
        box_xy = self._sigmoid(out[..., :2])  # first two of 85 for all the N, N, 3 (N, N, 3, 2)
        box_wh = np.exp(out[..., 2:4]) # second two of 85 for all the N, N, 3 (N, N, 3, 2)
        box_wh = box_wh * anchors
        box_confidence = self._sigmoid(out[..., 4]) # p 5th of 85 for all the N, N, 3 (N, N, 3, 1)
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = self._sigmoid(out[..., 5:]) # 6th onwards of 85 for all the N, N, 3 (N, N, 3, 80)

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w) # (grid_w, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h) # (grid_h, grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2) # (grid_w, grid_w, 3, 1)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2) # (grid_h, grid_h, 3, 1)
        grid = np.concatenate((col, row), axis=-1) # (grid_w, grid_h, 3, 2)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= (416, 416)
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.

        # Arguments
            boxes: ndarray, boxes of objects. (N, N, 3, 4)
            box_confidences: ndarray, confidences of objects. (N, N, 3, 1)
            box_class_probs: ndarray, class_probs of objects. (N, N, 3, 80)

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        box_scores = box_confidences * box_class_probs # p * class_scores (N, N, 3, 80)
        box_classes = np.argmax(box_scores, axis=-1)   # index of max box scores (tracking maximum index) (N, N, 3, 1)
        box_class_scores = np.max(box_scores, axis=-1) # max box score (N, N, 3, 1)
        pos = np.where(box_class_scores >= self._t1)   # all boxes > threshold (N, N, pos)

        boxes = boxes[pos] # (pos, 4)
        classes = box_classes[pos] # (pos, 80)
        scores = box_class_scores[pos] # (pos, 1)

        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes : ndarray, boxes of objects. (:, 4)
            scores: ndarray, scores of objects. (:, 1)

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h # array of area
        order = scores.argsort()[::-1] # index, decending order of scores

        keep = []
        while order.size > 0:
            i = order[0] # index of highest score currently in order
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]]) # array of element-wise comparision (x)
            yy1 = np.maximum(y[i], y[order[1:]]) # array of element-wise comparision (y)
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]]) # similar to get width
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]]) # similar to get height

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1 # area of all the other boxes
            uni = areas[i] + areas[order[1:]] - inter # aUb(x) = a(x) + b(x) - aNb(x)

            iou = inter / uni
            inds = np.where(iou <= self._t2)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep

    def _yolo_out(self, outs, shape):
        """Process output of yolo base net.

        # Argument:
            outs: output of yolo base net.
            shape: shape of original image.

        # Returns:
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []

        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        # Scale boxes back to original image shape.
        width, height = shape[1], shape[0]
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims # since box width, height is scaled on 1/image_dim

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def predict(self, image, shape):
        """Detect the objects with yolo.

        # Arguments
            image: ndarray, processed input image.
            shape: shape of original image.

        # Returns
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
        """

        outs = self._yolo.predict(image)
        boxes, classes, scores = self._yolo_out(outs, shape)

        return boxes, classes, scores
