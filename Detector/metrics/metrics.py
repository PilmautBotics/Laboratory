class ImageDetectionMetrics:
    def __init__(self, true_labels, predicted_labels, iou_threshold=0.5):
        """
        Initialize the class with true labels, predicted labels, and an IoU threshold.

        :param true_labels: List of true labels (each label is a tuple (x, y, width, height, class))
        :param predicted_labels: List of predicted labels (each label is a tuple (x, y, width, height, class, score))
        :param iou_threshold: IoU threshold to consider a detection as correct
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.iou_threshold = iou_threshold
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.calculate_metrics()

    def iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        :param box1: Tuple (x, y, width, height)
        :param box2: Tuple (x, y, width, height)
        :return: IoU of the two boxes
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the coordinates of the corners of the boxes
        x1_min, x1_max = x1, x1 + w1
        y1_min, y1_max = y1, y1 + h1
        x2_min, x2_max = x2, x2 + w2
        y2_min, y2_max = y2, y2 + h2

        # Calculate the coordinates of the intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calculate the area of the intersection
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Calculate the area of the boxes
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Calculate the IoU
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou

    def calculate_metrics(self):
        """
        Calculate the image detection metrics.
        """
        detected = set()

        for pred in self.predicted_labels:
            best_iou = 0
            best_true = None
            for idx, true in enumerate(self.true_labels):
                if idx not in detected:
                    iou_score = self.iou((true[0], true[1], true[2], true[3]),
                                         (pred[0], pred[1], pred[2], pred[3]))
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_true = idx

            if best_iou > self.iou_threshold and best_true is not None:
                self.true_positives += 1
                detected.add(best_true)
            else:
                self.false_positives += 1

        self.false_negatives = len(self.true_labels) - len(detected)

    def precision(self):
        """
        Calculate precision.

        :return: Precision
        """
        if self.true_positives + self.false_positives > 0:
            return self.true_positives / (self.true_positives + self.false_positives)
        return 0.0

    def recall(self):
        """
        Calculate recall.

        :return: Recall
        """
        if self.true_positives + self.false_negatives > 0:
            return self.true_positives / (self.true_positives + self.false_negatives)
        return 0.0

    def f1_score(self):
        """
        Calculate the F1 score.

        :return: F1 Score
        """
        precision = self.precision()
        recall = self.recall()
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def get_metric(self, metric_name):
        """
        Return the value of the requested metric.

        :param metric_name: Name of the metric ('precision', 'recall', 'f1_score')
        :return: Value of the metric
        """
        if metric_name == 'precision':
            return self.precision()
        elif metric_name == 'recall':
            return self.recall()
        elif metric_name == 'f1_score':
            return self.f1_score()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def get_metrics(self, metric_names):
        """
        Return a dictionary with the values of the requested metrics.

        :param metric_names: List of metric names ('precision', 'recall', 'f1_score')
        :return: Dictionary with the values of the metrics
        """
        results = {}
        for name in metric_names:
            results[name] = self.get_metric(name)
        return results

# Example usage
# metrics = ImageDetectionMetrics(true_labels, predicted_labels)
# print(metrics.get_metrics(['precision', 'recall', 'f1_score']))
