import unittest


class TestImageDetectionMetrics(unittest.TestCase):
    def setUp(self):
        # Define some example true and predicted labels for testing
        self.true_labels = [
            (50, 50, 100, 100, 'class1'),
            (30, 30, 60, 60, 'class2')
        ]
        self.predicted_labels = [
            (50, 50, 100, 100, 'class1', 0.9),
            (25, 25, 50, 50, 'class2', 0.85),
            (10, 10, 20, 20, 'class3', 0.75)
        ]
        self.metrics = ImageDetectionMetrics(
            self.true_labels, self.predicted_labels, iou_threshold=0.5)

    def test_precision(self):
        # Expected precision calculation
        # 2 true positives / 3 predicted positives
        expected_precision = 0.6666666666666666
        self.assertAlmostEqual(self.metrics.precision(),
                               expected_precision, places=5)

    def test_recall(self):
        # Expected recall calculation
        expected_recall = 1.0  # 2 true positives / 2 actual positives
        self.assertAlmostEqual(self.metrics.recall(),
                               expected_recall, places=5)

    def test_f1_score(self):
        # Expected F1 score calculation
        expected_f1_score = 0.8  # Harmonic mean of precision and recall
        self.assertAlmostEqual(self.metrics.f1_score(),
                               expected_f1_score, places=5)

    def test_get_metrics(self):
        # Test getting multiple metrics at once
        expected_metrics = {
            'precision': 0.6666666666666666,
            'recall': 1.0,
            'f1_score': 0.8
        }
        metrics = self.metrics.get_metrics(['precision', 'recall', 'f1_score'])
        for key in expected_metrics:
            self.assertAlmostEqual(
                metrics[key], expected_metrics[key], places=5)


if __name__ == '__main__':
    unittest.main()
