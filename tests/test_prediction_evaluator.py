import unittest

from prediction_evaluator import PredictionEvaluator


class PredictionEvaluatorTests(unittest.TestCase):
    def test_interpolates_truth_at_exact_target_time(self):
        evaluator = PredictionEvaluator(max_interpolation_gap=0.2)
        evaluator.observe(0.0, 0.0, 0.0, valid=True)
        evaluator.add_prediction(0.0, 0.08, 8.0, 0.0)

        evaluations = evaluator.observe(0.1, 10.0, 0.0, valid=True)

        self.assertEqual(len(evaluations), 1)
        evaluation = evaluations[0]
        self.assertAlmostEqual(evaluation.x_true, 8.0)
        self.assertAlmostEqual(evaluation.error, 0.0)
        self.assertTrue(evaluation.valid)

    def test_two_predictions_are_not_compared_to_one_current_position(self):
        evaluator = PredictionEvaluator(max_interpolation_gap=0.2)
        evaluator.observe(0.0, 0.0, 0.0, valid=True)
        evaluator.add_prediction(0.0, 0.02, 2.0, 0.0)
        evaluator.add_prediction(0.0, 0.08, 8.0, 0.0)

        evaluations = evaluator.observe(0.1, 10.0, 0.0, valid=True)

        self.assertEqual([round(item.x_true, 4) for item in evaluations], [2.0, 8.0])
        self.assertEqual([round(item.error, 4) for item in evaluations], [0.0, 0.0])

    def test_long_detection_gap_skips_prediction(self):
        evaluator = PredictionEvaluator(max_interpolation_gap=0.1)
        evaluator.observe(0.0, 0.0, 0.0, valid=True)
        evaluator.add_prediction(0.0, 0.08, 8.0, 0.0)

        self.assertEqual(evaluator.observe(0.3, 30.0, 0.0, valid=True), [])
        self.assertEqual(evaluator.skipped_predictions, 1)

    def test_reset_marks_pending_predictions_as_skipped(self):
        evaluator = PredictionEvaluator()
        evaluator.add_prediction(0.0, 0.08, 0.0, 0.0)
        evaluator.reset()

        self.assertEqual(evaluator.pending_count, 0)
        self.assertEqual(evaluator.skipped_predictions, 1)


if __name__ == "__main__":
    unittest.main()
