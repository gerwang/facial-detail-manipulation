from evaluation.swap_visualization_delta_age_only_evaluator import SwapVisualizationDeltaAgeOnlyEvaluator


class SwapVisualizationDeltaExpOnlyEvaluator(SwapVisualizationDeltaAgeOnlyEvaluator):
    def __init__(self, opt, target_phase):
        super().__init__(opt, target_phase)
        self.only_type = 'exp'
