import numpy as np
import torch

from .grader import Case, Grader

CKPT_TEMPLATE = "*_{}.pth"

MAX_NUM_PARAMS = 300_000_000  # 300M

def model_size_check(model):
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > MAX_NUM_PARAMS:
        raise ValueError(
            f"Model has {num_params} parameters, which is greater than the maximum allowed {MAX_NUM_PARAMS}"
        )

class VLMGrader(Grader):
    """VLM Model Grader"""

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    VALIDATION_ACC_BOUND = 0.0, 0.7
    EXTRA_CREDIT_ACC_BOUND = 0.8, 0.85

    EXTRA_CREDIT_RATIO = 5.0 / 100.0
    model_name = "vlm"

    def load_model(self) -> torch.nn.Module:
        llm = getattr(self.module, f"load_{self.model_name}")()
        llm.model.eval()
        model_size_check(llm.model)
        return llm

    def normalize_score(self, score, min_score, max_score):
        """
        Returns a score based on model's score normalized to [0, 1]

        If the score is greater than or equal to max_score, you get 1.0 (full score)
        If the score is less than or equal to min_score, you get 0.0 (no points)
        If the score is greater than or equal to extra_score lower bound, you get extra credit
        Score is linearly interpolated between these extremes
        """
        score_normalized = (score - min_score) / (max_score - min_score)
        normal_score = np.clip(score_normalized, 0.0, 1.0)

        extra_bounds = self.EXTRA_CREDIT_ACC_BOUND
        extra_score_normalized = (score - extra_bounds[0]) / (extra_bounds[1] - extra_bounds[0])
        extra_score = np.clip(extra_score_normalized, 0.0, 1.0) * self.EXTRA_CREDIT_RATIO

        return normal_score + extra_score

    @Case(score=100, timeout=60000)
    def test_validation_loss(self):
        """Test the answer accuracy"""
        dataset = self.module.data.VQADataset("valid_grader")
        model = self.load_model()
        benchmark_result = self.module.data.benchmark(model, dataset, 128)
        print("accuracy is:", benchmark_result.accuracy)

        return self.normalize_score(benchmark_result.accuracy, *self.VALIDATION_ACC_BOUND)
