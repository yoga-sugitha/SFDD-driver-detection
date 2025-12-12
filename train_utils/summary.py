#train_utils/summary.py
from omegaconf import DictConfig

def print_summary(
    experiment_name: str,
    cfg: DictConfig,
    test_result: dict,
    complexity_metrics: dict,
    latency_metrics: dict
):
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Experiment:        {experiment_name}")
    print(f"Model:             {cfg.model.name}")
    print(f"Dataset:           {cfg.data.name}")
    print(f"Optimizer:         {cfg.optimizer.name}")
    print(f"Seed:              {cfg.experiment.seed}")
    print(f"{'='*70}")
    print(f"Test Accuracy:     {test_result.get('test_acc', 0.0):.4f}")
    print(f"Test Precision:    {test_result.get('test_precision', 0.0):.4f}")
    print(f"Test Recall:       {test_result.get('test_recall', 0.0):.4f}")
    print(f"Test F1 (Macro):   {test_result.get('test_f1_macro', 0.0):.4f}")
    print(f"Test F1 (Weighted):{test_result.get('test_f1_weighted', 0.0):.4f}")
    print(f"{'='*70}")
    print(f"Model Parameters:  {complexity_metrics['num_params']:,}")
    if complexity_metrics['flops'] > 0:
        print(f"FLOPs:             {complexity_metrics['flops']:,}")
    print(f"Avg Latency:       {latency_metrics['avg_inference_latency_ms']:.2f} ms")
    print(f"{'='*70}\n")
