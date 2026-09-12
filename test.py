from config import CFG
from core.engine import evaluate_one_fold, seed_everything, summarize_folds


def main() -> None:
    seed_everything(CFG.random_seed)
    print(f"Experiment directory: {CFG.experiment_dir()}")

    folds = range(CFG.fold_num) if CFG.run_all_folds else [CFG.fold_index]
    results = []
    for fold_index in folds:
        result = evaluate_one_fold(CFG, fold_index)
        results.append(result)
        print(f"fold {fold_index}: {result['metrics']}")
    if len(results) > 1:
        summary = summarize_folds(CFG, results)
        print(f"Cross-validation averages: {summary['average']}")


if __name__ == "__main__":
    main()
