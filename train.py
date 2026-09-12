from config import CFG
from core.engine import train_one_fold


def main() -> None:
    print(f"Experiment: {CFG.experiment_name}")
    print(f"Interaction: {CFG.interaction_type}")
    print(f"KAN basis: {CFG.fastkan_basis} (K={CFG.basis_count()})")
    print(
        f"Token dim: {CFG.token_dim} -> regression input {CFG.regression_input_dim()}"
    )
    print(f"KAN hidden dims: {CFG.validated_hidden_dims()}")
    print(f"Output directory: {CFG.experiment_dir()}")

    folds = (
        range(CFG.fold_index, CFG.fold_num) if CFG.run_all_folds else [CFG.fold_index]
    )
    for fold_index in folds:
        print(f"===== Training fold {fold_index} =====")
        result = train_one_fold(CFG, fold_index)
        print(f"Completed: {result}")


if __name__ == "__main__":
    main()
