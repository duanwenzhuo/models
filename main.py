# main.py
import argparse
import contextlib
import os

import config
from tools import slugify_dataset_name


def main():
    parser = argparse.ArgumentParser(description="LLM-based Multi-Agent Omics Integration")
    parser.add_argument("--dataset", default="", help="Dataset folder name under DATA_DIR (e.g. Lung_atlas_public)")
    parser.add_argument("--data-path", default="", help="Explicit path to .h5ad (overrides --dataset)")
    parser.add_argument(
        "--task",
        default="Load the data and perform integration while automatically detecting the batch column.",
    )
    parser.add_argument("--subset-genes", help="Comma separated gene list", default="")
    parser.add_argument("--subset-celltypes", help="Comma separated cell type list", default="")
    parser.add_argument("--subset-batches", help="Comma separated batch ids", default="")
    parser.add_argument("--benchmark-fraction", type=float, default=0.2)
    parser.add_argument("--run-all-methods", action="store_true", help="Force running all integration methods")
    parser.add_argument("--no-search-params", action="store_true", help="Disable parameter search")
    parser.add_argument("--allow-multi-top", action="store_true", help="Keep all tuned methods instead of top1")
    args = parser.parse_args()

    data_path = args.data_path.strip().strip("'")
    dataset = args.dataset.strip()

    if not data_path:
        if not dataset:
            raise ValueError("You must provide either --data-path or --dataset")
        data_path = os.path.join(config.DATA_DIR, dataset, f"{dataset}.h5ad")

    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    subset_config = {
        "genes": [g.strip().upper() for g in args.subset_genes.split(",") if g.strip()],
        "celltypes": [c.strip() for c in args.subset_celltypes.split(",") if c.strip()],
        "batches": [b.strip() for b in args.subset_batches.split(",") if b.strip()],
    }

    dataset_slug = slugify_dataset_name(data_path)
    log_path = config.configure_logging(dataset_slug)

    # Late import to avoid any module-level logging config from polluting console output.
    from workflow import create_workflow_app

    initial_state = {
        "user_intent": args.task,
        "data_path": data_path,
        "plan": {},
        "data_hvg": {},
        "data_raw": {},
        "inspection": {},
        "results": {},
        "evaluation": None,
        "report_path": None,
        "benchmark_fraction": args.benchmark_fraction,
        "run_all_methods": bool(args.run_all_methods),
        "search_params": not args.no_search_params,
        "top1_only": not args.allow_multi_top,
        "data_hvg_full": {},
        "data_raw_full": {},
        "views": {},
        "view_meta": {},
        "view_build_log": [],
        "logs": [],
        "error": None,
        "subset_config": subset_config,
        "task_modalities": [],
        "preprocess_recommendations": {},
        "method_run_log": [],
    }

    stdout_path = os.path.splitext(log_path)[0] + ".stdout.log"

    try:
        app = create_workflow_app()

        # Hard guarantee: absolutely no library print spam on console.
        with open(stdout_path, "a", encoding="utf-8") as _out, \
                contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_out):
            final_state = app.invoke(initial_state)

        if final_state.get("error"):
            print(f"✗ Workflow error: {final_state['error']}")
            print(f"Log: {log_path}")
            print(f"Stdout/Stderr: {stdout_path}")
            return

        results = final_state.get("results") or {}
        if not results:
            print("✗ No integration results produced.")
            print(f"Log: {log_path}")
            print(f"Stdout/Stderr: {stdout_path}")
            return

        saved_paths = []
        for modality, modality_results in results.items():
            for method, adata in (modality_results or {}).items():
                out_name = f"integrated_{modality}_{method}.h5ad"
                out_path = os.path.join(config.RESULTS_DIR, out_name)
                adata.write(out_path)
                saved_paths.append(out_path)

        final_sel = final_state.get("final_selection") or {}
        if final_sel:
            for modality, info in final_sel.items():
                score = info.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                print(f"✓ {modality}: best={info.get('method')} score={score_str}")
        else:
            print("✓ Integration finished.")

        if saved_paths:
            print(f"Outputs ({len(saved_paths)}):")
            for p in saved_paths:
                print(f"  - {p}")

        print(f"Log: {log_path}")
        print(f"Stdout/Stderr: {stdout_path}")

    except Exception as e:
        print(f"Critical System Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"Log: {log_path}")
        print(f"Stdout/Stderr: {stdout_path}")


if __name__ == "__main__":
    main()
