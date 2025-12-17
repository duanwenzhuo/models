# main.py
import os
import config
from workflow import create_workflow_app


def main():
    print("==================================================")
    print("LLM-based Multi-Agent Omics Integration System")
    print("==================================================")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Model: {config.LLM_MODEL}")

    # 1. Prepare initial input
    data_path = os.path.join("data/small_atac_gene_activity/small_atac_gene_activity.h5ad")

    initial_state = {
        "user_intent": (
            "Load the data. I want to perform integration, "
            "and automatically detect the appropriate batch column from the metadata."
        ),
        "data_path": data_path,
        "plan": {},
        "data_hvg": {},
        "data_raw": {},
        "inspection": {},
        "results": {},
        "evaluation": None,
        "report_path": None,
        "benchmark_fraction": 0.2,
        "run_all_methods": True,
        "search_params": True,
        "top1_only": True,
        "data_hvg_full": {},
        "data_raw_full": {},
        "logs": [],
        "error": None,
    }

    # 2. Create and run workflow
    try:
        app = create_workflow_app()

        print("\n--- Starting Workflow Execution ---\n")
        final_state = app.invoke(initial_state)

        # 3. Handle results
        if final_state.get("error"):
            print(f"\nWorkflow terminated with error: {final_state['error']}")
            return

        print("\n================ FINAL REPORT ================")
        print(f"Execution Logs: {final_state.get('logs')}")

        evaluation_df = final_state.get("evaluation")
        if evaluation_df is not None:
            try:
                print("\nEvaluation (head):")
                print(evaluation_df.head())
            except Exception:
                print("\nEvaluation available.")

        results = final_state.get("results", {})
        if not results:
            print("No integration results found.")
            return

        for modality, modality_results in results.items():
            if not modality_results:
                print(f"No results for modality {modality}")
                continue
            for method, adata in modality_results.items():
                print(f"Success: [{modality}] - [{method}]")
                print(f"   Shape: {adata.shape}")
                embedding_keys = list(adata.obsm.keys())
                if embedding_keys:
                    print(f"   Embedding keys: {embedding_keys}")

                out_name = f"integrated_{modality}_{method}.h5ad"
                out_path = os.path.join(config.RESULTS_DIR, out_name)
                adata.write(out_path)
                print(f"   Saved to: {out_path}")

    except Exception as e:
        print(f"\nCritical System Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
