import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import logger

def compile_final_experiment_report():
    """
    Compiles and prints the master experiment comparison table across all V1, V2,
    Phase 5 Champion, and Phase 6 Expanded model runs on real audio.
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    bench_file = os.path.join(output_dir, "benchmark_results.json")
    exp_file = os.path.join(output_dir, "expanded_results.json")

    # Hardcoded/Verified V1 & V2 Baseline Metrics on Real RAVDESS (Actors 19-24)
    table_data = [
        {
            "Configuration": "V1 Baseline",
            "Dataset": "Real RAVDESS",
            "Train Size": "1080 clips",
            "Features": 162,
            "Model Architecture": "SVM RBF (C=1.0)",
            "Macro F1": "53.90%",
            "Accuracy": "53.89%"
        },
        {
            "Configuration": "V2 Representation",
            "Dataset": "Real RAVDESS",
            "Train Size": "1080 clips",
            "Features": 342,
            "Model Architecture": "SVM RBF (C=1.0)",
            "Macro F1": "52.70%",
            "Accuracy": "54.44%"
        }
    ]

    # Load Phase 5 Champion Model Results if available
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            bdata = json.load(f)
        champ_arch = bdata.get("champion_architecture", "Best SVM")
        tm = bdata.get("final_test_metrics", {})
        table_data.append({
            "Configuration": "Champion Model",
            "Dataset": "Real RAVDESS",
            "Train Size": "1080 clips",
            "Features": 342,
            "Model Architecture": champ_arch,
            "Macro F1": f"{tm.get('macro_f1', 0)*100:.2f}%",
            "Accuracy": f"{tm.get('accuracy', 0)*100:.2f}%"
        })

    # Load Phase 6 Expanded Model Results if available
    if os.path.exists(exp_file):
        with open(exp_file, "r") as f:
            edata = json.load(f)
        em = edata.get("metrics", {})
        tot_size = edata.get("train_size_total", 2280)
        table_data.append({
            "Configuration": "Best Expanded Model",
            "Dataset": "Real RAVDESS + CREMA-D",
            "Train Size": f"{tot_size} clips",
            "Features": 342,
            "Model Architecture": "SVM RBF (C=10.0, g=0.001)",
            "Macro F1": f"{em.get('macro_f1', 0)*100:.2f}%",
            "Accuracy": f"{em.get('accuracy', 0)*100:.2f}%"
        })

    df = pd.DataFrame(table_data)

    print("\n" + "="*85)
    print("                     VIORA V2 EXPERIMENTAL BENCHMARK SUMMARY TABLE                    ")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85 + "\n")

    summary_file = os.path.join(output_dir, "final_experiment_summary.csv")
    df.to_csv(summary_file, index=False)
    logger.info(f"Saved master benchmark table to {summary_file}")
    return df

if __name__ == "__main__":
    compile_final_experiment_report()
