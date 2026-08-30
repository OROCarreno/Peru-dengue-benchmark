from models import season_naive,RandomForest,XGBoost_tree,Linear_regression

def Evaluate():
    results = {
        "Season naive":      season_naive(),
        "Random Forest":     RandomForest(),
        "XGBoost tree":      XGBoost_tree(),
        "Linear Regression": Linear_regression(),
    }
    print("MODELS MAE RESULT")
    print("-----------------")
    for name,mae in results.items():
        print(f"{name}:{mae}")

    best_name  = min(results, key=results.get)
    worst_name = max(results, key=results.get)
    print("-----------------")
    print(f"Best:  {best_name} {results[best_name]}")
    print(f"Worst: {worst_name} {results[worst_name]}")


Evaluate()