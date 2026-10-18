from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def plot_raw_data(df, output_dir):
    """Plot the original dataset before cleaning and preprocessing."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #1. Target distribution
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x="charges", bins=30, kde=True)
    plt.title("Distribution of Insurance Charges")
    plt.xlabel("Insurance Charges")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_charges_distribution.png", dpi=150)
    plt.show()
    plt.close()

    #2. Charges versus age
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker")
    plt.title("Insurance charges vs Age")
    plt.xlabel("Age")
    plt.ylabel("Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_charges_vs_age.png", dpi=150)
    plt.show()
    plt.close()

    #3. Charges versus BMI
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker")
    plt.title("Insurance charges vs BMI")
    plt.xlabel("BMI")
    plt.ylabel("Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_charges_vs_bmi.png", dpi=150)
    plt.show()
    plt.close()

    #4. Charges by smoker status
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="smoker", y="charges")
    plt.title("Insurance Charges by Smoker Status")
    plt.xlabel("Smoker")
    plt.ylabel("Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_charges_by_smoker.png", dpi=150)
    plt.show()
    plt.close()

    #5. Charges by region
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="region", y="charges")
    plt.title("Insurance Charges by Region")
    plt.xlabel("Region")
    plt.ylabel("Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_charges_by_region.png", dpi=150)
    plt.show()
    plt.close()


def plot_preprocessed_data(X_train, y_train, output_dir):
    """Visualize scaled training features and the unscaled target."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #1. Distribution of standardized continuous features
    continuous_features = ["age", "bmi", "children"]

    for feature in continuous_features:
        plt.figure(figsize=(7, 4))
        sns.histplot(X_train[feature], bins=25, kde=True)
        plt.title(f"Standardized Distribution: {feature}")
        plt.xlabel(f"{feature} (standardized)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"scaled_{feature}_distribution.png", dpi=150)
        plt.show()
        plt.close()

    #2. Target against Standardized age
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_train["age"], y=y_train["charges"])
    plt.title("Training Charges vs Standardized Age")
    plt.xlabel("Standardized Age")
    plt.ylabel("Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "processed_charges_vs_age.png", dpi=150)
    plt.show()
    plt.close()

    #3. Target against Standardized BMI
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_train["bmi"], y=y_train["charges"])
    plt.title("Training Charges vs Standardized BMI")
    plt.xlabel("Standardized BMI")
    plt.ylabel("Insurance Charges")
    plt.tight_layout()
    plt.savefig(output_dir / "processed_charges_vs_bmi.png", dpi=150)
    plt.show()
    plt.close()

def plot_learning_curve(J, i, alpha, output_dir):
    """Visualize the learning curve"""

    output_dir = Path(output_dir).parent / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(i, J)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Training Cost J(W,b)")
    plt.title(f"Gradient descent Learning curve for alpha = {alpha}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"Learning Curve for alpha = {alpha}.png", dpi=150)
    plt.show()
    plt.close()

def plot_selected_candidates_comparison(final_results, output_dir):
    """Compare the Learning curves of the final selected candidates to choose the best alpha"""

    output_dir = Path(output_dir).parent / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for alpha, result in final_results.items():
        plt.plot(
            result["iterations"],
            result["J_history"],
            label=f"alpha = {alpha}"
        )

    plt.xlabel("Iterations")
    plt.ylabel("Training Cost")
    plt.title("Final Learning rate comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "final_learning_rate_comparison.png", dpi=150)
    plt.show()
    plt.close()


def plot_predicted_and_actual_curve(Y_train, Y_train_pred, Y_val, Y_val_pred, output_dir):
    """Plot the actual and predicted Insurance Charges"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_train_actual = Y_train.flatten()
    y_train_predicted = Y_train_pred.flatten()

    y_val_actual = Y_val.flatten()
    y_val_predicted = Y_val_pred.flatten()

    # Actual vs Predicted: Training and Validation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_data = [
        (axes[0], y_train_actual, y_train_predicted, "Training Set"),
        (axes[1], y_val_actual, y_val_predicted, "Validation Set"),
    ]

    for ax, actual, predicted, title in plot_data:
        ax.scatter(actual, predicted, alpha=0.6)

        # Reference line for perfect predictions
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--")

        ax.set_xlabel("Actual Charges ($)")
        ax.set_ylabel("Predicted Charges ($)")
        ax.set_title(f"Actual vs Predicted — {title}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_insurance_charges.png", dpi = 150)
    plt.show()
    plt.close()

    # Calculate residuals
    train_residuals = y_train_actual - y_train_predicted
    val_residuals = y_val_actual - y_val_predicted

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    residual_data = [
        (axes[0], y_train_predicted, train_residuals, "Training Set"),
        (axes[1], y_val_predicted, val_residuals, "Validation Set"),
    ]

    for ax, predicted, residuals, title in residual_data:
        ax.scatter(predicted, residuals, alpha=0.6)
        ax.axhline(y=0, linestyle="--")

        ax.set_xlabel("Predicted Charges ($)")
        ax.set_ylabel("Residuals ($)")
        ax.set_title(f"Residual Plot — {title}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "residual_plots.png", dpi=150)
    plt.show()
    plt.close()