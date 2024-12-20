import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_from_stdin():
    if sys.stdin.isatty():
        print("Error: No file provided. Please pipe a CSV file into the script.")
        sys.exit(1)
    try:
        data = pd.read_csv(sys.stdin, header=None)
        if data.shape != (2, 2):
            print("Error: The provided data does not form a 2x2 matrix.")
            sys.exit(1)
        return data
    except pd.errors.EmptyDataError:
        print("Error: No data provided.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: Could not parse the provided CSV data.")
        sys.exit(1)

def main():
    data = read_csv_from_stdin()

    TN, FP = data.iloc[0]
    FN, TP = data.iloc[1]

    confusion_matrix = [[TN, FP],
                        [FN, TP]]

    x_labels = ["Predicted Negative", "Predicted Positive"]
    y_labels = ["Actual Negative", "Actual Positive"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    output_file = "confusion_matrix.png"
    plt.savefig(output_file)
    print(f"Confusion matrix saved to {output_file}")

if __name__ == "__main__":
    main()
