import pickle
import pandas as pd
import numpy as np

# Load everything from the .pkl file
with open("model_with_explainer.pkl", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["model"]
explainer = saved_data["explainer"]
label_encoder = saved_data["label_encoder"]
feature_encoders = saved_data["feature_encoders"]
feature_names = saved_data["feature_names"]
class_names = saved_data["class_names"]


def predict_with_explanation(input_data):
    """
    Make prediction and generate SHAP explanation for a single example
    input_data: dict with feature names as keys
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Preprocess categorical features
    for col, encoder in feature_encoders.items():
        if col in input_df.columns:
            original_value = input_df[col].iloc[0]
            # Handle unseen categories
            if original_value not in encoder.classes_:
                # Use the first known category as default
                input_df[col] = encoder.classes_[0]
            else:
                input_df[col] = encoder.transform([original_value])[0]

    # Ensure correct column order and handle missing columns
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0  # default value for missing features

    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Convert to class name
    predicted_class = label_encoder.inverse_transform([prediction])[0]

    # Get class probabilities
    class_probs = {}
    for i, class_name in enumerate(class_names):
        class_probs[class_name] = float(probabilities[i])

    # Generate SHAP explanation
    shap_values = explainer.shap_values(input_df)
    base_value = explainer.expected_value

    # Prepare SHAP explanation
    if isinstance(shap_values, list):
        # Old SHAP format: list of arrays (one per class)
        shap_for_pred = shap_values[prediction][0]
        base_val = (
            base_value[prediction] if hasattr(base_value, "__len__") else base_value
        )
    else:
        # New SHAP format: array shaped (1, n_features, n_classes)
        # Select predicted class correctly
        shap_for_pred = shap_values[0, :, prediction]

        if isinstance(base_value, (list, np.ndarray)):
            base_val = base_value[prediction]
        else:
            base_val = base_value

        # Create feature contributions
        feature_contributions = {}
        for i, feature in enumerate(feature_names):
            feature_contributions[feature] = {
                "contribution": float(shap_for_pred[i]),
                "value": float(input_df.iloc[0, i]),
                "feature_name": feature,
            }

    return {
        "prediction": {
            "predicted_class": predicted_class,
            "predicted_class_index": int(prediction),
            "confidence": float(probabilities[prediction]),
            "all_probabilities": class_probs,
        },
        "explanation": {
            "base_value": float(base_val),
            "feature_contributions": feature_contributions,
            "prediction_score": float(base_val + np.sum(shap_for_pred)),
        },
    }


def display_result(result, row_index=None):
    """
    Display prediction and explanation results
    """
    if row_index is not None:
        print(f"\n" + "=" * 60)
        print(f"RESULTS FOR ROW {row_index}")
        print("=" * 60)
    else:
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)

    print(f"Predicted class: {result['prediction']['predicted_class']}")
    print(f"Confidence: {result['prediction']['confidence']:.4f}")

    print("\nClass Probabilities:")
    for class_name, prob in sorted(
        result["prediction"]["all_probabilities"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {class_name}: {prob:.4f}")

    print("\nEXPLANATION:")
    print(f"Base value: {result['explanation']['base_value']:.4f}")
    print(f"Prediction score: {result['explanation']['prediction_score']:.4f}")

    print("\nTop Feature Contributions:")
    contributions = result["explanation"]["feature_contributions"]
    # Sort by absolute contribution value
    sorted_contributions = sorted(
        contributions.items(), key=lambda x: abs(x[1]["contribution"]), reverse=True
    )

    for i, (feature, info) in enumerate(sorted_contributions[:10], 1):  # Show top 10
        sign = "+" if info["contribution"] > 0 else ""
        print(
            f"  {i:2d}. {feature}: {sign}{info['contribution']:.4f} (value: {info['value']})"
        )


def process_csv_file(csv_file_path, start_row=0, max_rows=None, interactive=True):
    """
    Process a CSV file row by row for predictions and explanations

    Parameters:
    - csv_file_path: path to the CSV file
    - start_row: starting row index (0-based)
    - max_rows: maximum number of rows to process (None for all)
    - interactive: whether to pause after each prediction
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        # Determine rows to process
        end_row = len(df) if max_rows is None else min(start_row + max_rows, len(df))
        rows_to_process = range(start_row, end_row)

        print(f"Processing rows {start_row} to {end_row-1}")

        for i in rows_to_process:
            row = df.iloc[i]

            print(f"\n" + "=" * 60)
            print(f"PROCESSING ROW {i}")
            print("=" * 60)
            print("Input features:")

            # Display first few features to show what's being processed
            for j, (feature, value) in enumerate(row.items()):
                if j < 5:  # Show first 5 features
                    print(f"  {feature}: {value}")
                elif j == 5:
                    print(f"  ... and {len(row) - 5} more features")
                    break

            # Convert row to dictionary
            input_data = row.to_dict()

            # Make prediction and get explanation
            result = predict_with_explanation(input_data)

            # Display results
            display_result(result, row_index=i)

            # Interactive mode: ask to continue
            if interactive and i < end_row - 1:
                print(f"\n" + "-" * 40)
                continue_choice = (
                    input("Press Enter to continue to next row, or 'q' to quit: ")
                    .strip()
                    .lower()
                )
                if continue_choice == "q":
                    print(f"Stopped at row {i}. Processed {i - start_row + 1} rows.")
                    break
            elif not interactive:
                print(f"Completed row {i} - {i - start_row + 1}/{len(rows_to_process)}")

        if not interactive:
            print(f"\nProcessing completed! Processed {len(rows_to_process)} rows.")

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
    except Exception as e:
        print(f"Error processing CSV file: {e}")


def batch_process_csv(csv_file_path, output_file=None, start_row=0, max_rows=None):
    """
    Process entire CSV file in batch mode and save results
    """
    try:
        df = pd.read_csv(csv_file_path)
        results = []

        end_row = len(df) if max_rows is None else min(start_row + max_rows, len(df))

        print(f"Batch processing {end_row - start_row} rows...")

        for i in range(start_row, end_row):
            row = df.iloc[i]
            input_data = row.to_dict()

            # Make prediction
            result = predict_with_explanation(input_data)

            # Store results
            row_result = {
                "row_index": i,
                "predicted_class": result["prediction"]["predicted_class"],
                "confidence": result["prediction"]["confidence"],
                "top_features": [],
            }

            # Get top 5 features by contribution
            contributions = result["explanation"]["feature_contributions"]
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]["contribution"]),
                reverse=True,
            )[:5]

            for feature, info in sorted_contributions:
                row_result["top_features"].append(
                    {
                        "feature": feature,
                        "contribution": info["contribution"],
                        "value": info["value"],
                    }
                )

            results.append(row_result)

            # Progress indicator
            if (i - start_row + 1) % 10 == 0:
                print(f"Processed {i - start_row + 1}/{end_row - start_row} rows")

        # Save results if output file specified
        if output_file:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return results

    except Exception as e:
        print(f"Error in batch processing: {e}")
        return []


# Main execution
if __name__ == "__main__":
    csv_file_path = "kdd_test.csv"  # Change this to your CSV file path

    print("Model loaded successfully!")
    print(f"Available classes: {class_names}")
    print(f"Total features: {len(feature_names)}")

    # Option 1: Interactive mode (process one row at a time with pauses)
    print("\nStarting interactive CSV processing...")
    process_csv_file(csv_file_path, start_row=0, interactive=True)

    # Option 2: Uncomment below for batch processing
    # print("\nStarting batch processing...")
    # batch_results = batch_process_csv(csv_file_path, output_file="predictions_results.csv")

    # Option 3: Uncomment below to process specific range without interaction
    # process_csv_file(csv_file_path, start_row=0, max_rows=5, interactive=False)
