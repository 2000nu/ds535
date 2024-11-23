from collections import defaultdict
import pandas as pd
import re  # For extracting recall@10 values
import os

# Dictionary to store results
results_dict = defaultdict(list)

model = 'dsl'
file_name = 'ciao-tune_Nov-23-2024_14-03-50'
file_path = os.path.join(model, file_name + ".log")

# Regular expression to extract recall@10
recall_pattern = re.compile(r"recall@10:\s*([\d.]+)")

highest_recall = 0  # Variable to track the highest recall
best_result = None  # To store the best hyperparameter and its results

with open(file_path, 'r') as file:
    current_hyperparameters = None
    best_epoch_line = None

    for line in file:
        if "hyperparameter" in line:
            # Extract hyperparameter line
            current_hyperparameters = line.split(" - ", 1)[-1].strip()
            best_epoch_line = None
        elif "Best Epoch" in line:
            # Extract Best Epoch without timestamp
            best_epoch_line = line.split(" - ", 1)[-1].strip()
        elif "Test set" in line and current_hyperparameters and best_epoch_line:
            # Extract Test set metrics without timestamp
            test_set_line = line.split(" - ", 1)[-1].strip()
            
            # Extract recall@10 from the test set line
            recall_match = recall_pattern.search(test_set_line)
            if recall_match:
                recall_value = float(recall_match.group(1))
                
                # Check if this is the highest recall
                if recall_value > highest_recall:
                    highest_recall = recall_value
                    best_result = {
                        "Hyperparameters": current_hyperparameters,
                        "Best Epoch": best_epoch_line,
                        "Test Set": test_set_line,
                        "Recall@10": recall_value
                    }

# Display the best result
if best_result:
    print(best_result["Hyperparameters"])
    print(best_result["Best Epoch"])
    test_result = best_result["Test Set"]
    recall_pattern = re.compile(r"recall@(\d+):\s*([\d.]+)")
    ndcg_pattern = re.compile(r"ndcg@(\d+):\s*([\d.]+)")
    recall_matches = recall_pattern.findall(test_result)
    ndcg_matches = ndcg_pattern.findall(test_result)
    values = [match[1] for match in recall_matches] + [match[1] for match in ndcg_matches]


    print(" & ".join(values))
else:
    print("No valid recall@10 value found in the log file.")
