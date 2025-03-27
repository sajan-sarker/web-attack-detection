import os
import json

def save_classification_reports(binary, multi, average, model_name, file_name):
    """ save the classification results into a json file """
    metrics = ["Train Acc", "Test Acc", "Train Loss", "Test Loss", "F1-score", "Precision", "Recall", "TPR", "FPR"]   # define the metrics keys

    # unpack the tuples
    binary_dict = {key: round(value, 4) for key, value in zip(metrics, binary)}
    multi_dict = {key: round(value, 4) for key, value in zip(metrics, multi)}
    avg_dict = {key: round(value, 4) for key, value in zip(metrics, average)}

    # structure the data
    model_data = {
        model_name: {
            "Binary Classification Results": binary_dict,
            "Multi-class Classification Results": multi_dict,
            "Average Classification Results": avg_dict
        }
    }
    try:
        if os.path.exists(file_name):  # check for file exists
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
            existing_data.update(model_data)
        else:
            existing_data = model_data
        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)
        print("Data saved successfully!")
    except Exception as e:
        print("Data not save!",e)
