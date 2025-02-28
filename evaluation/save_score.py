import json

def save_accuracy_score(name, train, validation, test, path):
  data = {
    name:{
      "train_acc": train,
      "val_acc": validation,
      "test_acc": test
    }
  }
  
  try: 
    with open(path, 'r') as file:
      current_data = json.load(file)
  except FileNotFoundError:
    #current_data = {}
    print("File not Found!")
  
  current_data.update(data)
  
  try: 
    with open(path, 'w') as file:
      json.dump(current_data, file, indent=4)
  except Exception as e:
    print("Unable to save data:", e)
  
  print(f"Accuracy data for {name} save successfully!")
  