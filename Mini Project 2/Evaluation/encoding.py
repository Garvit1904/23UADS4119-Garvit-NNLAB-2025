import numpy as np

# Load the label classes (road condition names)
label_classes_path = "Data/label_classes.npy"

try:
    label_classes = np.load(label_classes_path, allow_pickle=True)  # ← fix is here
    print("Label classes and their corresponding indices:\n")
    for idx, label in enumerate(label_classes):
        print(f"{idx}: {label}")
except FileNotFoundError:
    print(f"File not found: {label_classes_path}")
except Exception as e:
    print(f"An error occurred: {e}")

'''Label classes and their corresponding indices:

0: Bitumin
1: Block
2: Concrete
3: Kanker'''