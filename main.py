import tkinter as tk
from data import load_subject_data
from data import preprocess_data
from feature_extractor import extract_features_csp
from classification import train_and_evaluate_classifiers
from ui import BCIInterface

# Main execution function
def run_bci_system():
    # Load and preprocess data for subject 1
    subject_id = 1
    data = load_subject_data(subject_id)
    
    print("Preprocessing data...")
    epochs, labels = preprocess_data(data)
    
    print("Extracting CSP features...")
    X, y, unique_labels = extract_features_csp(epochs, labels)
    
    print("Training and evaluating classifiers...")
    results = train_and_evaluate_classifiers(X, y)
    
    # Map numerical labels to class names
    label_names = ["Left Hand", "Right Hand", "Foot", "Tongue"]
    
    # Create the UI
    root = tk.Tk()
    bci_interface = BCIInterface(root, results, label_names)
    root.mainloop()

if __name__ == "__main__":
    run_bci_system()