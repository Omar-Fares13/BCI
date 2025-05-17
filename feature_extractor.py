import numpy as np
from mne.decoding import CSP


def extract_features_csp(epochs, labels, n_components=4):
    # Convert labels to 0-based indexing if needed
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[label] for label in labels])
    
    # Initialize CSP for each class (one-vs-rest approach)
    csp_features = []
    
    # Add regularization to help with numerical stability
    reg = 0.1  # Tried values between 0.1 and 1.0 and 0.1 gave best results
    
    for class_idx in range(len(unique_labels)):
        # Create binary labels (1 for current class, 0 for others)
        binary_labels = (mapped_labels == class_idx).astype(int)
        
        # Apply CSP with regularization
        csp = CSP(n_components=n_components, reg=reg)
        
        try:
            csp.fit(epochs, binary_labels)
            features = csp.transform(epochs)
            csp_features.append(features)
        except Exception as e:
            print(f"Error in CSP for class {class_idx}: {e}")
            print("Trying with higher regularization...")
            # Try with higher regularization
            csp = CSP(n_components=n_components, reg=1.0)
            try:
                csp.fit(epochs, binary_labels)
                features = csp.transform(epochs)
                csp_features.append(features)
            except Exception as e2:
                print(f"Still failed with error: {e2}")
                # Fallback: use zeros as features
                print("Using fallback features for this class...")
                features = np.zeros((epochs.shape[0], n_components))
                csp_features.append(features)
    
    # Concatenate features from all CSP transformations
    X = np.hstack(csp_features)
    y = mapped_labels
    
    return X, y, unique_labels