import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train_and_evaluate_classifiers(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define classifiers
    svm = SVC(probability=True)
    lda = LinearDiscriminantAnalysis()
    
    # Define parameter grids for tuning
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['linear', 'rbf']
    }
    
    lda_param_grid = {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
    }
    
    # Perform grid search
    svm_grid = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy')
    lda_grid = GridSearchCV(lda, lda_param_grid, cv=5, scoring='accuracy')
    
    # Train classifiers
    svm_grid.fit(X_train, y_train)
    lda_grid.fit(X_train, y_train)
    
    # Get best models
    best_svm = svm_grid.best_estimator_
    best_lda = lda_grid.best_estimator_
    
    # Make predictions
    svm_preds = best_svm.predict(X_test)
    lda_preds = best_lda.predict(X_test)
    
    # Evaluate models
    svm_acc = accuracy_score(y_test, svm_preds)
    lda_acc = accuracy_score(y_test, lda_preds)
    
    svm_cm = confusion_matrix(y_test, svm_preds)
    lda_cm = confusion_matrix(y_test, lda_preds)
    
    # Print results
    print(f"SVM Best Parameters: {svm_grid.best_params_}")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print("SVM Classification Report:")
    print(classification_report(y_test, svm_preds))
    
    print(f"LDA Best Parameters: {lda_grid.best_params_}")
    print(f"LDA Accuracy: {lda_acc:.4f}")
    print("LDA Classification Report:")
    print(classification_report(y_test, lda_preds))
    
    return {
        'X_test': X_test,
        'y_test': y_test,
        'svm': best_svm,
        'lda': best_lda,
        'svm_preds': svm_preds,
        'lda_preds': lda_preds,
        'svm_acc': svm_acc,
        'lda_acc': lda_acc,
        'svm_cm': svm_cm,
        'lda_cm': lda_cm
    }

def plot_confusion_matrices(results, label_names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(results['svm_cm'], interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(f'SVM Confusion Matrix (Acc: {results["svm_acc"]:.4f})')
    
    im2 = ax2.imshow(results['lda_cm'], interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title(f'LDA Confusion Matrix (Acc: {results["lda_acc"]:.4f})')
    
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticks(np.arange(len(label_names)))
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
    
    plt.tight_layout()
    return fig