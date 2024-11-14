import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report, f1_score
import matplotlib.pyplot as plt

def get_f1_score(y_true, y_pred, average='weighted', report=False):
    if report:
       
        print("Classification Report:\n")
        print(classification_report(y_true, y_pred))
    else:
        f1 = f1_score(y_true, y_pred, average=average)
        print(f"F1 Score: {f1:.4f}")
        return f1

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")

    plt.close(fig)

    return fig  
