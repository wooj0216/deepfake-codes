import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Custom Dataset
class FeatureDataset(Dataset):
    def __init__(self, dataset_root):
        self.data = self._load_feature_paths(dataset_root)

    @staticmethod
    def _load_feature_paths(dataset_root):
        feature_paths = []
        for class_folder in os.listdir(dataset_root):
            class_path = os.path.join(dataset_root, class_folder)
            if not os.path.isdir(class_path):
                continue
            for label_name, label in [("0_real", 1), ("1_fake", 0)]:
                label_folder = os.path.join(class_path, label_name)
                if not os.path.exists(label_folder):
                    print(f"Warning: Missing folder {label_folder}, skipping.")
                    continue
                for file in os.listdir(label_folder):
                    if file.endswith(".npz"):
                        feature_paths.append((os.path.join(label_folder, file), label))
        return feature_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        data = np.load(file_path)
        feature = data['arr_0'].reshape(-1)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Model Definition
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()

# Training Function with Iteration-Level Evaluation
def train_model_with_eval_iteration(
    model, model_type, train_loader, test_loader, criterion, optimizer, num_epochs, save_dir, batch_size, eval_interval=500
):
    best_accuracy = 0.0
    best_iteration = 0
    loss_history = []
    accuracy_history = []
    eval_accuracy_history = []

    total_iterations = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for features, labels in pbar:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * features.size(0)
                total_iterations += 1

                pbar.set_description(f"Loss: {running_loss / (total_iterations * batch_size):.3f}")

                # Evaluate at predefined iterations
                if total_iterations % eval_interval == 0:
                    model.eval()
                    eval_accuracy = evaluate_model(model, test_loader, log=False)
                    eval_accuracy_history.append((total_iterations, eval_accuracy))
                    print(
                        f"Iteration {total_iterations}, Eval Accuracy: {eval_accuracy:.4f}"
                    )

                    # Save the model if it achieves the best evaluation accuracy
                    if eval_accuracy > best_accuracy:
                        best_accuracy = eval_accuracy
                        best_iteration = total_iterations
                        torch.save(
                            model.state_dict(), os.path.join(save_dir, f"{model_type}_weights.pth")
                        )
                        print(
                            f"New best model saved at iteration {total_iterations} with eval accuracy: {eval_accuracy:.4f}"
                        )

                    model.train()

            epoch_loss = running_loss / len(train_loader.dataset)
            loss_history.append(epoch_loss)

    print(f"Best Eval Accuracy: {best_accuracy:.4f} at Iteration {best_iteration}")

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss (Epoch)")
    if eval_accuracy_history:
        eval_iters, eval_accs = zip(*eval_accuracy_history)
        plt.plot(eval_iters, eval_accs, label="Eval Accuracy (Iteration)", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Training Loss and Evaluation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_curve_with_eval_iter.png"))
    plt.show()

    return best_accuracy, best_iteration

# Evaluation Function
def evaluate_model(model, data_loader, log=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = (outputs >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    if log:
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    return accuracy

# Main Script
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--model_type", type=str, default="dino", help="Model type, such as 'dino' or 'clip'")
    arg_parser.add_argument("--train_dir", type=str, default="features/dino/progan_train")
    arg_parser.add_argument("--test_dir", type=str, default="features/dino/cnn_synth_test")
    arg_parser.add_argument("--batch_size", type=int, default=512)
    arg_parser.add_argument("--num_epochs", type=int, default=10)
    arg_parser.add_argument("--learning_rate", type=float, default=0.001)
    arg_parser.add_argument("--eval_interval", type=int, default=20)  # Evaluate every 500 iterations
    arg_parser.add_argument("--save_dir", type=str, default="pretrained_models")

    args = arg_parser.parse_args()

    print("Arguments:", args)

    os.makedirs(args.save_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets and loaders
    train_dataset = FeatureDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = FeatureDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, criterion, and optimizer
    sample_feature, _ = train_dataset[0]
    input_dim = sample_feature.shape[0]

    model = BinaryClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    print("Training the model with iteration-level evaluation...")
    best_accuracy, best_iteration = train_model_with_eval_iteration(
        model, args.model_type, train_loader, test_loader, criterion, optimizer, args.num_epochs, args.save_dir, args.batch_size, args.eval_interval
    )
    
    # Load the best model for final evaluation
    print("Evaluating the best model...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.model_type}_weights.pth")))
    evaluate_model(model, test_loader)
