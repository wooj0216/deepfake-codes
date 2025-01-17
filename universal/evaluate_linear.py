import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import argparse


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class FolderDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []

        # Both classes and corresponding (0_real, 1_fake) folders exist
        if "0_real" in os.listdir(folder_path) and "1_fake" in os.listdir(folder_path):
            subfolders = ["0_real", "1_fake"]
        # Only 0_real or 1_fake folder exists
        elif "0_real" in os.listdir(folder_path) or "1_fake" in os.listdir(folder_path):
            subfolders = ["0_real"] if "0_real" in os.listdir(folder_path) else ["1_fake"]
        else: # class in folder_path
            class_subfolders = os.listdir(folder_path)
            subfolders = [os.path.join(class_folder, sf) for class_folder in class_subfolders
                            for sf in ["0_real", "1_fake"] if os.path.isdir(os.path.join(folder_path, class_folder, sf))]

        for subfolder in subfolders:
            label = 1 if "0_real" in subfolder else 0
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.exists(subfolder_path):
                continue

            for file in os.listdir(subfolder_path):
                if file.endswith(".npz"):
                    self.samples.append((os.path.join(subfolder_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)
        feature = data["arr_0"].flatten()
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)


def evaluate_model(model, data_loader, device):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int).flatten())
            y_true.extend(labels.cpu().numpy().flatten())

    return y_pred, y_true


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the test dataset root")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--results_file", type=str, default="evaluation_results.txt", help="File to save evaluation results")
    args = parser.parse_args()

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    input_dim = checkpoint['linear.weight'].shape[1]
    model = LinearClassifier(input_dim)
    model.load_state_dict(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model checkpoint loaded.")

    # Evaluate each folder
    results = []
    print("Evaluating folders...")
    for model_folder in tqdm(os.listdir(args.test_dir), desc="Evaluating Folders"):
        folder_path = os.path.join(args.test_dir, model_folder)
        if not os.path.isdir(folder_path):
            continue
        
        dataset = FolderDataset(folder_path)
        if len(dataset) == 0:
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        y_pred, y_true = evaluate_model(model, loader, device)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        results.append((model_folder, accuracy, f1))

    # Save results to file
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        for model_folder, accuracy, f1 in results:
            f.write(f"Model Folder: {model_folder}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}\n")
        overall_accuracy = np.mean([acc for _, acc, _ in results])
        overall_f1 = np.mean([f1 for _, _, f1 in results])
        f.write(f"\nOverall Mean Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Overall Mean F1 Score: {overall_f1:.4f}")
    print(f"Results saved to {args.results_file}")
