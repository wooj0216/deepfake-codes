import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
        else:  # sub-folder (class) in folder_path
            class_subfolders = os.listdir(folder_path)
            subfolders = [
                os.path.join(class_folder, sf) 
                for class_folder in class_subfolders
                for sf in ["0_real", "1_fake"] 
                if os.path.isdir(os.path.join(folder_path, class_folder, sf))
            ]

        for subfolder in subfolders:
            # If "0_real" is in the path, label=1; if "1_fake" is in the path, label=0
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
        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32).unsqueeze(0),
        )


def evaluate_model(model, data_loader, device, threshold=0.5, compute_auc=False, only_inference=False):
    """
    Evaluate model on a given DataLoader.
    Returns:
        y_true (list): ground-truth labels
        y_pred (list): hard-thresholded predictions
        y_prob (list): raw probabilities (only if compute_auc=True, else empty)
    """
    model.eval()
    y_pred = []
    y_true = []
    y_prob = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            # For classification:
            preds = (outputs >= threshold).float().cpu().numpy().flatten()
            y_pred.extend(preds.astype(int))
            y_true.extend(labels.cpu().numpy().flatten())

            if compute_auc or only_inference:
                probs = outputs.cpu().numpy().flatten()
                y_prob.extend(probs)

    return y_true, y_pred, y_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the test dataset root")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--results_file_or_dir", type=str, default="results", help="txt file or directory to save evaluation results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("--metric", type=str, required=False, nargs="+", default=["acc"], help="Metrics to evaluate")
    parser.add_argument("--only_inference", action="store_true", help="Only perform inference without evaluation")
    args = parser.parse_args()

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
    input_dim = checkpoint["linear.weight"].shape[1]
    model = LinearClassifier(input_dim)
    model.load_state_dict(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model checkpoint loaded.")

    compute_auc = "auc" in args.metric

    # Evaluate each folder
    print("Evaluating folders...")
    results = []
    all_accuracies = []
    all_f1s = []

    labels = []
    preds = []

    inference_results = {}

    for model_folder in tqdm(os.listdir(args.test_dir), desc="Evaluating Folders"):
        inference_results[model_folder] = []
        folder_path = os.path.join(args.test_dir, model_folder)
        if not os.path.isdir(folder_path):
            continue

        dataset = FolderDataset(folder_path)
        if len(dataset) == 0:
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        whole_samples = loader.dataset.samples
        y_true, y_pred, y_prob = evaluate_model(
            model, loader, device, threshold=args.threshold, compute_auc=compute_auc,
            only_inference=args.only_inference
        )

        if args.only_inference:
            for (sample_path, label), score, pred_label in zip(whole_samples, y_prob, y_pred):
                sample_path = sample_path.replace(args.test_dir + "/", "")
                pred_label = "real" if pred_label == 1 else "fake"
                inference_results[model_folder].append(f"('{sample_path}', {score:.7f}, {pred_label}),")
        else:
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=1)
            
            if compute_auc:
                labels.extend(y_true)
                preds.extend(y_prob)

            results.append((model_folder, accuracy, f1, None))

            all_accuracies.append(accuracy)
            all_f1s.append(f1)

    # Save results to file
    if args.only_inference:
        if not os.path.isdir(args.results_file_or_dir):
            output_dir = os.path.dirname(args.results_file_or_dir)
        else:
            output_dir = args.results_file_or_dir
        os.makedirs(output_dir, exist_ok=True)
        for model_folder, results in inference_results.items():
            with open(os.path.join(output_dir, f"{model_folder}.txt"), "w") as f:
                for result in results:
                    f.write(result + "\n")
        
        print(f"Inference results saved to {output_dir}")
        exit()

    if not os.path.isdir(args.results_file_or_dir):
        os.makedirs(args.results_file_or_dir, exist_ok=True)
        output_path = args.results_file_or_dir
    else:
        output_path = os.path.join(args.results_file_or_dir, "results.txt")

    os.makedirs(args.results_file_or_dir, exist_ok=True)
    output_path = os.path.join(args.results_file_or_dir, "results.txt")
    with open(output_path, "w") as f:
        for item in results:
            folder_name, accuracy, f1, auc_val = item
            if auc_val is not None:
                f.write(
                    f"Model Folder: {folder_name}, "
                    f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc_val:.4f}\n"
                )
            else:
                f.write(
                    f"Model Folder: {folder_name}, "
                    f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}\n"
                )

        # Overall metrics
        overall_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        overall_f1 = np.mean(all_f1s) if all_f1s else 0.0
        f.write(f"\nOverall Mean Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Overall Mean F1 Score: {overall_f1:.4f}\n")
        if compute_auc:
            if torch.unique(torch.tensor(labels)).shape[0] == 1:
                print("Only one class found in the dataset. Skipping AUC computation.")
            else:
                overall_auc = roc_auc_score(labels, preds)
                f.write(f"Overall Mean AUC: {overall_auc:.4f}\n")
    print(f"Results saved to {args.results_file_or_dir}")
    print("Done.")
