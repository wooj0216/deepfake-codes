import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from tqdm import tqdm


def train_load_features_from_dataset(dataset_root):
    features = []
    labels = []
    count = 0

    # Collect all parent directories with 0_real or 1_fake
    parent_dirs = [
        os.path.basename(root)
        for root, dirs, files in os.walk(dataset_root)
        for label_name in ["0_real", "1_fake"]
        if label_name in dirs
    ]
    
    # Use tqdm with parent directories
    for parent_dir in tqdm(parent_dirs):
        for root, dirs, files in os.walk(dataset_root):
            if os.path.basename(root) == parent_dir:
                for label_name, label in [("0_real", 1), ("1_fake", 0)]:
                    if label_name in dirs:
                        label_folder = os.path.join(root, label_name)
                        for file in os.listdir(label_folder)[:1000]:  # Limit to 1000 samples per class
                            if file.endswith(".npz"):
                                data = np.load(os.path.join(label_folder, file))
                                feature = data['arr_0']
                                feature = feature.reshape(-1)
                                features.append(feature)
                                labels.append(label)
                                count += 1

    print(f"Loaded {count} total files from training dataset.")
    return np.array(features), np.array(labels)


def test_load_features_from_dataset(dataset_root):
    features = []
    labels = []
    count = 0

    # Collect all parent directories with 0_real or 1_fake
    parent_dirs = [
        os.path.basename(root)
        for root, dirs, files in os.walk(dataset_root)
        for label_name in ["0_real", "1_fake"]
        if label_name in dirs
    ]
    
    # Use tqdm with parent directories
    for parent_dir in tqdm(parent_dirs):
        for root, dirs, files in os.walk(dataset_root):
            if os.path.basename(root) == parent_dir:
                for label_name, label in [("0_real", 1), ("1_fake", 0)]:
                    if label_name in dirs:
                        label_folder = os.path.join(root, label_name)
                        for file in os.listdir(label_folder)[:1000]:  # Limit to 1000 samples per class
                            if file.endswith(".npz"):
                                data = np.load(os.path.join(label_folder, file))
                                feature = data['arr_0']
                                feature = feature.reshape(-1)
                                features.append(feature)
                                labels.append(label)
                                count += 1

    print(f"Loaded {count} files from testing dataset.")
    return np.array(features), np.array(labels)


def test(knn, test_root):
    # Load the testing data
    print("Loading testing data...")
    X_test, y_test = test_load_features_from_dataset(test_root)
    print(f"Testing data shape: Features {X_test.shape}, Labels {y_test.shape}")

    # Test the classifier
    print("Making predictions on the test set...")
    y_pred = knn.predict(X_test)

    return y_pred, y_test

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--model_type", type=str, default="dino",
        help="Model type, such as 'dino' or 'clip'")
    arg_parser.add_argument("--train_dir", type=str, default="features/dino/progan_train")
    arg_parser.add_argument("--test_dir", type=str, default="features/dino/cnn_synth_test")
    arg_parser.add_argument("--gen_type", type=str, default="gan")
    arg_parser.add_argument("--logging_dir", type=str, default="results")
    
    arg_parser.add_argument("--num_k", type=int, default=5, help="Number of k in KNN")

    args = arg_parser.parse_args()

    if args.gen_type == "gan":
        data_names = ['progan', 'cyclegan', 'biggan', 'stylegan2', 'gaugan', 'stargan', 'deepfake', 'seeingdark', 'san', 'crn', 'imle']
    elif args.gen_type == "diffusion":
        data_names = ['guided', 'ldm_200', 'ldm_200_cfg', 'ldm_100', 'glide_100_27', 'glide_50_27', 'glide_100_10', 'dalle']

    log_dir = os.path.join(args.logging_dir, args.model_type)
    os.makedirs(log_dir, exist_ok=True)
    logging_path = os.path.join(log_dir, f"knn_{args.num_k}_{args.gen_type}.txt")
    print("\nLogging path:", logging_path)

    # Load the training data
    X_train, y_train = train_load_features_from_dataset(args.train_dir)
    print(f"Training data shape: Features {X_train.shape}, Labels {y_train.shape}")

    # Train a k-NN classifier
    print("Training k-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=args.num_k, metric='euclidean')
    knn.fit(X_train, y_train)
    print("k-NN classifier training completed.")

    with open(logging_path, "w", encoding="utf-8") as f:
        for data_name in data_names:
            print(f"\n----------Testing on {data_name}----------\n")
            test_path = os.path.join(args.test_dir, data_name)
            y_pred, y_test = test(knn, test_path)

            # Evaluation and Logging
            accuracy = accuracy_score(y_test, y_pred)
            f.write(f"Data: {data_name}\n")
            f.write(f"Accuracy: {accuracy}")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(confusion_matrix(y_test, y_pred)))
            f.write("\n\n")

    print("Evaluation completed.")
    print("The results are saved in", logging_path)
    print("\n\n")
