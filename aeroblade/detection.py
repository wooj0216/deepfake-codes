"""
Compute reconstruction distances and detection/attribution results.
"""

import argparse
from pathlib import Path

import pandas as pd
from aeroblade.evaluation import tpr_at_max_fpr
from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir, write_config
from sklearn.metrics import average_precision_score


def main(args):
    output_dir = args.save_dir
    safe_mkdir(output_dir)
    write_config(vars(args), output_dir)

    # compute distances for both real and fake images
    dirs = [args.real_dir] + args.fake_dirs

    distances = compute_distances(
        dirs=dirs,
        transforms=args.transforms,
        repo_ids=args.repo_ids,
        distance_metrics=args.distance_metrics,
        amount=args.amount,
        reconstruction_root=args.reconstruction_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # store distances
    categoricals = [
        "dir",
        "image_size",
        "repo_id",
        "transform",
        "distance_metric",
        "file",
    ]
    distances[categoricals] = distances[categoricals].astype("category")
    distances.to_parquet(output_dir / "distances.parquet")

    # compute detection results
    detection_results = []
    for (transform, repo_id, dist_metric), group_df in distances.groupby(
        ["transform", "repo_id", "distance_metric"], sort=False, observed=True
    ):
        y_score_real = group_df.query("dir == @args.real_dir.__str__()").distance.values
        for fake_dir in args.fake_dirs:
            y_score_fake = group_df.query("dir == @fake_dir.__str__()").distance.values
            y_score = y_score_real.tolist() + y_score_fake.tolist()
            y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)
            ap = average_precision_score(y_true=y_true, y_score=y_score)
            tpr5fpr = tpr_at_max_fpr(y_true=y_true, y_score=y_score, max_fpr=0.05)
            detection_results.append(
                {
                    "fake_dir": fake_dir,
                    "transform": transform,
                    "repo_id": repo_id,
                    "distance_metric": dist_metric,
                    "ap": ap,
                    "tpr5fpr": tpr5fpr,
                }
            )
    pd.DataFrame(detection_results).sort_values("fake_dir", kind="stable").to_csv(
        output_dir / "detection_results.csv"
    )
    print("The detection results are saved to", output_dir / "detection_results.csv")

    # compute attribution results
    attribution_results = []
    for (dir, transform, dist_metric), group_df in distances.groupby(
        ["dir", "transform", "distance_metric"], sort=False, observed=True
    ):
        for repo_id, repo_id_df in group_df.groupby(
            "repo_id", sort=False, observed=True
        ):
            if repo_id == "max":
                continue
            matches = (
                repo_id_df.distance.values
                == group_df.query("repo_id == 'max'").distance.values
            )
            fraction = matches.sum() / len(repo_id_df)
            attribution_results.append(
                {
                    "dir": dir,
                    "transform": transform,
                    "distance_metric": dist_metric,
                    "repo_id": repo_id,
                    "fraction": fraction,
                }
            )
    pd.DataFrame(attribution_results).sort_values("dir", kind="stable").to_csv(
        output_dir / "attribution_results.csv"
    )
    print("The attribution results are saved to", output_dir / "attribution_results.csv")
    


def parse_args():
    parser = argparse.ArgumentParser()

    # images
    parser.add_argument("--real-dir", type=Path, default="data/raw/real/laion/0_real")
    parser.add_argument(
        "--fake-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/generated/dalle/1_fake"),
            Path("data/raw/generated/ldm_100/1_fake"),
        ],
    )
    parser.add_argument("--amount", type=int)
    parser.add_argument("--transforms", nargs="*", default=["clean"])
    parser.add_argument(
        "--reconstruction-root", type=Path, default="data/reconstructions"
    )
    parser.add_argument(
        "--save-dir", type=Path, default="outputs/detections"
    )

    # autoencoder
    parser.add_argument(
        "--repo-ids",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",
            "stabilityai/stable-diffusion-2-base",
            "kandinsky-community/kandinsky-2-1",
        ],
    )
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=[
            "lpips_vgg_-1",
        ],
    )

    # technical
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
