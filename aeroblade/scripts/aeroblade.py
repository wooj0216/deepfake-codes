import argparse
from pathlib import Path
from aeroblade.high_level_funcs import compute_distances
from aeroblade.misc import safe_mkdir

class AEROBLADE:
    def __init__(
        self,
        files_or_dirs,
        output_dir,
        autoencoders,
        distance_metric="lpips_vgg_2",
        num_workers=1,
        batch_size=1,
        print_results=False,
    ):
        self.files_or_dirs = files_or_dirs
        self.output_dir = output_dir
        self.autoencoders = autoencoders
        self.distance_metric = distance_metric
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.print_results = print_results

    def run(self):
        safe_mkdir(self.output_dir)

        # Compute distances
        distances = compute_distances(
            dirs=self.files_or_dirs,
            transforms=["clean"],
            repo_ids=self.autoencoders,
            distance_metrics=[self.distance_metric],
            amount=None,
            reconstruction_root=self.output_dir / "reconstructions",
            seed=1,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # Print the computed distances
        if self.print_results:
            print("\n*** Computed Distances ***\n")
            print(distances)

        # Save the results
        distances.to_csv(self.output_dir / "distances.csv", index=False)
        print(f"\nSaving distances to {self.output_dir / 'distances.csv'}.\n")
