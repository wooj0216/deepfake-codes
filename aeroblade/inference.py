import argparse
from pathlib import Path
from scripts.aeroblade import AEROBLADE

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the AE reconstruction distances for images or directories."
    )
    parser.add_argument(
        "--files-or-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("example_images/real.png"),
            Path("example_images/SD1-1.png"),
        ],
        help="Paths to images or directories containing images. All images in a directory should have the same dimensions.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="outputs/inference/example_images", help="Output directory."
    )
    parser.add_argument(
        "--autoencoders",
        nargs="+",
        default=[
            "CompVis/stable-diffusion-v1-1",  # SD1
            "stabilityai/stable-diffusion-2-base",  # SD2
            "kandinsky-community/kandinsky-2-1",  # KD2.1
        ],
        help="HuggingFace model name of an LDM to use for reconstruction. Non-default models might need adaptation.",
    )
    parser.add_argument(
        "--distance-metric",
        default="lpips_vgg_2",
        choices=[
            "lpips_vgg_0",  # sum of all layers, original LPIPS definition
            "lpips_vgg_1",  # first layer
            "lpips_vgg_2",  # second layer
            "lpips_vgg_3",  # third layer
            "lpips_vgg_4",  # fourth layer
            "lpips_vgg_5",  # fifth layer
            "lpips_vgg_-1",  # returns all layers
        ],
        help="Distance metric to use.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print the computed distances if this flag is set.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    aeroblade = AEROBLADE(
        files_or_dirs=args.files_or_dirs,
        output_dir=args.output_dir,
        autoencoders=args.autoencoders,
        distance_metric=args.distance_metric,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        print_results=args.print_results,
    )
    aeroblade.run()
