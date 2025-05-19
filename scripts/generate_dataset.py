import argparse
from pathlib import Path

from src.api.lorem_picsum import LoremPicsum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)

    args = parser.parse_args()
    width = args.width
    height = args.height

    output_path = Path(args.output)
    dataset_dir = f"stock_images_{width}x{height}"
    color_image_dir = output_path / dataset_dir / "color"
    grayscale_image_dir = output_path / dataset_dir / "grayscale"

    color_image_dir.mkdir(parents=True, exist_ok=True)
    grayscale_image_dir.mkdir(parents=True, exist_ok=True)

    api = LoremPicsum(width, height)
    image_ids = LoremPicsum.get_image_ids()

    # Iterate through image_ids instead of a range as some IDs are missing
    for idx, image_id in enumerate(image_ids):
        color_output_path = color_image_dir / f"{idx}.jpg"
        grayscale_output_path = grayscale_image_dir / f"{idx}.jpg"

        api.download_image(color_output_path, image_id, grayscale=False)
        api.download_image(grayscale_output_path, image_id, grayscale=True)

        print(
            f"Image {image_id} successfully downloaded ({idx + 1} / {len(image_ids)})"
        )

    print("All done.")


if __name__ == "__main__":
    main()
