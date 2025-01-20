from pathlib import Path
from math import ceil
import pandas as pd
import torch
from tqdm.auto import tqdm

from classification.train_base import MultiPartitioningClassifier
from classification.dataset import FiveCropImageDataset


def load_model(checkpoint_path: Path, hparams_path: Path, use_gpu: bool):
    """Load the trained model from checkpoint."""
    model = MultiPartitioningClassifier.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        hparams_file=str(hparams_path),
        map_location=None,
    )
    model.eval()
    if use_gpu and torch.cuda.is_available():
        model.cuda()
    return model


def init_dataloader(image_dir: Path, batch_size: int, num_workers: int):
    """Initialize the DataLoader for the dataset."""
    dataloader = torch.utils.data.DataLoader(
        FiveCropImageDataset(meta_csv=None, image_dir=image_dir),
        batch_size=ceil(batch_size / 5),  # Adjusting batch size for 5 crops
        shuffle=False,
        num_workers=num_workers,
    )
    if len(dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    return dataloader


def run_inference(model, dataloader, use_gpu: bool):
    """Run inference on the dataset."""
    rows = []
    for X in tqdm(dataloader):
        if use_gpu:
            X[0] = X[0].cuda()
        img_paths, pred_classes, pred_latitudes, pred_longitudes = model.inference(X)
        for p_key in pred_classes.keys():
            for img_path, pred_class, pred_lat, pred_lng in zip(
                img_paths,
                pred_classes[p_key].cpu().numpy(),
                pred_latitudes[p_key].cpu().numpy(),
                pred_longitudes[p_key].cpu().numpy(),
            ):
                rows.append(
                    {
                        "img_id": Path(img_path).stem,
                        "p_key": p_key,
                        "pred_class": pred_class,
                        "pred_lat": pred_lat,
                        "pred_lng": pred_lng,
                    }
                )
    return pd.DataFrame.from_records(rows)


def save_results(df: pd.DataFrame, checkpoint_path: Path, image_dir: Path):
    """Save the inference results to a CSV file."""
    df.set_index(keys=["img_id", "p_key"], inplace=True)
    fout = checkpoint_path.parent / f"inference_{image_dir.stem}.csv"
    print("Write output to", fout)
    df.to_csv(fout)


def main(checkpoint: Path, hparams: Path, image_dir: Path, use_gpu: bool, batch_size: int, num_workers: int):
    """Main function to run inference and save the results."""
    print("Load model from ", checkpoint)
    model = load_model(checkpoint, hparams, use_gpu)

    print("Init dataloader")
    dataloader = init_dataloader(image_dir, batch_size, num_workers)

    print("Number of images: ", len(dataloader.dataset))

    df = run_inference(model, dataloader, use_gpu)

    print(df)
    save_results(df, checkpoint, image_dir)


if __name__ == '__main__':
    # Example of how to call the main function directly
    checkpoint_path = Path("models/base_M/epoch=014-val_loss=18.4833.ckpt")
    hparams_path = Path("models/base_M/hparams.yaml")
    image_dir = Path("resources/images/im2gps")
    use_gpu = True  # Set to False if you want to run on CPU
    batch_size = 64
    num_workers = 4

    main(checkpoint_path, hparams_path, image_dir, use_gpu, batch_size, num_workers)
