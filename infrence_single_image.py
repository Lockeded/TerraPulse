from pathlib import Path
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


def run_inference_for_single_image(model, image_path: Path, use_gpu: bool):
    """Run inference on a single image."""
    # Load the image and apply the five-crop transformation
    dataset = FiveCropImageDataset(meta_csv=None, image_dir=image_path.parent)  # Parent folder for dataset
    img_paths = [image_path]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

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


def save_results(df: pd.DataFrame, checkpoint_path: Path, image_path: Path):
    """Save the inference results to a CSV file."""
    df.set_index(keys=["img_id", "p_key"], inplace=True)
    fout = checkpoint_path.parent / f"inference_{image_path.stem}.csv"
    print("Write output to", fout)
    df.to_csv(fout)


def main(checkpoint: Path, hparams: Path, image_path: Path, use_gpu: bool):
    """Main function to run inference on a single image and save the results."""
    print("Load model from ", checkpoint)
    model = load_model(checkpoint, hparams, use_gpu)

    print("Run inference on single image:", image_path)
    df = run_inference_for_single_image(model, image_path, use_gpu)

    print(df)
    save_results(df, checkpoint, image_path)


if __name__ == '__main__':
    # Example of how to call the main function directly
    checkpoint_path = Path("models/base_M/epoch=014-val_loss=18.4833.ckpt")
    hparams_path = Path("models/base_M/hparams.yaml")
    image_path = Path("resources/images/im2gps/sample_image.jpg")  # Path to the image you want to predict
    use_gpu = True  # Set to False if you want to run on CPU

    main(checkpoint_path, hparams_path, image_path, use_gpu)
