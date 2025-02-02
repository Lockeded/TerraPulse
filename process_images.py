from argparse import ArgumentParser
import sys
from io import BytesIO
from pathlib import Path
import time
from multiprocessing import Pool
from functools import partial
import re
import logging
import msgpack
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MsgPackWriter:
    def __init__(self, path, chunk_size=4096):
        self.path = Path(path).absolute()
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        shards_re = r"shard_(\d+).msg"
        self.shards_index = [
            int(re.match(shards_re, x.name).group(1))
            for x in self.path.iterdir()
            if x.is_file() and re.match(shards_re, x.name)
        ]
        self.shard_open = None


    def open_next(self):
        if len(self.shards_index) == 0:
            next_index = 0
        else:
            next_index = sorted(self.shards_index)[-1] + 1
        self.shards_index.append(next_index)

        if self.shard_open is not None and not self.shard_open.closed:
            self.shard_open.close()

        self.count = 0
        self.shard_open = open(self.path / f"shard_{next_index}.msg", "wb")

    def __enter__(self):
        self.open_next()
        return self

    def __exit__(self, type, value, tb):
        self.shard_open.close()

    def write(self, data):
        if self.count >= self.chunk_size:
            self.open_next()

        self.shard_open.write(msgpack.packb(data))
        self.count += 1

def _thumbnail(img: Image, size: int) -> Image:
    w, h = img.size
    if w <= size and h <= size:
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), Image.BILINEAR)

def process_image(file_path: Path, min_edge_size: int):
    try:
        image = Image.open(file_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    
        image = _thumbnail(image, min_edge_size)
        fp = BytesIO()
        image.save(fp, "JPEG")
        raw_bytes = fp.getvalue()
        
        # Replace slashes (/) in the file name with underscores (_)
        image_id = str(file_path.stem).replace('_', '/')
        image_id += ".jpg"
        
        return {"image": raw_bytes, "id": image_id}
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--threads",
        type=int,
        default=24,
        help="Number of threads to process images",
    )
    args.add_argument(
        "--output",
        type=Path,
        default=Path("D:mp16"),
        help="Output directory where images are stored",
    )
    args.add_argument(
        "--input_dir",
        default=Path("D:mp16/downloads"),
        type=Path,
        help="Input directory containing images",
    )
    args.add_argument(
        "--size",
        type=int,
        default=320,
        help="Rescale image to a minimum edge size of SIZE",
    )
    return args.parse_args()

def main():
    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory {input_dir} does not exist or is not a directory.")
        return 1

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} images in {input_dir}.")

    counter_successful = 0
    with Pool(args.threads) as p:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            for i, x in enumerate(
                p.imap(partial(process_image, min_edge_size=args.size), image_files)
            ):
                if x is None:
                    continue

                f.write(x)
                counter_successful += 1

                if i % 100 == 0:
                    end = time.time()
                    logger.info(f"{i}: {100 / (end - start):.2f} image/s")
                    start = end

    logger.info(
        f"Successfully processed {counter_successful}/{len(image_files)} images ({counter_successful / len(image_files):.3f})"
    )
    return 0

logger = logging.getLogger("ImageProcessor")
if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(args.output / "processor.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    sys.exit(main())
