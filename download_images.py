import msgpack
import pandas as pd
import PIL
from PIL import ImageFile
from argparse import ArgumentParser
import sys
from io import BytesIO
from pathlib import Path
import time
import random
from multiprocessing import Pool
from functools import partial
import logging
import requests

# 新增请求头配置
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
]

ImageFile.LOAD_TRUNCATED_IMAGES = True
start_index = 40000
end_index = 50000

def download_image(x, output_dir):
    image_id = x["image_id"]
    url = x["url"]
    
    # 创建会话保持连接
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })

    save_path = output_dir / f"{image_id.replace('/', '_')}"
    max_retries = 3
    retry_delay = 3  # 缩短初始延迟秒数

    for attempt in range(max_retries):
        try:
            # 减少延迟，缩短到0.1-0.5秒之间
            time.sleep(0.1 + random.random() * 0.4)  # 快速请求

            response = session.get(url, timeout=10)
            response.raise_for_status()

            # 验证图片有效性
            image = PIL.Image.open(BytesIO(response.content))
            if image.mode != "RGB":
                image = image.convert("RGB")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path, "JPEG")
            logger.info(f"{image_id} : Downloaded")
            return save_path

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # 处理频率限制
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"Rate limited: {image_id}. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time + random.uniform(0, 2))  # 添加随机抖动
                continue
            logger.error(f"{image_id} : HTTP {response.status_code} - {str(e)}")
            return None
        except Exception as e:
            logger.error(f"{image_id} : {type(e).__name__} - {str(e)}")
            return None

    logger.error(f"{image_id} : Failed after {max_retries} attempts")
    return None


class ImageDataloader:
    def __init__(self, url_csv: Path, shuffle=False, nrows=None, start_index=0, end_index=None):
        logger.info("Read dataset")
        self.df = pd.read_csv(
            url_csv, names=["image_id", "url"], header=None, nrows=nrows
        )
        self.df = self.df.dropna()
        if shuffle:
            logger.info("Shuffle images")
            self.df = self.df.sample(frac=1, random_state=10)

        # Apply the index range filter
        self.df = self.df.iloc[start_index:end_index]
        logger.info(f"Number of URLs: {len(self.df.index)}")

    def __len__(self):
        return len(self.df.index)

    def __iter__(self):
        for image_id, url in zip(self.df["image_id"].values, self.df["url"].values):
            yield {"image_id": image_id, "url": url}


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--threads",
        type=int,
        default=8,  # 增加线程数
        help="Number of concurrent download threads (建议不超过8)"
    )
    args.add_argument(
        "--output",
        type=Path,
        default=Path("D:/mp16"),
        help="Output directory where images are stored",
    )
    args.add_argument(
        "--url_csv",
        type=Path,
        default=Path("resources/mp16_urls.csv"),
        help="CSV with Flickr image id and URL for downloading",
    )
    args.add_argument(
        "--size",
        type=int,
        default=320,
        help="Rescale image to a minimum edge size of SIZE",
    )
    args.add_argument("--nrows", type=int)
    args.add_argument(
        "--shuffle", action="store_true", help="Shuffle list of URLs before downloading"
    )
    args.add_argument(
        "--start_index", type=int, default=start_index, help="Index to start downloading images"
    )
    args.add_argument(
        "--end_index", type=int, default=end_index, help="Index to stop downloading images"
    )
    return args.parse_args()


def main():
    image_loader = ImageDataloader(
        args.url_csv, nrows=args.nrows, shuffle=args.shuffle, 
        start_index=args.start_index, end_index=args.end_index
    )

    download_dir = args.output / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading images...")
    with Pool(args.threads) as p:
        # 使用imap_unordered加快处理速度
        downloaded_images = list(
            p.imap_unordered(
                partial(download_image, output_dir=download_dir), 
                image_loader,
                chunksize=2  # 减少任务分块大小
            )
        )

    downloaded_images = [x for x in downloaded_images if x is not None]
    logger.info(f"Successfully downloaded {len(downloaded_images)} images")


logger = logging.getLogger("ImageDownloader")
if __name__ == "__main__":
    args = parse_args()

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(args.output / "writer.log"))
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
