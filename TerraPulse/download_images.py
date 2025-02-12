import requests
import pandas as pd
from PIL import ImageFile, Image

from io import BytesIO
from pathlib import Path
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# 请求头配置
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
]

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 下载图片函数
def download_image(image_id, url, output_dir, count, thread_id, session):
    save_path = output_dir / f"{image_id.replace('/', '_')}.jpg"

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    retry_count = 0
    max_retries = 3
    retry_delay = 3 + random.uniform(0.5, 3)

    while retry_count < max_retries:
        try:
            time.sleep(random.uniform(1, 3))

            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            if image.mode != "RGB":
                image = image.convert("RGB")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path, "JPEG")
            
            count += 1
            tqdm.write(f"Thread {thread_id} - Downloaded {count}: {image_id}")
            return count

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                tqdm.write(f"Thread {thread_id} - Rate limit hit for {image_id}, retrying...")
                retry_count += 1
                time.sleep(retry_delay)
                retry_delay *= 2 + random.uniform(0.5, 3)
            else:
                tqdm.write(f"Thread {thread_id} - HTTP Error {response.status_code} for {image_id}: {str(e)}")
                return count
        except Exception as e:
            tqdm.write(f"Thread {thread_id} - {image_id} : {type(e).__name__} - {str(e)}")
            return count

    tqdm.write(f"Thread {thread_id} - Max retries reached for {image_id}, skipping...")
    return count

# 读取CSV文件并返回指定范围的数据
def load_image_data(url_csv: Path, start_index=0, end_index=None):
    df = pd.read_csv(url_csv, names=["image_id", "url"], header=None)
    df = df.dropna()
    return df.iloc[start_index:end_index]

# 分割数据给每个线程
def split_data(image_data, num_threads):
    avg_size = len(image_data) // num_threads
    data_splits = []
    for i in range(num_threads):
        start = i * avg_size
        end = (i + 1) * avg_size if i < num_threads - 1 else len(image_data)
        data_splits.append(image_data.iloc[start:end])
    return data_splits

# 线程下载函数
def thread_download(data_chunk, output_dir, start_index, thread_id, total_count, session):
    count = 0
    for index, row in data_chunk.iterrows():
        image_id, url = row["image_id"], row["url"]
        count = download_image(image_id, url, output_dir, count, thread_id, session)
        total_count[0] += 1
        with total_count_lock:
            progress_bar.set_description(f"Progress: {total_count[0]}/{len(image_data)}")
            progress_bar.update(1)
    return count

# 主函数
def main():
    start_index = 100000
    end_index = 120000
    url_csv = "resources/mp16_urls.csv"
    output_dir = Path("D:/mp16/downloads")
    output_dir.mkdir(parents=True, exist_ok=True)

    global image_data
    image_data = load_image_data(url_csv, start_index=start_index, end_index=end_index)

    num_threads = 16

    global total_count_lock
    total_count_lock = threading.Lock()
    total_count = [0]

    global progress_bar
    progress_bar = tqdm(total=len(image_data), desc="Progress", ncols=100)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, data_chunk in enumerate(split_data(image_data, num_threads)):
            session = requests.Session()
            futures.append(executor.submit(thread_download, data_chunk, output_dir, start_index + i * len(data_chunk), i + 1, total_count, session))

        for future in as_completed(futures):
            future.result()

    progress_bar.close()
    tqdm.write("Download completed.")

if __name__ == "__main__":
    import threading
    main()