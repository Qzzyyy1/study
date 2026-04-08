import os
import urllib
import tqdm

def gen_bar_updater():
    pbar = tqdm.tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def download_url(url, path):
    urllib.request.urlretrieve(url, path, reporthook=gen_bar_updater())

def download(args, info):
    dataset_path = os.path.join('./datasets', info.path)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for url in info.url:
        filename = url.split('/')[-1]
        file_path = os.path.join(dataset_path, filename)
        if not os.path.exists(file_path):
            print(f'Download {url}')
            download_url(url, file_path)