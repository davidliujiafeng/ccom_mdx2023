import os
import torchaudio
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor


def create_metadata(root_dir, usage="train"):

    meta_data = {}

    with ThreadPoolExecutor(16) as pool:
        temp = []
        for subdir, dirs, files in os.walk(root_dir):
            if subdir.split("/")[-1].startswith('.') or dirs or subdir == root_dir:
                continue
            source = subdir.split("/")[-1]
            meta_data[source] = {}

            for file in files:
                file = subdir + '/' + file
                temp.append((source, pool.submit(_track_meta, file)))

        for source, info in tqdm.tqdm(temp, ncols=128):
            file_name, length, sr = info.result()
            meta_data[source][file_name] = {"length": length, "sr": sr}

    json.dump(meta_data, open(f"metadata/lbn_clean_{usage}.json", "w"))


def _track_meta(file_path):
    file_name = file_path.split("/")[-1]

    try:
        info = torchaudio.info(str(file_path))
    except RuntimeError:
        print(file_path)
        raise

    return file_name, info.num_frames, info.sample_rate


if __name__ == "__main__":

    train_root = '../labelnoise_clean_dataset_predicted_by_pretrain3_key6_speed200/train'
    create_metadata(train_root, usage="train")