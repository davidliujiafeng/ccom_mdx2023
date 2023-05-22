import os
import torchaudio
import tqdm
import json


def create_metadata(root_dir, usage="train"):
    from concurrent.futures import ThreadPoolExecutor

    temp = []
    meta_data = {}

    with ThreadPoolExecutor(16) as pool:
        for subdir, dirs, files in os.walk(root_dir):
            if subdir.startswith('.') or dirs or subdir == root_dir:
                continue

            song_name = str(subdir.split("/")[-1])
            temp.append((song_name, pool.submit(_track_meta, subdir)))

        for name, info in tqdm.tqdm(temp, ncols=128):
            meta_data[name] = info.result()

    json.dump(meta_data, open(f"metadata/bleeding_{usage}.json", "w"))


def _track_meta(song_path):
    sources = ['drums.wav', 'vocals.wav', 'other.wav', 'bass.wav']
    track_length = None
    track_samplerate = None

    for source in sources:
        file_path = song_path + "/" + source
        try:
            info = torchaudio.info(str(file_path))
        except RuntimeError:
            print(file_path)
            raise
        length = info.num_frames
        if track_length is None:
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file_path}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file_path}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")

    return {"length": length, "samplerate": track_samplerate}


if __name__ == "__main__":

    train_root = '/datasets/moisesdb23_bleeding_v1.0'
    create_metadata(train_root, usage="train")





