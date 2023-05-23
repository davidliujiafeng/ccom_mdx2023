import os, shutil
from collections import Counter
import torch
import numpy as np
import json, sys
import torchaudio
from tqdm import tqdm

def cal_volumn(audio_tensor):
    avg_amplitude = torch.sum(torch.abs(audio_tensor))
    volume_db = 20 * torch.log10(avg_amplitude)
    return volume_db.item()
def cal_volumn_saved(filepath, volumns_restore):
    if filepath in volumns_restore:
        return volumns_restore[filepath]

    audio_tensor, _ = torchaudio.load(filepath)
    ret = cal_volumn(audio_tensor)
    volumns_restore[filepath] = ret
    return ret

if __name__ == '__main__':

    # model = 'pretrain3'
    # key = 6
    # speed = 200
    # volumn_threshold = 20

    model = sys.argv[1]
    key = int(sys.argv[2])
    speed = int(sys.argv[3])
    volumn_threshold = int(sys.argv[4])


    suffix = f'{model}_key{key}_speed{speed}'
    candidates_folder = f'evaluator_outputs_{suffix}/'
    pred_res_json = f'labelnoise_track_clean_predicted_by_{suffix}.json'
    volumns_save_path = f'volumns_{suffix}.json'
    audio_path = f'../labelnoise_clean_dataset_predicted_by_{suffix}'

    files = []
    for source in ['drums', 'vocals', 'other', 'bass']:
        files += [f'{x}_{source}' for x in os.listdir(candidates_folder + source) if not x.startswith('.')]

    volumns_restore = {}
    if os.path.exists(volumns_save_path):
        with open(volumns_save_path) as f:
            volumns_restore = json.load(f)



    all_clean_tracks = dict()
    for file in tqdm(files[:]):
        
        filename = file.split('_')[0]
        input_source = file.split('_')[1]

        empty_threshold_volumn = -80
        try:
            volumns_other_tracks = []
            empty = True
            for source in ['drums', 'vocals', 'other', 'bass']:
                output_song_path = f'{candidates_folder}{input_source}/{filename}/{source}.wav'
                volumn = cal_volumn_saved(output_song_path, volumns_restore)
                if volumn > empty_threshold_volumn:
                    empty = False
                if source != input_source:
                    volumns_other_tracks.append((volumn, source))
                else:
                    volumns_main = volumn
            if empty:
                continue

            volumns_other_tracks.sort(reverse=True)
            if volumns_main - volumns_other_tracks[0][0] > volumn_threshold:
                # we predict this one clean.
                all_clean_tracks.setdefault(input_source, []).append(filename)  

        except Exception as e:
            print(e)
            continue

    with open(volumns_save_path, 'w') as f:
        json.dump(volumns_restore, f)

    with open(pred_res_json, 'w') as f:
        json.dump(all_clean_tracks, f)


    for source in all_clean_tracks:
        original_path = '../moisesdb23_labelnoise_v1.0'
        dest_path = f'{audio_path}/train/{source}'
        os.makedirs(dest_path, exist_ok=True)
        for file in all_clean_tracks[source]:
            src_wav = f'{original_path}/{file}/{source}.wav'
            dest_wav = f'{dest_path}/{file}.wav'
            shutil.copy(src_wav, dest_wav)
            print(f'copied from {src_wav} to {dest_wav}')