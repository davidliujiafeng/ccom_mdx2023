import os, sys
from aicrowd_wrapper import AIcrowdWrapper
from user_config import MODEL_NAME
def evaluate(LocalEvalConfig, ins, key, speed):
    """
    Runs local evaluation for the model
    Final evaluation code is the same as the evaluator
    """
    datafolder = LocalEvalConfig.DATA_FOLDER

    inputsfolder = datafolder
    groundtruthfolder = datafolder

    preds_folder = LocalEvalConfig.OUTPUTS_FOLDER

    model = AIcrowdWrapper(predictions_dir=preds_folder, dataset_dir=datafolder)
    folder_names = os.listdir(datafolder)

    for fname in folder_names:
        if fname.startswith('.'): # exclude .gitignore
            continue
        model.separate_music_file(fname, ins, key, speed)


if __name__ == "__main__":
    # change the local config as needed
    ins = sys.argv[1]
    key = int(sys.argv[2])
    speed = int(sys.argv[3])
    class LocalEvalConfig:
        DATA_FOLDER = '../moisesdb23_labelnoise_v1.0/'
        OUTPUTS_FOLDER = f'evaluator_outputs_{MODEL_NAME}_key{key}_speed{speed}'

    outfolder=  LocalEvalConfig.OUTPUTS_FOLDER
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    
    evaluate(LocalEvalConfig, ins, key, speed)
