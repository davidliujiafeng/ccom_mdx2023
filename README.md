# ccom_mdx2023
This is the code for the final submission to leaderboard A of Music Demixing Challenge 2023.
## Grade
- Submission ID: 219610
- Submitter: dramatic
- Final rank: 1st place
- Final scores:

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- | --------- | ---------- |
  | 7.455    | 8.117    | 7.993     | 5.342     | 8.369      |
## How to Run
- Prepare the checkpoints.
    - Download the checkpoints from [Google Drive](https://drive.google.com/file/d/1nkemSMx6TjNc4sb6Qohc7Pclyr7bM_wc/view?usp=share_link) 
    - Unzip `ckpt_labelnoise_submission.tgz` to `checkpoints/`
- Prepare the test set.
    - Prepare your test set into `public_dataset/MUSDB18-7-WAV/test/`
- Run `evaluate_locally.py` 
    - The separated stems will be placed into `evaluator_outputs/`