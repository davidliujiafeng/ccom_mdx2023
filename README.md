# ccom_mdx2023

This is the repository of team CCOM for Sound Demixing Challenge 2023. For Music Demixing Track, We get rank #1 in the Leaderboard A and rank #4 in the Leaderboard B.

## Requierments

- For inference, see `apt.txt` and `requirement.txt` in the `labelnoise_submission` branches
- For labelnoise training, you need to clone [DEMUCS](https://github.com/facebookresearch/demucs) and install all its dependencies, with also packages in `apt.txt` and `requirement.txt` needed.

## Methods
- For Leaderboard A, we used loss truncation in pre-training phase and then we filtered the dataset automatically with the pretrained loss truncation model. For further details please see `README.md` in the `labelnoise_train` branch.
- For Leaderboard B, we designed a multi-spectrogram U-Net model. Details are shown in he `bleeding_train` branch.

## Results
### Leaderboard A

- Submission ID: 219610

- Submitter: dramatic

- Final rank: 1st place

- Final scores:

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- | --------- | ---------- |
  | 7.455    | 8.117    | 7.993     | 5.342     | 8.369      |



### Leaderboard B

- Submission ID: 217439
- Submitter: liujiafeng
- Final rank: 4th place
- Final scores:

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- | --------- | ---------- |
  | 6.203    | 6.337    | 6.323     | 4.284     | 7.867      |

