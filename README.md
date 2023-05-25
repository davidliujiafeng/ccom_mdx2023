# ccom_mdx2023

This branch will show how we use our pertained loss-truncation model `pretrain3` in the previous stage (presented in the branch `main`) to filter labelnoise dataset automatically and further training procedure:

1. Automatically filter the dataset with `pretrain3`

2. Train the 4 sources simultaneously for 360 epochs

3. Filter the dataset again with the new model `filtered360`

4. Fine-tune 4 single-source models

## Grade

### Before Step 1

- Submission ID: 216571

- Submitter: zhangxr_ccom

- Final Scores:

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- | --------- | ---------- |
  | 6.277    | 6.943    | 6.624     | 4.447     | 7.094      |

### After Step 2

- Submission ID: 217259

- Submitter: dramatic

- Final Scores:

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- | --------- | ---------- |
  | 6.884    | 7.339    | 7.581     | 4.875     | 7.741      |

### After Step 4

- Submission ID: 219610

- Submitter: dramatic

- Final rank: 1st place

- Final scores:

  | Mean SDR | SDR Bass | SDR Drums | SDR Other | SDR Vocals |
  | -------- | -------- | --------- | --------- | ---------- |
  | 7.455    | 8.117    | 7.993     | 5.342     | 8.369      |

## Pratical Steps

### prepare ckpt and data

- Download `ckpt_labelnoise_autofilter.tgz` from [Google Drive](https://drive.google.com/file/d/1GJ84D7vQk2bCBO0r4e6yIe9mLwaFXiPX/view?usp=share_link) and unzip it to `labelnoise_filter/checkpoints/` 
- Download [sdxdb23_labelnoise_v1.0_rc1.zip](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/dataset_files?unique_download_uri=246719&challenge_id=1108) and unzip it to `moisesdb23_labelnoise_v1.0/`

### labelnoise filter

```bash
cd labelnoise_filter
./filter.sh
```

This script will predict parallel with 4 gpus. After the procedure is finished, Run:

```bash
python auto_filter.py pretrain3 6 200 20
```

This script will save the clean tracks that the model predicts into `labelnoise_filter/labelnoise_track_clean_predicted_by_pretrain3_key6_speed200.json` and pick up all clean tracks into `labelnoise_clean_dataset_predicted_by_pretrain3_key6_speed200` folder.

### train 4 sources simultaneously

```bash
cd ../train_on_filtered_stems
python create_metadata.py
python trainer.py
```

The `trainer.py` script will train on 8 GPUs. The memory cost will be ~20GB. We terminate the training process after 360 epochs.

### labelnoise filter again

- change to `labelnoise_filter` directory
- modify `user_config.py` as follows:

```python
# from first_loss_trunc_filter import LTDemucsWrapper
# MySeparationModel = LTDemucsWrapper
# MODEL_NAME = 'pretrain3'

from second_filter_epo360 import MyDemucsWrapper
MySeparationModel = MyDemucsWrapper
MODEL_NAME = 'filtered360'
```

- Run `filter.sh`
- After the 4 process terminates, run `python auto_filter.py filtered360 6 200 20`

### fine-tune single-source models

#### metadata building

First change to `train_on_filtered_stems` directory and modify `create_metadata.py` in line 45 as follows:

```python
train_root = '../labelnoise_clean_dataset_predicted_by_filtered360_key6_speed200/train'
create_metadata(train_root, usage="train")
```

Then run `python create_metadata.py`

#### checkpoint restore

modify `trainer.py` in line 25 as follows:

```python
trainer.fit(model, datamodule=dm, ckpt_path='./lightning_logs/version_0/checkpoints/last.ckpt')
```

#### vocals

- modify `config/labelnoise.yaml`, change `weights` as follows:

```yaml
weights:
- 0.0
- 0.0
- 0.0
- 1.0
```

- run `python trainer.py` for 783 more epochs.
#### bass

We joint-trained bass and vocals.

- modify `config/labelnoise.yaml`, change `weights` as follows:

```yaml
weights:
- 0.0
- 1.0
- 0.0
- 1.0
```

- run `python trainer.py` for 644 more epochs.

#### drums

We joint-trained drums, bass and vocals.

- modify `config/labelnoise.yaml`, change `weights` as follows:

```yaml
weights:
- 0.9
- 0.05
- 0.0
- 0.05
```

- run `python trainer.py` for 84 more epochs.

#### other

Since the score of `other` is not good enough on the leaderboard, we assume that our predicted filtered dataset still has much noise within `other` stems.

So we use loss truncation again to fine-tune `other` source with 8 Nvidia A100-40GB GPUs:

1. use loss truncation to fine-tune 4 sources simultaneously.

   - modify `config/labelnoise.yaml` change `batch_size` to 4

   - modify `module.py` in line 37 as follows:

     ```python
     loss = F.l1_loss(estimate, sources, reduction='none')
     # loss = loss.mean(dims).mean(0)
     loss = loss.mean(dims)
     loss = loss.masked_fill(loss > loss.quantile(0.9, dim=0), 0)
     loss = loss.mean(0)
     weights = torch.tensor(self.args.weights, device=sources.device)
     ```

   - run `python trainer.py` for 324 more epochs

2. use loss truncation to fine-tune `other` stem only.

	- modify `config/labelnoise.yaml`, change `weights` as follows:

	```yaml
	weights:
	- 0.0
	- 0.0
	- 1.0
	- 0.0
	```
	- run `python trainer.py` for 196 more epochs
