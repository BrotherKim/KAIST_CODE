# Google Colab 환경에서 개발하는 방법

## Connect to Google drive

- Colab 의 경우에는, 한 세션 최대 이용 시간이 12시간으로 제한되어 있음
- 12시간 이상의 학습을 하기 위해서라면 이전에 저장해둔 checkpoint 를 불러와야 할 필요가 있음

```python
from google.colab import drive

drive.mount('/content/gdrive')
```

## Add ipython file to your directory

- `train.ipynb` 와 `decode.ipynb` 실습 코드 파일을 작업할 경로에 추가하기

## Append the Environment path

- 실습 코드(모듈)를 저장해둔 경로를 사용하기 위함

```python
import os

os.sys.path.append('/content/gdrive/path/to/module_dir')
```

## Continue training from stored checkpoint

- Colab 의 세션이 중단되었을 시에, 이전에 저장해둔 checkpoint 를 불러와 학습을 재개하기 위함
- 추가된 코드
    - `args['continue_training']`: True 로 설정되어있을시에 학습 재개

```python
if args['continue_training']:
	model = NMT.load(args['model_path'])
else:
	model = NMT(embed_size=int(args['embed_size']),
              hidden_size=int(args['hidden_size']),
              dropout_rate=float(args['dropout']),
              vocab=vocab)
```

```python
if args['continue_training']:
	print('restore parameters of the optimizers {}'.format(args['model_path'] + '.optim'), file=sys.stderr)
  optimizer.load_state_dict(torch.load(args['model_path'] + '.optim'))
```

## Building Vocabulary

- 아래 명령어를 colab 환경에서 실행

```python
!python3 vocab.py --train-src=/path/to/data_dir/train.de-en.de.wmixerprep --train-tgt=/path/to/data_dir/train.de-en.en.wmixerprep /path/to/data_dir/vocab.json
```

## Training & Decoding

- `train.ipynb` 코드 실행 *→ NMT training*
- `decode.ipynb` 코드 실행 *→ NMT decoding*

## Announcement

- `train.ipynb` 코드에 training iteration 을 break 시킨 부분을 지우고 진행하시면 됩니다

## 추가된 부분

- `train.ipynb`
    - colab 환경에서 런타임 세션이 종료되었을 시에, 모델의 학습을 재개하기 위해서 추가되었습니다.
    - 해당 코드는 checkpoint 를 저장한 경로에서 가장 최근 checkpoint 파일을 불러옵니다.

```python
# checkpoint 를 저장한 directory 의 최근 checkpoint 를 불러옴
# e.g., .bin, .optim
ckpt_file_time = []
optim_file_time = []
for f_name in os.listdir(model_save_path):
  if '.bin' not in f_name:
    continue

  written_time = os.path.getctime(f"{model_save_path}/{f_name}")

  if f_name.split('.')[-1] == 'optim':
    optim_file_time.append((f_name, written_time))
  else:
    ckpt_file_time.append((f_name, written_time))

sorted_ckpt = sorted(ckpt_file_time, key=lambda x: x[1], reverse=True)
sorted_optim = sorted(optim_file_time, key=lambda x: x[1], reverse=True)

recent_ckpt = sorted_ckpt[0][0]
recent_optim = sorted_optim[0][0]
```

- 해당 코드는 이전에 세션이 종료되었을 시에, 진행중이었던 학습 epoch 및 training iteration 에서 학습을 재개하기 위해 이전 epoch 및 training iteration 정보를 불러옵니다.

```python
last_train_prefixes = recent_ckpt.split('/')[-1].split('_')
last_train_epoch = int(last_train_prefixes[0])
last_train_iter = int(last_train_prefixes[1])

------------------------------------------------------------------

if args['continue_training']:
  train_iter = last_train_iter
  epoch = last_train_epoch - 1

------------------------------------------------------------------

if args['continue_training'] and idx < last_train_iter:
	continue
```

- `decode.ipynb`
    - 해당 파일에서는 변동된 부분이 없습니다.
