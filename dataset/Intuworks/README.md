## 1. json파일 만드는 법
1. Intuworks 데이터 셋 중 `dataset6_myeon`을 다운받아 압축을 풀어줍니다.
2. `python makeJSon.py`를 실행하면 `train.json` 파일이 생성됩니다.

<br>
<br>

## 2. visualization.ipynb
1. 원하는 이미지로 `img_name`, `img_path`, `label_path`를 수정합니다.
2. `font`가 없다면 가지고 있는 ttf 파일로 `font`를 설정해줍니다.
3. 코드를 실행시키면 원하는 이미지의 bbox를 확인할 수 있습니다.

<br>
<br>

## 3. inference.ipynb
1. `config_file`. `ckpt_file`, `test_files`을 자신의 경로에 맞게 수정해줍니다.

<br>
<br>

## 4. mmdetection
1. 아직 `val.json` 파일을 따로 생성하지 않아서 `python tools/train.py` 를 실행시킬 때 `--no-validate` 옵션을 추가했습니다.

<br>
<br>