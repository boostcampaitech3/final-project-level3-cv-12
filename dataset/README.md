## 1. json파일 만드는 법
1. Training, Validation 폴더 안에 zip파일을 각각 넣고 unzip.py로 압축을 푼다(unzip.py안의 경로 확인)
2. `python makejson.py`를 실행한다.

<br>
<br>

## 2. EDA.ipynb 사용시 한글 폰트 설치하는 법
1. EDA.ipynb 상단에 있는 
```
import matplotlib
matplotlib.__file__
```
를 실행후 matplotlib이 설치된 경로를 확인한다.

2. `install_font.sh`에 들어가서 7번째 line의 두번째 path를 본인 상황에 맞게 수정해준다
3. `sh install_font.sh`를 실행한다.
4. EDA.ipynb를 restart 후 사용한다.

<br>
<br>


## 3. 이미지 resize하는법
1. `python resize.py`를 실행하시면 `reTraining`, `reValidation` 디렉토리에 `resize`된 크기로 이미지가 생성됩니다.
2. `json`파일은 `retrain.json`, `reval.json`이라는 이름으로 생성됩니다.
3. 기본 크기는 `weight=1024, height=1024`로 설정해두었습니다. main부분에서 바꾸실 수 있습니다.