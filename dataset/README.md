## json파일 만드는 법
1. Training, Validation 폴더 안에 zip파일을 각각 넣고 unzip.py로 압축을 푼다(unzip.py안의 경로 확인)
2. `python makejson.py`를 실행한다.

## EDA.ipynb 사용시 한글 폰트 설치하는 법
1. EDA.ipynb 상단에 있는 
```
import matplotlib
matplotlib.__file__
```
를 실행후 matplotlib이 설치된 경로를 확인한다.

2. `install_font.sh`에 들어가서 7번째 line의 두번째 path를 본인 상황에 맞게 수정해준다
3. `sh install_font.sh`를 실행한다.
4. EDA.ipynb를 restart 후 사용한다.