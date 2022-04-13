# import matplotlib
# matplotlib.__file__로 본인 matplotlib 경로 확인해고 
# 7 line 두번째 path 수정해야 정상작동함

apt-get install fonts-nanum*
fc-cache -fv
cp /usr/share/fonts/truetype/nanum/Nanum* /opt/conda/envs/project/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/
rm -rf ~/.cache/matplotlib/*