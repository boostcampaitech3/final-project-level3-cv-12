import random 
import pandas as pd
import streamlit as st


def write_info(num, slot, label=None):
    label = '##### '+str(num)
    if num == 2:
        slot.success(label)
    elif num == 1 or num == 0:
        slot.info(label)
    elif num == -1:
        slot.warning(label)
    elif num < -1:
        slot.error(label)
    else:
        slot.write(label)


def show_table():
    status_df = pd.read_csv('temp.csv')
    for idx, row in status_df.iterrows():
        lines[2 * idx].write('------------')
        col1, col2, col3 = lines[2 * idx + 1].columns((1, 3, 1))

        col1.write('#### ' + str(idx)) 
        col2.write('#### ' + row['Product'])  
        write_info(row['Status'] + random.randint(-1, 2), col3) ## 나중에 이 부분에서 random 부분 없애주시면됩니다.

# table format
# status_df = pd.DataFrame([
#     ['진라면', -1],
#     ['신라면', -1],
#     ['짜파게티', -1],
#     ['생생우동', -1],
#     ['불닭볶음면', -1],
# ], 
# columns=['Product', 'Status']
# )

_, col2 = st.columns((6, 1))
refresh_button = col2.button('Refresh')
title = st.empty()
colms = title.columns((1, 3, 1))
fields = ["Idx", 'Product', 'Status']

for col, field_name in zip(colms, fields):
    col.write('#### ' + field_name)


status_df = pd.read_csv('temp.csv')
lines = [st.empty() for _ in range(len(status_df) * 2 )]
show_table()

if refresh_button:
    show_table()
