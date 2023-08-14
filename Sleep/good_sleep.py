import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

curr_filepath = os.getcwd()
data_filepath = os.path.join(curr_filepath, 'data')

df = pd.DataFrame()

for filename in os.listdir(data_filepath):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_filepath, filename)
        data = pd.read_csv(file_path,delimiter=';')
        df = pd.concat([df, data], ignore_index=True)

