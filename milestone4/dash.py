import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import json

sns.set(font_scale=8)

#####Load data and process-----------------------------------------
df_clean = pd.read_csv("df_clean.csv")

with open("terms.json", "r") as fp:
    terms = json.load(fp)
#####--------------------------------------------------------------

opts = df_clean["kp"].unique()
opts = np.insert(opts,0,"All")
select_box = st.sidebar.selectbox(label="Key point",
                                  options=opts
)
#st.write(select_box)

#Distribution of medical specialities by keypoint
st.write("""
### Distribution of medical speciality by key point
#### Key point:\n{0}
""".format(select_box))

fig = plt.figure(figsize=(50,40))

if select_box == "All":
    sns.countplot(
        y="medical_specialty_new",
        data=df_clean,
        order=df_clean["medical_specialty_new"].value_counts().index
    )
else:
    sns.countplot(
        y="medical_specialty_new",
        data=df_clean[df_clean["kp"] == select_box],
        order=df_clean["medical_specialty_new"].value_counts().index
    ) 
    
st.pyplot(fig)

#Top 10 wikipedia terms for the key point
if select_box != "All":
    st.write("""
    #### Top 10 wikipedia terms related to this keypoint
    """)
    
    dc = dict(sorted(terms[select_box].items(), key= lambda x: x[1], reverse=True)[:10])
    for t in dc:
        st.write(t)
else:
    st.write("""
    Select a particular key point to see the top 10 wikipedia terms related to it.""")
