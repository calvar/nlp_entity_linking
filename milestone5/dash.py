import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import json

sns.set(font_scale=8)

#####Load data and process-----------------------------------------
clean_df = pd.read_csv("clean_df.csv")

with open("terms.json", "r") as fp:
    terms = json.load(fp)

In = open("topic.txt","r")
topic = In.readline()
In.close()
#####--------------------------------------------------------------

opts = clean_df["kp"].unique()
opts = np.insert(opts,0,"All")
select_box = st.sidebar.selectbox(label="Key point",
                                  options=opts
)
#st.write(select_box)

#Distribution of medical specialities by keypoint
st.write("""
### TOPIC: 
#### {0}
### Distribution of medical speciality by key point
#### Key point:\n{1}
""".format(topic, select_box))

fig = plt.figure(figsize=(50,40))

if select_box == "All":
    sns.countplot(
        y="medical_specialty_new",
        data=clean_df,
        order=clean_df["medical_specialty_new"].value_counts().index
    )
else:
    sns.countplot(
        y="medical_specialty_new",
        data=clean_df[clean_df["kp"] == select_box],
        order=clean_df["medical_specialty_new"].value_counts().index
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
