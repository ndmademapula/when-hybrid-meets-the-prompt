import streamlit as st
import pandas as pd

st.set_page_config(page_title="Plotting Demo", page_icon="📈")
st.markdown("# Plotting Demo")
st.sidebar.header("Data Collection")
st.write(
    """Hi biatch ヾ(•ω•`)o . Enjoy!"""
)

df_info = pd.read_csv('Data/courses_list.csv',index_col=0)
df_courses = pd.read_csv('Data/courses.csv',index_col=0)
df_ratings = pd.read_csv("Data/ratings.csv")

df_show_info = pd.read_csv('Data/courses_list.csv',nrows=50, index_col=0)
df_show_courses = pd.read_csv('Data/courses.csv',nrows=50, index_col=0)
df_show_ratings = pd.read_csv("Data/ratings.csv",nrows=50)

tab_ratings, tab_courses, tab_info = st.tabs(["ratings","courses list", "courses info"])
with tab_ratings:
    st.dataframe(df_show_ratings,use_container_width=True)
with tab_courses:
    st.dataframe(df_show_courses,use_container_width=True)
with tab_info:
    st.dataframe(df_show_info,use_container_width=True)

with st.form("recommend"):
    # Let the user select the user to investigate
    user = st.selectbox(
        "Select a customer to get his recommendations",
        df_courses.user_id.unique(),
    )

    items_to_recommend = st.slider("How many items to recommend?", 1, 10, 5)
    print(items_to_recommend)

    submitted = st.form_submit_button("Recommend!")