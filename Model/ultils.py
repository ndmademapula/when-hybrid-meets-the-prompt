import pandas as pd
import numpy as np
import streamlit as st
import webbrowser
df_info = pd.read_csv('Data/courses_list.csv')
df_courses = pd.read_csv('Data/courses.csv')

def get_categories():
    categories = []
    categories = df_courses["item_category"].value_counts().index.to_list()

    # for i in df_courses.index:
    #     category = df_courses.loc[i,'item_category']
    #     for j in range(len(category)):
    #         categories.append(category[j].strip())
    # categories = pd.Series(categories).value_counts().index.to_list()
    categories.insert(0,"All")
    return categories 

def print_rec(top_k,df):
    id_to_link = df_info.set_index('item_id')['item_urls'].to_dict()
    id_to_img = df_info.set_index('item_id')['item_imgs'].to_dict()

    for i in range(top_k):
        rec_id = df.iloc[i]['item_id']
        rec_link = id_to_link.get(rec_id, 'Link not found')
        rec_img = id_to_img.get(rec_id, "Image not found")

        rec_title = df.iloc[i]["item_name"]
        rec_avg_rating = df.iloc[i]["item_avg_rating"]
        rec_category = df.iloc[i]["item_category"]

        # divide the page into 2 columns
        col1, col2 = st.columns(2)

        # print the attributes on the page
        with col1:
            st.image(rec_img)
        with col2:
            st.markdown(f'<p style = "font-size: 20px; font-weight:bold;">{rec_title}</p>', unsafe_allow_html=True)
            st.write(rec_avg_rating)
            st.write(rec_category)
            st.markdown(f'<a href="{rec_link}" style="display: inline-block; padding: 10px 10px; background-color: red; color: white; text-align: center; text-decoration: none; font-size: 10px; font-weight: bold; border-radius: 4px;"> more information</a>',
                        unsafe_allow_html=True)
        st.divider()

def print_list(top_k,df):
    id_to_link = df_info.set_index('item_id')['item_urls'].to_dict()
    id_to_img = df_info.set_index('item_id')['item_imgs'].to_dict()

    for i in range(top_k):
        rec_id = df.iloc[i]['item_id']
        rec_link = id_to_link.get(rec_id, 'Link not found')
        rec_img = id_to_img.get(rec_id, "Image not found")

        rec_title = df.iloc[i]["item_name"]
        rec_avg_rating = df.iloc[i]["item_avg_rating"]
        rec_category = df.iloc[i]["item_category"]

        # divide the page into 2 columns
        col1, col2 = st.columns(2)

        # print the attributes on the page
        with col1:
            st.markdown(f'<p style = "font-size: 16px; font-weight:bold;">{rec_title}</p>', unsafe_allow_html=True)
            st.image(rec_img)
        with col2:
            st.write(rec_avg_rating)
            st.write(rec_category)
            st.markdown(f'<a href="{rec_link}" style="display: inline-block; padding: 10px 10px; background-color: red; color: white; text-align: center; text-decoration: none; font-size: 10px; font-weight: bold; border-radius: 4px;"> more information</a>',
                        unsafe_allow_html=True)
        st.divider()