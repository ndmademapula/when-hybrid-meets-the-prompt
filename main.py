import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
# from sentence_transformers import SentenceTransformer 

from Model.Hybrid.Rec_Hybrid import *
from Model.Filtering.Rec_Filtering import RecommenderFiltering
from Model.ultils import *
from Model.Prompt.prompt import RecommenderPrompt

df_courses = pd.read_csv('Data/courses.csv').dropna()
df_courses= df_courses.reset_index(drop=True)
df_rating = pd.read_csv("Data/ratings.csv")

top_k = 10
script_about = '''
This web app is a demo of the Recommender system Scientific research project

***when Hybrid meets the Prompt*** (๑•̀ㅂ•́)و✧
'''
script_appreciation = '''
Words fail to express how grateful and appreciative I am for the bombass opportunity to work with my amazing and admirable co-researcher mates - :red[**Group 5**], the greatest, coolest, and most honourable homies.☆･ﾟ*.✩°｡ ⋆⸜ ✮.'''

def run():
    #--------------------------------------------------------
    # Homepage
    st.set_page_config(
        page_title="When Hybrid meets the Prompt",
        layout="wide",
        page_icon="💚",
    )

    with st.sidebar:
        st.caption("How would you like to get recommendations?")
        rec_option = option_menu(
            menu_title=None,
            options=["Filter","Hybrid Model","Prompt"],
            orientation="vertical",
            icons=["filter", "cpu" , "chat-heart"]
        )
        
        about_tab, appreciation_tab = st.tabs(["About", "Appreciation"])
        with about_tab: st.markdown(script_about)
        with appreciation_tab: st.markdown(script_appreciation)

    maincol1, maincol2 = st.columns([2,1],gap="medium")
    with maincol1:
        match rec_option:
            
            case "Filter":
                category_tab, title_tab = st.tabs(["Filter by Category","Filter by Title"])
                with category_tab:
                    slbCategory = st.selectbox("Category", get_categories())

                    col1, col2 = st.columns(2)
                    with col1:
                        slbSortOrder = st.radio("Sort Order", ["Descending","Ascending"])
                    with col2:
                        slbSortByMem = st.radio("Sort By", ["Rating", "Popular"])
                    btnType = st.button("Recommend by Category")
                    
                    sort_order = False if slbSortOrder == "Descending" else True
                    sort_by = False if slbSortByMem == "Average ratings" else True

                    if btnType:
                        st.balloons()
                        filterer = RecommenderFiltering(df=df_courses,
                                                        top_k=top_k, 
                                                        sort_order=sort_order, 
                                                        sort_by_mem=sort_by)
                        if slbCategory == "All":
                            rec_filter = filterer._sort_values(df_courses)
                        else:
                            filterer.keyword_search(item_categories=slbCategory)
                            rec_filter = filterer.rec_rs
                        try:
                            print_rec(top_k=top_k,df=rec_filter)
                        except: st.write("Can't find any recommendations for you")
                        
                with title_tab:   
                    inputTitle = st.text_input("Enter title")
                    btnTitle = st.button("Recommend by Title")

                    if btnTitle:
                        st.balloons()
                        filterer = RecommenderFiltering(df=df_courses,
                                                        top_k=top_k, 
                                                        sort_order=sort_order, 
                                                        sort_by_mem=sort_by)
                        filterer.keyword_search(item_title=inputTitle)
                        try:
                            rec_filter = filterer.rec_rs
                            print_rec(top_k=len(rec_filter),df=rec_filter)
                        except AttributeError:
                            st.write("Can't find any recommendations for you")                        

            case "Hybrid Model":
                st.warning("Sr sr (￣ε(#￣) Model still in testing...")

            case "Prompt":

                with st.chat_message("assistant", avatar="😎"):
                    st.write("How can i help you generate recommender tasks?")
                input_prompt = st.text_input("Describe your desired recommendations",value="",key=1)
                
                btnPrompt = st.button("Prompt Recommend")
                if btnPrompt:
                    st.balloons()
                    if input_prompt == "":
                        st.warning("Please describe your desired recommendations")
                    else:
                        vectors = joblib.load('Assets/courses_embeddings.pkl')
                        prompt = RecommenderPrompt(df=df_courses,
                                                top_k=top_k,
                                                vectors=vectors, 
                                                input_prompt=input_prompt)
                        with st.chat_message("assistant"):
                            st.write("Here is my recommendations for you")
                            prompt.recCourses()

    with maincol2:
        with st.container():
            st.caption('Most popular')
            print_list(top_k=5,df=df_courses.sort_values("item_members",ascending=False))
        st.divider()
        with st.container():
            st.caption("Most rated")
            print_list(top_k=5,df=df_courses.sort_values("item_avg_rating", ascending=False))

if __name__ == "__main__":
    run()