import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
# from sentence_transformers import SentenceTransformer 

from Model.Hybrid.Rec_Filtering import RecommenderFiltering
from Model.ultils import *
from Model.Prompt.prompt import RecommenderPrompt

df_courses = pd.read_csv('Data/courses.csv')
top_k = 10
script_about = '''
This web app is a demo of the Recommender system Scientific research project
'''
script_appreciation = '''
Words fail to express how grateful and appreciative I am for the bombass opportunity to work with my amazing and admirable co-researcher mates - :red[**Group 5**], the greatest, coolest, and most honourable homies.☆･ﾟ*.✩°｡ ⋆⸜ ✮.'''
#page config
st.set_page_config(
    page_title="Group 5 Scientific Research",
    layout="wide",
    menu_items={
        'About': script_about
    }
)

with st.sidebar:
    about_tab = st.tabs(["About"])
    st.markdown('''
                    # :red[E-learning Recommender System]
                    *When Hybrid meets the Prompt*''')
    st.markdown(script_about)
    appreciation_tab = st.tabs(["Appreciation"])
    st.markdown(script_appreciation)



maincol1, maincol2 = st.columns([2,1],gap="medium")
with maincol1:
    st.caption("Choose how would you like to get recommended")
    rec_optio = option_menu(
        menu_title=None,
        options=["Filter","RecSys Model","Prompt"],
        orientation="horizontal",
    )

    match rec_optio:
        case "Filter":
            category_tab, title_tab = st.tabs(["Filter by Level","Filter by Title"])
            with category_tab:
                slbCategory = st.selectbox("Level",  get_categories())

                col1, col2 = st.columns(2)
                with col1:
                    slbSortOrder = st.radio("Sort Order", ["Descending","Ascending"])
                with col2:
                    #? fixed sorted by popular!!!
                    slbSortByMem = st.radio("Sort By", ["Rating"])
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

        case "RecSys Model":
            st.warning("Sr sr (￣ε(#￣) Model still in testing...")
        case "Prompt":
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.write("hawo hawo 🫶🏻 needs some recommendations? i'm here to please u ✨✨✨")
                st.write("pls enter prompt below 👇🏻👇🏻👇🏻")
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
    # with st.container():
    #     st.caption('Most popular')
    #     print_rec(top_k=5,df=df_courses.sort_values("item_members",ascending=False))
    # st.divider()
    with st.container():
        st.caption("Most rated")
        print_rec(top_k=5,df=df_courses.sort_values("item_avg_rating", ascending=False))