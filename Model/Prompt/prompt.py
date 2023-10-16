import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df_info = pd.read_csv("Data/courses_list.csv", index_col=0)

class RecommenderPrompt:

    def __init__(self, top_k, df, vectors, input_prompt):
        self.df = df
        self.top_k = top_k
        self.vectors = vectors
        self.input_prompt = input_prompt
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rec_rs = None

    def userInput(self):
        user_input = self.input_prompt
        input_embeddings = self.model.encode(user_input)
        similarity = cosine_similarity(input_embeddings.reshape(1,input_embeddings.shape[0]),self.vectors)
        return similarity

    def recCourses(self):
        similarity_score = self.userInput()
        # sorting by similarity score
        if self.rec_rs is not np.nan:
            self.rec_rs = sorted(list(enumerate(similarity_score[0])), reverse=True, key=lambda x: x[1])
        self.return_recommend()
        
    def return_recommend(self):
        id_to_link = df_info.set_index('item_id')['item_urls'].to_dict()
        id_to_img = df_info.set_index('item_id')['item_imgs'].to_dict()
        for i in self.rec_rs[0:self.top_k]:
            try:
                rec_id = self.df.iloc[i[0]]['item_id']
                rec_link = id_to_link.get(rec_id, 'Link not found')
                rec_img = id_to_img.get(rec_id, "Image not found")

                rec_title = self.df.iloc[i[0]]["item_name"]
                rec_avg_rating = self.df.iloc[i[0]]["item_avg_rating"]
                rec_genre = self.df.iloc[i[0]]["item_category"]

                # divide the page into 2 columns
                col1, col2 = st.columns(2)

                # print the attributes on the page
                with col1:
                    st.markdown(f'<p style = "font-size: 16px; font-weight:bold;">{rec_title}</p>', unsafe_allow_html=True)
                    st.image(rec_img)
                with col2:
                    st.write(rec_avg_rating)
                    st.write(rec_genre)
                    st.markdown(f'<a href="{rec_link}" style="display: inline-block; padding: 10px 10px; background-color: red; color: white; text-align: center; text-decoration: none; font-size: 10px; font-weight: bold; border-radius: 4px;"> more information</a>',
                        unsafe_allow_html=True)
                st.divider()
            except:
                continue