import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
        for i in self.rec_rs[0:self.top_k]:
            try:
                rec_title = self.df.iloc[i[0]]["item_name"]
                rec_avg_rating = self.df.iloc[i[0]]["item_avg_rating"]
                rec_genre = self.df.iloc[i[0]]["item_category"]

                # divide the page into 2 columns
                col1, col2 = st.columns(2)

                # print the attributes on the page
                with col1:
                    st.write(rec_title)
                with col2:
                    st.write(rec_avg_rating)
                    st.write(rec_genre)
            except:
                continue