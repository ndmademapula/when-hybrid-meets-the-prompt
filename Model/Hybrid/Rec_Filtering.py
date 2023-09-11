import pandas as pd
import numpy as np

class RecommenderFiltering:
    # constructor
    def __init__(self, df, top_k = 5, sort_by_mem=False, sort_order = False):
        self.df = df
        self.top_k = top_k
        self.sort_by_mem = sort_by_mem
        self.sort_order = sort_order

        self.rec = self._sort_values(df=df)

    # internal funtion
    def _filter_categories(self):
        idx = []
        for i in self.rec.index: #search through index
            if self.rec.loc[i,"item_category"] is not np.nan:
                keyword_search = self.rec.loc[i, "item_category"].split(',')
                if self.item_categories.lower().strip() in str(keyword_search).lower().strip():
                    idx.append(i)
        self.rec = self.rec.loc[idx]
            
    # internal fucntion
    def _filter_title(self):
        idx = []
        for i in self.rec.index: #search through index
            if self.rec.loc[i,"item_name"] is not np.nan:
                keyword_search = self.rec.loc[i, "item_name"].split(",")
                if self.item_title.lower().strip() in str(keyword_search).lower().strip():
                    idx.append(i)
            else:
                return None
        self.rec = self.rec.loc[idx]
    
    def _sort_values(self, df):
        df_sorted = df.sort_values("item_avg_rating", ascending=self.sort_order)
    #     if self.sort_by_mem == True:
    #         df_sorted = df.sort_values("item_members", ascending=self.sort_order)
    #     else:
    #         df_sorted = df.sort_values("item_avg_rating", ascending=self.sort_order)
        return df_sorted
        
    def return_recommend(self):
        if len(self.rec) == 0:
            return 
        elif len(self.rec) > self.top_k:
            self.rec_rs = self.rec.iloc[:self.top_k].copy()
        else:
            self.rec_rs = self.rec[:].copy()  
        return self.rec_rs    

    # main fuction
    def keyword_search(self, item_categories=None, item_title=None):
        self.item_title = item_title
        self.item_categories = item_categories

        # filter by item_categories
        if self.item_categories != None:
            self._filter_categories()   
            if len(self.rec) == 0:
                print(f"No matching products found for {self.item_categories}")
                return None

        # filter by item_category
        if self.item_title != None:
            self._filter_title()
            if len(self.rec) == 0:
                print(f"No matching products found for {self.item_title}")
                return 
        
        self.rec = self._sort_values(self.rec)
        self.return_recommend()