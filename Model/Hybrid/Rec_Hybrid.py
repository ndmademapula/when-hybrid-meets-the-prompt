from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from typing import Dict, Any, Union, List
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

df_rating = pd.read_csv("Data/ratings.csv")

MODEL = {
    "lmf": LogisticMatrixFactorization,
    "als": AlternatingLeastSquares,
    "bpr": BayesianPersonalizedRanking,
}


def _get_sparse_matrix(values, user_idx, product_idx):
    return csr_matrix(
        (values, (user_idx, product_idx)),
        shape=(len(user_idx.unique()), len(product_idx.unique())),
    )


def _get_model(name: str, **params):
    model = MODEL.get(name)
    if model is None:
        raise ValueError("No model with name {}".format(name))
    return model(**params)

class InternalStatusError:
    pass


class Recommender:
    def __init__(
        self,
        values,
        user_idx,
        product_idx,
    ):
        self.user_product_matrix = _get_sparse_matrix(values, user_idx, product_idx)
        self.user_idx = user_idx
        self.product_idx = product_idx

        # This variable will be set during training phase
        self.model = joblib.load("Assets/ALS_model/pkl")
        self.fitted = False

    # def create_and_fit(
    #     self,
    #     model_name: str,
    #     weight_strategy: str = "bm25",
    #     model_params: Dict[str, Any] = {},
    # ):
    #     weight_strategy = weight_strategy.lower()
    #     if weight_strategy == "bm25":
    #         data = bm25_weight(
    #             self.user_product_matrix,
    #             K1=1.2,
    #             B=0.75,
    #         )
    #     elif weight_strategy == "balanced":
    #         # Balance the positive and negative (nan) entries
    #         # http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf
    #         total_size = (
    #             self.user_product_matrix.shape[0] * self.user_product_matrix.shape[1]
    #         )
    #         sum = self.user_product_matrix.sum()
    #         num_zeros = total_size - self.user_product_matrix.count_nonzero()
    #         data = self.user_product_matrix.multiply(num_zeros / sum)
    #     elif weight_strategy == "same":
    #         data = self.user_product_matrix
    #     else:
    #         raise ValueError("Weight strategy not supported")

    #     self.model = _get_model(model_name, **model_params)
    #     self.fitted = True

    #     self.model.fit(data)

    #     return self

    def recommend_products(
        self,
        user_id,
        items_to_recommend=5,
    ):
        """Finds the recommended items for the user.
        Returns:
            (items, scores) pair, where item is already the name of the suggested item.
        """

        if not self.fitted:
            raise InternalStatusError(
                "Cannot recommend products without previously fitting the model."
                " Please, consider fitting the model before recommening products."
            )

        return self.model.recommend(
            user_id,
            self.user_product_matrix[user_id],
            filter_already_liked_items=True,
            N=items_to_recommend,
        )

    def explain_recommendation(
        self,
        user_id,
        suggested_item_id,
        recommended_items,
    ):
        _, items_score_contrib, _ = self.model.explain(
            user_id,
            self.user_product_matrix,
            suggested_item_id,
            N=recommended_items,
        )

        return items_score_contrib

    def similar_users(self, user_id):
        return self.model.similar_users(user_id)

    @property
    def item_factors(self):
        return self.model.item_factors


def load_and_preprocess_data():
    df = df_rating

    # Remove nans values
    df = df.dropna()

    # Get unique entries in the dataset of users and products
    users = df["user_id"].unique()
    items = df["item_id"].unique()

    # Create a categorical type for users and product. User ordered to ensure
    # reproducibility
    user_cat = pd.CategoricalDtype(categories=sorted(users), ordered=True)
    item_cat = pd.CategoricalDtype(categories=sorted(items), ordered=True)

    # Transform and get the indexes of the columns
    user_idx = df["user_id"].astype(user_cat).cat.codes
    item_idx = df["item_id"].astype(item_cat).cat.codes

    # Add the categorical index to the starting dataframe
    df["UserIndex"] = user_idx
    df["ItemIndex"] = item_idx

    return df, user_idx, item_idx

def create_and_fit_recommender(
    model_name: str,
    values: Union[pd.DataFrame, "np.ndarray"],
    users: Union[pd.DataFrame, "np.ndarray"],
    products: Union[pd.DataFrame, "np.ndarray"],
) -> Recommender:
    recommender = Recommender(
        values,
        users,
        products,
    )

    recommender.create_and_fit(
        model_name,
        # Fine-tuned values
        model_params=dict(
            factors=190,
            alpha=0.6,
            regularization=0.06,
            random_state=42,
        ),
    )
    return recommender

def explain_recommendation(
    recommender: Recommender,
    user_id: int,
    suggestions: List[int],
    df: pd.DataFrame,
):
    output = []

    n_recommended = len(suggestions)
    for suggestion in suggestions:
        explained = recommender.explain_recommendation(
            user_id, suggestion, n_recommended
        )

        suggested_items_id = [id[0] for id in explained]

        suggested_description = (
            df.loc[df.ItemIndex == suggestion][["item_id", "ItemIndex"]]
            .drop_duplicates(subset=["ItemIndex"])["item_id"]
            .unique()[0]
        )
        similar_items_description = (
            df.loc[df["ItemIndex"].isin(suggested_items_id)][
                ["item_id", "ItemIndex"]
            ]
            .drop_duplicates(subset=["ItemIndex"])["item_id"]
            .unique()
        )

        output.append(
            f"The item **{suggested_description.strip()}** "
            "has been suggested because it is similar to the following products"
            " bought by the user:"
        )
        for description in similar_items_description:
            output.append(f"- {description.strip()}")

    with st.expander("See why the model recommended these products"):
        st.write("\n".join(output))

    st.write("------")


def print_suggestions(suggestions: List[int], df: pd.DataFrame):
    similar_items_description = (
        df.loc[df["ItemIndex"].isin(suggestions)][["item_id", "ItemIndex"]]
        .drop_duplicates(subset=["ItemIndex"])["item_id"]
        .unique()
    )

    output = ["The model suggests the following products:"]
    for description in similar_items_description:
        output.append(f"- {description.strip()}")

    st.write("\n".join(output))


def display_user_char(user: int, data: pd.DataFrame):
    subset = data[data.CustomerIndex == user]
    # products = subset.groupby("ItemIndex").agg(
    #     {"item_id": lambda x: x.iloc[0], "Quantity": sum}
    # )

    st.write(
        "The user {} bought {} distinct products. Here is the purchase history: ".format(
            user, subset["item_id"].nunique()
        )
    )
    st.dataframe(subset)
    st.write("-----")


def _extract_description(df, products):
    desc = df[df["ItemIndex"].isin(products)].drop_duplicates(
        "ItemIndex", ignore_index=True
    )[["ItemIndex", "item_id"]]
    return desc.set_index("ItemIndex")


def display_recommendation_plots(
    user_id: int,
    suggestions: List[int],
    df: pd.DataFrame,
    model: Recommender,
):
    """Plots a t-SNE with the suggested items, togheter with the purchases of
    similar users.
    """
    # Get the purchased items that contribute the most to the suggestions
    contributions = []
    n_recommended = len(suggestions)
    for suggestion in suggestions:
        items_and_score = model.explain_recommendation(
            user_id, suggestion, n_recommended
        )
        contributions.append([t[0] for t in items_and_score])

    contributions = np.unique(np.concatenate(contributions))

    print("Contribution computed")
    print(contributions)
    print("=" * 80)

    # Find the purchases of similar users
    bought_by_similar_users = []

    sim_users, _ = model.similar_users(user_id)

    for u in sim_users:
        _, sim_purchases = model.user_product_matrix[u].nonzero()
        bought_by_similar_users.append(sim_purchases)

    bought_by_similar_users = np.unique(np.concatenate(bought_by_similar_users))

    print("Similar bought computed")
    print(bought_by_similar_users)
    print("=" * 80)

    # Compute the t-sne

    # Concate all the vectors to compute a single time the decomposition
    to_decompose = np.concatenate(
        (
            model.item_factors[suggestions],
            model.item_factors[contributions],
            model.item_factors[bought_by_similar_users],
        )
    )

    print(f"Shape to decompose: {to_decompose.shape}")

    with st.spinner("Computing plots (this might take around 60 seconds)..."):
        elapsed = time.time()
        decomposed = _tsne_decomposition(
            to_decompose,
            dict(
                perplexity=30,
                metric="euclidean",
                n_iter=1_000,
                random_state=42,
            ),
        )
    elapsed = time.time() - elapsed
    print(f"TSNE computed in {elapsed}")
    print("=" * 80)

    # Extract the decomposed vectors
    suggestion_dec = decomposed[: len(suggestions), :]
    contribution_dec = decomposed[
        len(suggestions) : len(suggestions) + len(contributions), :
    ]
    items_others_dec = decomposed[-len(bought_by_similar_users) :, :]

    # Also, extract the description to create a nice hover in
    # the final plot.

    contribution_description = _extract_description(df, contributions)
    items_other_description = _extract_description(df, bought_by_similar_users)
    suggestion_description = _extract_description(df, suggestions)

    # Plot the scatterplot

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=contribution_dec[:, 0],
            y=contribution_dec[:, 1],
            mode="markers",
            opacity=0.8,
            name="Similar bought by user",
            marker_symbol="square-open",
            marker_color="#010CFA",
            marker_size=10,
            hovertext=contribution_description.loc[contributions].values.squeeze(),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=items_others_dec[:, 0],
            y=items_others_dec[:, 1],
            mode="markers",
            name="Product bought by similar users",
            opacity=0.7,
            marker_symbol="circle-open",
            marker_color="#FA5F19",
            marker_size=10,
            hovertext=items_other_description.loc[
                bought_by_similar_users
            ].values.squeeze(),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=suggestion_dec[:, 0],
            y=suggestion_dec[:, 1],
            mode="markers",
            name="Suggested",
            marker_color="#1A9626",
            marker_symbol="star",
            marker_size=10,
            hovertext=suggestion_description.loc[suggestions].values.squeeze(),
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(plot_bgcolor="white")

    return fig


def _tsne_decomposition(data: np.ndarray, tsne_args: Dict[str, Any]):
    if data.shape[1] > 50:
        print("Performing PCA...")
        data = PCA(n_components=50).fit_transform(data)
    return TSNE(
        n_components=2,
        # n_jobs=cpu_count(),
        **tsne_args,
    ).fit_transform(data)

