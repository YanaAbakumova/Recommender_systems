import pandas as pd
import numpy as np


def prefilter_items(data, item_features, take_n_popular=None, categories=[]):
    copy = data.copy()

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity['share_unique_users'] = popularity['share_unique_users'] / data['user_id'].nunique()
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    copy = copy[~copy['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    copy = copy[~copy['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    recent_items = data[
        data['day'] >= data['day'].max() - 365 + 28].item_id.tolist()  # +28 т.к. в data_test еще 4 недели
    copy = copy[copy['item_id'].isin(recent_items)]

    # Уберем не интересные для рекоммендаций категории (department)
    if categories:
        item_features = item_features[item_features['department'].isin(categories)]
        categories_items = pd.merge(data, items_features, on='item_id', how='inner').item_id.tolist()
        copy = copy[copy['item_id'].isin(categories_items)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    copy['price'] = copy['sales_value'] / (np.maximum(copy['quantity'], 1))
    copy = copy[copy['price'] > data['price'].quantile(0.20)]

    # Уберем слишком дорогие товары
    copy = copy[copy['price'] < data['price'].quantile(0.99995)]

    # ...
    if take_n_popular:
        popular_items = copy.groupby('item_id')['quantity'].sum().reset_index()
        popular_items.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_n = popular_items.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
        copy.loc[~copy['item_id'].isin(top_n), 'item_id'] = 999_999

    return copy


def postfilter_items(user_id, recommednations):
    pass