import pandas as pd
import numpy as np


def prefilter_items(data, quantile_pop_items, item_features=None):
    copy = data.copy()

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity['share_unique_users'] = popularity['share_unique_users'] / data['user_id'].nunique()
    # top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    # copy = copy[~copy['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.001].item_id.tolist()
    #copy.loc[copy['item_id'].isin(top_notpopular), 'item_id'] = 999999

    # Уберем товары, которые не продавались за последние 12 месяцев
    recent_items = data[
        data['day'] >= data['day'].max() - 365].item_id.tolist()  # +21 т.к. в data_test еще 3 недели
    copy.loc[~copy['item_id'].isin(recent_items), 'item_id'] = 999999

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features:
        department_size = pd.DataFrame(item_features. \
                                   groupby('department')['item_id'].nunique(). \
                                   sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
        item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        # copy.loc[copy['item_id'].isin(items_in_rare_departments), 'item_id'] = 999999

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    # copy['price'] = copy['sales_value'] / (np.maximum(copy['quantity'], 1))
    # copy = copy[copy['price'] > data['price'].quantile(0.20)]
    # copy = copy[copy['price'] > 2]

    # Уберем слишком дорогие товары
    # copy = copy[copy['price'] < data['price'].quantile(0.99995)]
    # copy = copy[copy['price'] < 50]

    # ...
    popular_items = copy.groupby('item_id')['quantity'].sum().reset_index()
    popular_items.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_popular = popular_items[popular_items['n_sold'] >=np.quantile(popular_items['n_sold'], q = quantile_pop_items)].item_id.tolist()
    copy.loc[~copy['item_id'].isin(top_popular), 'item_id'] = 999999

    return copy

