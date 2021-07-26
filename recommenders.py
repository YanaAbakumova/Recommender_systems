import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import AnnoyAlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, bm25_weight, tfidf_weight, BM25Recommender


class MainRecommender:

    def __init__(self, data: pd.DataFrame):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)
        # при обучении моделей передаем взвешенную матрицу, а при выдаче рекомендаций - невзвешенную
        self.weighted_matrix = bm25_weight(self.user_item_matrix.T).T
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        self.model = self.fit(self.weighted_matrix)
        self.own_recommender = self.fit_own_recommender(self.weighted_matrix)
        self.cosine_recommender = self.fit_cosine_recommender(self.weighted_matrix)
        self.tfidf_recommender = self.fit_tfidf_recommender(self.weighted_matrix)
        self.bm25_recommender = self.fit_bm25_recommender(self.weighted_matrix)

    @staticmethod
    def _prepare_matrix(data: pd.DataFrame):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=15)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=180, regularization=0.1, iterations=8, num_threads=15):
        """Обучает ALS"""

        model = AnnoyAlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads,
                                        calculate_training_loss=True)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    @staticmethod
    def fit_cosine_recommender(user_item_matrix):
        cosine_rec = CosineRecommender(K=1, num_threads=15)
        cosine_rec.fit(csr_matrix(user_item_matrix).T.tocsr())

        return cosine_rec

    @staticmethod
    def fit_tfidf_recommender(user_item_matrix):
        tfidf_rec = TFIDFRecommender(K=1, num_threads=15)
        tfidf_rec.fit(csr_matrix(user_item_matrix).T.tocsr())

        return tfidf_rec

    @staticmethod
    def fit_bm25_recommender(user_item_matrix):
        bm25_rec = BM25Recommender(K=1, B=0.9, num_threads=15)
        bm25_rec.fit(csr_matrix(user_item_matrix).T.tocsr())

        return bm25_rec


    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            purchases = np.setdiff1d(self.overall_top_purchases, recommendations)
            recommendations.extend(purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _extend_with_tfidf(self, recommendations, user,N=5):
        """Если кол-во рекоммендаций < N, то увеличиваем кол-во соседей в tfidf"""

        tfidf_rec = TFIDFRecommender(K=N//len(recommendations), num_threads=15)
        tfidf_rec.fit(csr_matrix(self.weighted_matrix).T.tocsr())
        return self._get_recommendations(user, model=tfidf_rec, N=N, recursion=True)


    def _get_recommendations(self, user, model, N=5, recursion=False):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix), #невзвешенная матрица
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=True)]
        # для модели tfidf: если нужное кол-во рекомендаций не получено, сначала пробуем увеличить кол-во соседей
        # в tfidf рекоммендере
        if model == self.tfidf_recommender and len(res) >0 and len(res) <= (N / 2) and not recursion:
            res = self._extend_with_tfidf(res, user, N=N)

        #если рекомендаций не достаточно, добавляем товары, которых еще нет в рекомендациях, из топ-популярных
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_cosine_recommendations(self, user, N=5):

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.cosine_recommender, N=N)

    def get_tfidf_recommendations(self, user, N=5):

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.tfidf_recommender, N=N)

    def get_bm25_recommendations(self, user, N=5):

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.bm25_recommender, N=N)


    def get_similar_items_recommendation(self, user_id, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user_id].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user_id, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user_id], N=N + 1)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for _user_id in similar_users:
            res.extend(self.get_own_recommendations(_user_id, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
