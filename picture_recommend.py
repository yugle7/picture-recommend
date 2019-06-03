from datetime import date
from collections import defaultdict
import csv
from math import sqrt
from random import shuffle
from score import mapk


class PictureRecommendSystem:
    user_pictures = defaultdict(list)
    picture_weight = defaultdict(int)

    picture_popularity = defaultdict(float)
    popular_pictures = []

    user_weighted_pictures = {}
    test_user_pictures = defaultdict(list)
    test_users = []

    user_picture_date = {}
    picture_clicks = defaultdict(int)

    pictures_count = 100  # максимальное количество картинок, которые предсказываем
    pictures_delta = 10  # добавляем столько картинок человеку и снова вычисляем для него предсказания
    actual_days = 28  # время, по которому вычисляем актуальность картинок
    test_users_count = 40000  # берем столько пользователей и выбираем из них тех, у кого были старые клики и новые

    def read_user_pictures(self):
        # читаем картинки из файла
        src = open('data/train_clicks.csv')
        src.readline()

        user_picture_count = 0

        for u, p, d in csv.reader(src, delimiter=','):
            self.user_picture_date[(int(u), int(p))] = d
            self.user_pictures[int(u)].append(int(p))
            user_picture_count += 1
        src.close()

        print('{}/{} clicks'.format(len(self.user_picture_date), user_picture_count))

    def calc_picture_weight(self):
        # вычисляем вес картинки - насколько клик по ней влияет на схожесть пользователей

        today = date(2019, 3, 24)
        picture_dates = defaultdict(list)

        for (u, p), d in self.user_picture_date.items():
            days = (today - date(int(d[:4]), int(d[5:7]), int(d[8:]))).days
            picture_dates[p].append(days)
            if days < self.actual_days:
                self.picture_clicks[p] += 1

        picture_clicks = list(self.picture_clicks.items())
        picture_clicks.sort(key=lambda q: -q[1])
        self.popular_pictures = [p for p, c in picture_clicks[:self.pictures_count]]

        s = sum(c for p, c in picture_clicks[:self.pictures_count])
        self.picture_popularity = {p: c / s for p, c in picture_clicks}

        for p, ds in picture_dates.items():
            # чем больше кликали, тем меньше картинка влияет на расстояние между пользователями
            n = len(ds)
            self.picture_weight[p] = sqrt((1 + min(ds) + max(ds)) / n) if n > 10 else 3

        print(len(self.picture_weight), 'pictures with weight')

    def read_test_users(self):
        # читаем тестовых пользователей из файла

        with open('data/test_users.csv') as src:
            src.readline()
            self.test_users = list(map(int, src.read().split()))

    def make_test_users(self):
        # создаем тестовых пользователей

        users = set()

        with open('data/clicks.csv') as f:
            f.readline()
            for u, p, d in csv.reader(f, delimiter=','):
                users.add(int(u))

        users = list(users)
        shuffle(users)

        # случайно берем некоторое количество людей
        self.test_users = set(users[:self.test_users_count])

        src = open('data/clicks.csv')
        dst = open('data/train_clicks.csv', 'w')
        dst.write(src.readline())

        user_in_test = defaultdict(int)

        for q in src:
            u, p, d = q.split(',')
            if int(u) in self.test_users:
                if d < '2019-03-17\n':
                    # человек имеет старые клики
                    dst.write(q)
                    user_in_test[int(u)] |= 1
                else:
                    # есть что ему предсказывать
                    self.test_user_pictures[int(u)].append(int(p))
                    user_in_test[int(u)] |= 2
            else:
                dst.write(q)

        for u in list(self.test_users):
            if user_in_test[u] != 3:
                self.test_users.remove(u)

        src.close()
        dst.close()

        self.test_users = list(self.test_users)
        print(len(self.test_users), 'test users')

    def distance(self, u, v):
        # расстояние это взвешенная сумма картинок, которые отличаются у людей

        upw = self.user_weighted_pictures[u]
        vpw = self.user_weighted_pictures[v]

        d = 0
        for p, w in upw.items():
            if p not in vpw:
                d += w
        for p, w in vpw.items():
            if p not in upw:
                d += w

        return d

    def get_user_weighted_pictures(self):
        # получаем взвешенное описание человека через картинки

        for u, ps in self.user_pictures.items():
            ps = set(ps)
            s = sum(self.picture_weight[p] for p in ps)
            self.user_weighted_pictures[u] = {p: self.picture_weight[p] / s for p in ps}

    def predict_user_pictures(self, u, delta_pictures):
        # делаем предсказание картинок, по которым человек кликнет

        user_pictures = set(self.user_pictures[u])
        user_distances = []

        # считаем расстояния до всех пользователей
        for v in self.user_weighted_pictures:
            if u == v:
                continue
            common_pictures = user_pictures.intersection(self.user_weighted_pictures[v].keys())
            if common_pictures and len(common_pictures) != len(self.user_weighted_pictures[v]):
                d = self.distance(u, v)
                user_distances.append((d, v))

        # считаем предсказания в зависимости от близости людей
        picture_prediction = defaultdict(float)
        user_distances.sort()

        for d, v in user_distances:
            for p in self.user_weighted_pictures[v]:
                if p not in user_pictures:
                    picture_prediction[p] += 2 - d

        for p in picture_prediction:
            picture_prediction[p] += self.picture_popularity.get(p, 0)

        predicted_pictures = [(p, s) for p, s in picture_prediction.items() if s > 0.5]

        if len(predicted_pictures) < 3:
            # если картинок с хорошим предсказанием не хватает, то берем несколько с плохим
            picture_prediction = list(picture_prediction.items())
            picture_prediction.sort(key=lambda q: -q[1])
            predicted_pictures = [p for p, s in picture_prediction[:2]]
        else:
            # берем картинки с наибольшим предсказанием
            predicted_pictures.sort(key=lambda q: -q[1])
            predicted_pictures = [p for p, s in predicted_pictures[:self.pictures_count]]

        print(u, len(user_pictures), len(predicted_pictures))

        if not predicted_pictures:
            # добавляем самые популярные картинки
            predicted_pictures = delta_pictures + [p for p in self.popular_pictures if p not in user_pictures]
        else:
            if len(delta_pictures) + len(predicted_pictures) < self.pictures_count:
                # добавляем человеку несколько предсказанных картинок и делаем новое предсказание
                predicted_pictures = predicted_pictures[:self.pictures_delta]

                self.user_pictures[u] += predicted_pictures
                user_pictures.update(predicted_pictures)

                s = sum(self.picture_weight[p] for p in user_pictures)
                self.user_weighted_pictures[u] = {p: self.picture_weight[p] / s for p in user_pictures}

                return self.predict_user_pictures(u, delta_pictures + predicted_pictures)

            predicted_pictures = delta_pictures + predicted_pictures

        return predicted_pictures[:self.pictures_count]

    def save_predicted_pictures(self):
        # делаем предсказание картинок для тестовых пользователей

        dst = open('predictions.csv', 'w')
        dst.write('user_id,predictions\n')

        user_number = 0
        for u in self.test_users:
            user_number += 1
            print(user_number)

            predicted_pictures = self.predict_user_pictures(u, [])
            dst.write(str(u) + ',' + ' '.join(map(str, predicted_pictures)) + '\n')

        dst.close()

    def get_score(self):
        # вычисляем mapk

        test_user_pictures = defaultdict(list)
        src = open('predictions.csv')
        src.readline()

        for q in src:
            u, ps = q.split(',')
            test_user_pictures[int(u)] = list(map(int, ps.split()))

        src.close()

        actual = [self.test_user_pictures[u] for u in self.test_users]
        predicted = [test_user_pictures[u] for u in self.test_users]
        return mapk(actual, predicted)


if __name__ == '__main__':
    prs = PictureRecommendSystem()

    prs.make_test_users()
    prs.read_user_pictures()

    prs.calc_picture_weight()
    prs.get_user_weighted_pictures()

    prs.save_predicted_pictures()
    print(prs.get_score())
