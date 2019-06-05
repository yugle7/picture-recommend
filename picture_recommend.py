from datetime import date, timedelta
from collections import defaultdict
import csv
from math import sqrt
from random import shuffle
from score import mapk


class PictureRecommendSystem:
    user_pictures = defaultdict(list)
    picture_weight = defaultdict(float)

    picture_actuality = {}
    picture_popularity = defaultdict(float)
    popular_pictures = []

    group_popular_pictures = []
    user_group = {}

    user_weighted_pictures = {}
    test_user_pictures = defaultdict(list)
    test_users = []

    user_picture_date = {}
    picture_clicks = defaultdict(int)

    pictures_count = 100  # максимальное количество картинок, которые предсказываем
    pictures_delta = 10  # добавляем столько картинок человеку и снова вычисляем для него предсказания
    shift_days = 7  # смещение по времени (для тестов)
    actual_days = 9  # время, по которому вычисляем актуальность картинок
    test_users_count = 40000  # берем столько пользователей и выбираем из них тех, у кого были старые клики и новые

    def __init__(self, sex, age):
        self.user_pictures.clear()
        self.picture_weight.clear()

        self.picture_actuality.clear()
        self.picture_popularity.clear()
        self.popular_pictures.clear()

        self.group_popular_pictures.clear()
        self.user_group.clear()

        self.groups_count = 4
        self.sex = sex
        self.age = age

        self.user_weighted_pictures.clear()
        self.test_user_pictures.clear()
        self.test_users.clear()

        self.user_picture_date.clear()
        self.picture_clicks.clear()

        self.today = date(2019, 3, 25) - timedelta(self.shift_days)

    def read_user_pictures(self):
        # читаем картинки из файла

        src = open('data/train_clicks.csv')
        src.readline()

        user_picture_count = 0

        for u, p, d in csv.reader(src, delimiter=','):
            u = int(u)
            p = int(p)

            self.user_picture_date[(u, p)] = d
            self.user_pictures[u].append(p)
            user_picture_count += 1
        src.close()

        print('{}/{} clicks'.format(len(self.user_picture_date), user_picture_count))

    def calc_picture_weight(self):
        # вычисляем вес картинки - насколько клик по ней влияет на схожесть пользователей

        picture_dates = defaultdict(list)

        for (u, p), d in self.user_picture_date.items():
            days = (self.today - date(int(d[:4]), int(d[5:7]), int(d[8:]))).days
            if days < 0:
                continue

            picture_dates[p].append(days)
            if days < self.actual_days:
                self.picture_clicks[p] += 1

            self.picture_actuality[p] = min(self.picture_actuality.get(p, self.actual_days), days)

        picture_clicks = list(self.picture_clicks.items())
        picture_clicks.sort(key=lambda q: -q[1])
        self.popular_pictures = [p for p, c in picture_clicks[:self.pictures_count]]

        s = sum(c for p, c in picture_clicks[:self.pictures_count])
        self.picture_popularity = {p: c / s for p, c in picture_clicks}

        for p, ds in picture_dates.items():
            # чем больше кликали, тем меньше картинка влияет на расстояние между пользователями

            w = len(ds) / (1 + min(ds) + max(ds))
            w = max(float(1), min(float(10), sqrt(w)))
            self.picture_weight[p] = 1 / w

        print(len(self.picture_weight), 'pictures with weight')

    def calc_user_group(self):
        # вычисляем популярность картинок по группам

        with open('data/user_information.csv') as src:
            src.readline()
            for q in src:
                u, d, es = q.split(',')
                es = list(map(float, es.split()))
                g = (es[0] > self.sex) + 2 * (es[11] > self.age)
                self.user_group[int(u)] = g

        group_picture_clicks = [defaultdict(int) for g in range(self.groups_count + 1)]

        for (u, p), d in self.user_picture_date.items():
            g = self.user_group.get(u, self.groups_count)
            group_picture_clicks[g][p] += 1

        for g in range(self.groups_count + 1):
            picture_clicks = list(group_picture_clicks[g].items())
            picture_clicks.sort(key=lambda q: -q[1])
            popular_pictures = [p for p, c in picture_clicks[:self.pictures_count]]
            self.group_popular_pictures.append(popular_pictures)

    def read_test_users(self):
        # читаем тестовых пользователей из файла

        with open('data/test_users.csv') as src:
            src.readline()
            self.test_users = list(map(int, src.read().split()))

    def take_actual_clicks(self):
        # оставляем только самые актуальные клики

        src = open('data/clicks.csv')
        dst = open('data/train_clicks.csv', 'w')
        dst.write(src.readline())

        after = (self.today - timedelta(self.actual_days)).strftime('%Y-%m-%d')

        for q in src:
            u, p, d = q.split(',')

            if d < after:
                continue

            dst.write(q)

        src.close()
        dst.close()

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
        today = self.today.strftime('%Y-%m-%d')
        after = (self.today - timedelta(self.actual_days)).strftime('%Y-%m-%d')

        for q in src:
            u, p, d = q[:-1].split(',')
            u = int(u)
            p = int(p)

            if d < after:
                continue

            if d < today:
                dst.write(q)

            if u in self.test_users:
                if d < today:
                    user_in_test[u] |= 1  # обучение

                else:
                    self.test_user_pictures[u].append(p)
                    user_in_test[u] |= 2  # тест

        for u in list(self.test_users):
            if user_in_test[u] != 3:
                self.test_users.remove(u)

        src.close()
        dst.close()

        self.test_users = list(self.test_users)
        print(len(self.test_users), 'test users')

    def take_test_users(self):
        # берем тестовых пользователей из файла и делаем последнюю неделю тестовой

        self.read_test_users()

        src = open('data/clicks.csv')
        dst = open('data/train_clicks.csv', 'w')
        dst.write(src.readline())

        user_in_test = defaultdict(int)
        today = self.today.strftime('%Y-%m-%d')
        after = (self.today - timedelta(self.actual_days)).strftime('%Y-%m-%d')

        for q in src:
            u, p, d = q[:-1].split(',')
            u = int(u)
            p = int(p)

            if d < after:
                continue

            if d < today:
                dst.write(q)

            if u in self.test_users:
                if d < today:
                    user_in_test[u] |= 1  # обучение

                else:
                    self.test_user_pictures[u].append(p)
                    user_in_test[u] |= 2  # тест

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

        # при равном предсказании давать предпочтение более новым

        for p in picture_prediction:
            picture_prediction[p] -= self.picture_actuality[p] / 16 / self.actual_days

        predicted_pictures = [(p, s) for p, s in picture_prediction.items() if s > 0.5]

        # берем картинки с наибольшим предсказанием

        predicted_pictures.sort(key=lambda q: -q[1])
        predicted_pictures = [p for p, s in predicted_pictures[:self.pictures_count]]

        print(u, len(user_pictures), len(predicted_pictures))

        if not predicted_pictures and picture_prediction:
            # берем картинку с максимальным предсказанием

            p, s = max(list(picture_prediction.items()), key=lambda q: q[1])
            predicted_pictures = [p]

        if not predicted_pictures:
            # добавляем самые популярные картинки

            g = self.user_group.get(u, self.groups_count)
            popular_pictures = self.group_popular_pictures[g]

            predicted_pictures = delta_pictures + [p for p in popular_pictures if p not in user_pictures]
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
    prs = PictureRecommendSystem(0.6, 0.5)

    prs.take_test_users()
    prs.read_user_pictures()

    prs.calc_picture_weight()
    prs.get_user_weighted_pictures()

    prs.calc_user_group()

    prs.save_predicted_pictures()
    print(prs.get_score())

# 26.09963632786263
