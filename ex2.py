import sys
import numpy as np
from abc import ABC, abstractmethod


train_x_path, train_y_path, test_x_path, output_log_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train_x = np.loadtxt(train_x_path, delimiter=',')
train_y = np.loadtxt(train_y_path, delimiter=',')
test_x = np.loadtxt(test_x_path, delimiter=',')


def min_max_normalization(values, test):
    min_value = np.min(values, axis=0)  # per column
    max_value = np.max(values, axis=0)
    new_values = (values - min_value) / (max_value - min_value)
    test = (test - min_value) / (max_value - min_value)
    return new_values, test


def z_score_normalization(values, test):
    mean = np.average(values, axis=0)
    stand_dev = np.std(values, axis=0)
    new_values = (values - mean) / stand_dev
    test = (test - mean) / stand_dev
    return new_values, test


def shuffle_table(values, tags):
    seed_value = np.random.randint(0, 100)   # each run different seed
    np.random.seed(seed_value)               # same seed for x and y (values and tags)
    np.random.shuffle(values)
    np.random.seed(seed_value)
    np.random.shuffle(tags)


def split_validation(values, tags):
    train = values[:int(len(values)*0.8), :]                # 80% of the data
    validation = values[int(len(values)*0.8):, :]           # 20% of the data
    tags_train = tags[:int(len(values)*0.8)]
    tags_validation = tags[int(len(values)*0.8):]
    return train, validation, tags_train, tags_validation


# 3 out of 4s algo are similar ---> template method
class AbstractClass(ABC):
    def template_method(self):
        self.run()
        tags = self.run_test()
        return tags

    def __init__(self, train, validation, tags_train, tags_validation, test):
        self.w = np.zeros((len(set(tags_train)), train.shape[1] + 1))  # added 1 for the bias
        self.best_w = [self.w, len(tags_train)]                        # save the best and the num of mistake to compare
        self.x_train = np.ones((len(train), train.shape[1] + 1))
        self.x_train[:, :-1] = train
        self.x_validation = np.ones(((len(validation)), validation.shape[1] + 1))
        self.x_validation[:, :-1] = validation
        self.y_train = tags_train
        self.y_validation = tags_validation
        self.test_x = np.ones((len(test_x), test_x.shape[1] + 1))
        self.test_x[:, :-1] = test

    @abstractmethod
    def train(self):
        pass

    def validation(self):
        mistakes = 0
        for x, y in zip(self.x_validation, self.y_validation):
            y_hat = np.argmax(np.dot(self.w, x))
            if y != y_hat:
                mistakes += 1
        if mistakes < self.best_w[1]:
            self.best_w[0] = self.w
            self.best_w[1] = mistakes
        # print(f"mistakes: {mistakes}, accurate: {1 - (mistakes/len(self.y_validation)):.3}")

    def run(self):
        epochs = 400
        eta = 0.1
        for e in range(1, epochs):
            if e % 100 == 0:
                eta /= 2
            self.train(eta)
            self.validation()
        # print(f"the min num of mistakes: {self.best_w[1]},"
        #      f" accurate:  {1 - (self.best_w[1]/len(self.y_validation)):0.3}")

    def run_test(self):
        tags = np.empty(0)
        for x in self.test_x:
            y = np.argmax(np.dot(self.w, x))
            tags = np.append(tags, int(y))
        return tags


class Perceptron(AbstractClass):
    def train(self, eta=0.1):
        shuffle_table(self.x_train, self.y_train)
        for x, y in zip(self.x_train, self.y_train):
            y_hat = np.argmax(np.dot(self.w, x))
            if y != y_hat:
                y = int(y)
                self.w[y, :] = self.w[y, :] + eta * x
                self.w[y_hat, :] = self.w[y_hat, :] - eta * x


class PA(AbstractClass):
    def train(self):
        shuffle_table(self.x_train, self.y_train)
        for x, y in zip(self.x_train, self.y_train):
            y_hat = np.argmax(np.dot(self.w, x))
            y_hat = int(y_hat)
            y = int(y)
            loss = max(0, 1 - np.dot(self.w[y], x) + np.dot(self.w[y_hat], x))
            tau = loss / (2 * ((np.linalg.norm(x)) ** 2))
            if loss > 0:
                self.w[y, :] += tau * x
                self.w[y_hat, :] -= tau * x

    def run(self):
        epochs = 400
        for e in range(0, epochs):
            self.train()
            self.validation()
        # print(f"the min num of mistakes: {self.best_w[1]},"
        #      f" accurate:  {1 - (self.best_w[1]/len(self.y_validation)):0.3}")


class SVM(AbstractClass):

    def train(self, eta=0.1, lamda=0.08):
        shuffle_table(self.x_train, self.y_train)
        for x, y in zip(self.x_train, self.y_train):
            y_hat = np.argmax(np.dot(self.w, x))
            self.w = self.w * (1 - lamda * eta)
            if y != y_hat:
                y = int(y)
                self.w[y, :] += eta * x
                self.w[y_hat, :] -= eta * x


class KNN:
    def __init__(self, train, tags_train, validation=None, tags_validation=None):
        self.k = 3
        self.x_train = train
        self.x_validation = validation
        self.y_train = tags_train
        self.y_validation = tags_validation

    def distance(self, test):
        dist = np.empty(0)
        for i in range(len(self.x_train)):
            dist = np.append(dist, np.linalg.norm(self.x_train[i] - test))
        sort = np.argsort(dist)
        classes = np.empty(0)
        for i in range(self.k):
            classes = np.append(classes, self.y_train[sort[i]])
        return classes.max()

    def validation(self, tags):
        mistake = 0
        for i in range(len(tags)):
            if tags[i] != self.y_validation[i]:
                mistake += 1
        # print(f"num of mistakes: {mistake}, accurate: {1 - mistake/ len(tags):0.3}")

    def run(self, test):
        tags = np.empty(0)
        for num in test:
            tag = self.distance(num)
            tags = np.append(tags, tag)
        self.validation(tags)
        return tags

    def run_test(self, test):
        tags = np.empty(0)
        for num in test:
            tag = self.distance(num)
            tags = np.append(tags, tag)
        return tags


def algo_run(abstract_class: AbstractClass):
    tags = abstract_class.template_method()
    return tags


if __name__ == '__main__':
    normal_train, normal_test = min_max_normalization(train_x, test_x.copy())
    z_normal_train, test = z_score_normalization(train_x, test_x.copy())

    # shuffle_table(train_x, train_y)
    shuffle_table(z_normal_train, train_y)      # making sure the split of data doesn't affect
    train, validation, tags_train, tags_validation = split_validation(z_normal_train, train_y)

    knn_train = KNN(train.copy(), tags_train.copy(), validation.copy(), tags_validation.copy())
    # tags = knn_train.run(validation)                # helped to chose the k
    knn = KNN(z_normal_train.copy(), train_y.copy())
    knn_tags = knn.run_test(test)
    percept_tags = algo_run(Perceptron(train.copy(), validation.copy(), tags_train.copy(),
                                       tags_validation.copy(), test))
    pA_tags = algo_run(PA(train.copy(), validation.copy(), tags_train.copy(), tags_validation.copy(), test))
    svm_tags = algo_run(SVM(train.copy(), validation.copy(), tags_train.copy(), tags_validation.copy(), test))

    f = open(output_log_name, 'w')
    for i in range(len(test_x)):
        f.write(f"knn: {int(knn_tags[i])}, perceptron: {int(percept_tags[i])},"
                f" svm: {int(svm_tags[i])}, pa: {int(pA_tags[i])}\n")
    f.close()
