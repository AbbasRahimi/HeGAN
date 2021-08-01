import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math


class FAMILY_EVALUATION():
    def __init__(self):
        self.family2id = {}
        self.family_num = 0

        with open('../data/family1_map_id.dat') as infile:
            for line in infile.readlines():
                id, family = line.strip().split()[:2]
                id = int(id) - 1
                family = int(family)

                self.family2id[family] = id
                self.family_num += 1

        # load family label
        # id - label
        self.family_label = {}
        self.sample_num = 0
        with open('../data/family1_label.dat') as infile:
            for line in infile.readlines():
                family, label = line.strip().split()[:2]
                family = int(family)
                label = int(label) - 1
                # print("here: ", family," ", label)
                self.family_label[self.family2id[family]] = label
                self.sample_num += 1

        self.train_link_label = []
        with open('../data/family_lp/family_ub.train_0.8_lr.dat') as infile:
            for line in infile.readlines():
                u, b, label = [int(item) for item in line.strip().split()]
                self.train_link_label.append([u, b, label])
        self.test_link_label = []
        with open('../data/family_lp/family_ub.test_0.2_new.dat') as infile:
            for line in infile.readlines():
                u, b, label = [int(item) for item in line.strip().split()]
                self.test_link_label.append([u, b, label])

    def evaluate_family_cluster(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()

        X = []
        Y = []
        for family in self.family_label:
            X.append(embedding_list[family])
            Y.append(self.family_label[family])

        pred_Y = KMeans(4).fit(np.array(X)).predict(X)
        score = normalized_mutual_info_score(np.array(Y), pred_Y)

        return score

    def evaluate_family_classification(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()

        X = []
        Y = []
        for family in self.family_label:
            X.append(embedding_list[family])
            Y.append(self.family_label[family])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)

        Y_pred = lr.predict(X_test)
        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1

    def evaluation_link_prediction(self, embedding_matrix):
        embedding_list = embedding_matrix.tolist()
        train_x = []
        train_y = []
        for a, p, label in self.train_link_label:
            print("a:",a,"p:",p,"train length:",len(self.train_link_label))
            train_x.append(embedding_list[a] + embedding_list[p])
            train_y.append(label)

        test_x = []
        test_y = []
        for a, p, label in self.test_link_label:
            test_x.append(embedding_list[a] + embedding_list[p])
            test_y.append(label)

        print("test_x: \n", test_x, "\ntest_y:\n", test_y)

        lr = LogisticRegression()
        lr.fit(train_x, train_y)

        pred_y = lr.predict_proba(test_x)[:, 1]
        pred_label = lr.predict(test_x)

        '''
        test_y = []
        pred_y = []
        pred_label = []
        for u, b, label in self.link_label:
            test_y.append(label)
            pred_y.append(embedding_matrix[u].dot(relation_matrix[1]).dot(embedding_matrix[b]))

            if pred_y[-1] >= 0.5:
                pred_label.append(1)
            else:
                pred_label.append(0)
        '''
        auc = roc_auc_score(test_y, pred_y)
        f1 = f1_score(test_y, pred_label)
        acc = accuracy_score(test_y, pred_label)

        return auc, f1, acc


def str_list_to_float(self, str_list):
    return [float(item) for item in str_list]


if __name__ == '__main__':
    family_evaluation = FAMILY_EVALUATION()
