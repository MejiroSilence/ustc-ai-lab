from pickle import FALSE
import numpy as np
import cvxpy as cp


class SupportVectorMachine:
    def __init__(self, C=1, kernel="Linear", epsilon=1e-4):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel

        # Hint: 你可以在训练后保存这些参数用于预测
        # SV即Support Vector，表示支持向量，SV_alpha为优化问题解出的alpha值，
        # SV_label表示支持向量样本的标签。
        self.sv = []
        self.sv_alpha = []
        self.sv_label = []
        self.b = None

    def KERNEL(self, x1, x2, d=2, sigma=1):
        # d for Poly, sigma for Gauss
        if self.kernel == "Gauss":
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif self.kernel == "Linear":
            K = np.dot(x1, x2)
        elif self.kernel == "Poly":
            K = (np.dot(x1, x2) + 1) ** d
        else:
            raise NotImplementedError()
        return K

    def fit(self, train_data, train_label):
        size = np.size(train_label)
        alpha = cp.Variable(size)
        constraint = [
            alpha <= 1,
            alpha >= 0,
            alpha.T @ train_label == 0,
        ]
        gram = np.array(
            [
                [
                    train_label[i]
                    * train_label[j]
                    * self.KERNEL(train_data[i], train_data[j])
                    for j in range(size)
                ]
                for i in range(size)
            ]
        )
        # print(gram)
        obj = cp.Maximize(cp.sum(alpha) - (1 / 2) * cp.quad_form(alpha, gram))
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.ECOS, verbose=False)
        for i in range(size):
            if alpha.value[i] > self.epsilon:
                self.sv.append(train_data[i])
                self.sv_alpha.append(alpha.value[i])
                self.sv_label.append(train_label[i])
        self.sv = np.array(self.sv)
        self.sv_alpha = np.array(self.sv_alpha)
        self.sv_label = np.array(self.sv_label)
        self.b = self.sv_label[0] - np.sum(
            self.sv_alpha
            * self.sv_label
            * np.array([self.KERNEL(self.sv[0], i) for i in self.sv])
        )

    def predict(self, test_data):
        m = np.size(test_data, 0)
        ans = np.empty(m)
        for i in range(m):
            temp = self.b + np.sum(
                self.sv_alpha
                * self.sv_label
                * np.array([self.KERNEL(test_data[i], j) for j in self.sv])
            )
            if temp > 0:
                ans[i] = 1
            else:
                ans[i] = -1
        return ans

