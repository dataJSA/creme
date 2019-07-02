from creme import compat
from creme import datasets
from creme import dummy
from creme import linear_model
from creme import metrics
from creme import optim
from creme import preprocessing
from sklearn import linear_model as sk_linear_model

import benchmark


def main():

    benchmark.benchmark(
        get_X_y=datasets.fetch_electricity,
        n=45312,
        get_pp=preprocessing.StandardScaler,
        models=[
            # ('No-change', 'No-change', dummy.NoChangeClassifier()),
            ('creme', 'Logistic regression', linear_model.LogisticRegression(
                optimizer=optim.VanillaSGD(0.05),
                l2=0,
                intercept_lr=0.05
            )),
            # ('creme', 'PA-I', linear_model.PAClassifier(C=1, mode=1)),
            # ('creme', 'PA-II', linear_model.PAClassifier(C=1, mode=2)),
            ('sklearn', 'Logistic regression', compat.CremeClassifierWrapper(
                sklearn_estimator=sk_linear_model.SGDClassifier(
                    loss='log',
                    learning_rate='constant',
                    eta0=0.05,
                    penalty='none'
                ),
                classes=[False, True]
            )),
            # ('sklearn', 'PA-I', compat.CremeClassifierWrapper(
            #     sklearn_estimator=sk_linear_model.PassiveAggressiveClassifier(
            #         C=1,
            #         loss='hinge'
            #     ),
            #     classes=[False, True]
            # )),
            # ('sklearn', 'PA-II', compat.CremeClassifierWrapper(
            #     sklearn_estimator=sk_linear_model.PassiveAggressiveClassifier(
            #         C=1,
            #         loss='squared_hinge'
            #     ),
            #     classes=[False, True]
            # )),

            # ('sklearn', 'Logistic regression NI', compat.CremeClassifierWrapper(
            #     sklearn_estimator=sk_linear_model.SGDClassifier(
            #         loss='log',
            #         learning_rate='constant',
            #         eta0=0.01,
            #         fit_intercept=True,
            #         penalty='none'
            #     ),
            #     classes=[False, True]
            # )),
            # ('sklearn', 'PA-I NI', compat.CremeClassifierWrapper(
            #     sklearn_estimator=sk_linear_model.PassiveAggressiveClassifier(
            #         C=1,
            #         loss='hinge',
            #         fit_intercept=False
            #     ),
            #     classes=[False, True]
            # )),
            # ('sklearn', 'PA-II NI', compat.CremeClassifierWrapper(
            #     sklearn_estimator=sk_linear_model.PassiveAggressiveClassifier(
            #         C=1,
            #         loss='squared_hinge',
            #         fit_intercept=False
            #     ),
            #     classes=[False, True]
            # )),
        ],
        get_metric=metrics.Accuracy
    )


if __name__ == '__main__':
    main()
