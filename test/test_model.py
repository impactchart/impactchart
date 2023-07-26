import unittest

import matplotlib.pyplot as plt
import pandas as pd
import impactchart.model as imm
import numpy as np


class LinearModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._n = 10
        self._k = 5

        generator = np.random.default_rng(17)
        self._X = pd.DataFrame(
            {
                'X0': generator.uniform(0, 10, self._n),
                'X1': generator.uniform(0, 5, self._n),
                'X2': generator.uniform(0, 10, self._n),
            }
        )
        #
        # The function is
        #
        #     y = X0 + 3 * X2 - 10 + noise
        #
        # X1 has no effect on the value.
        #
        self._y = self._X['X0'] + 3 * self._X['X2'] - 10 + generator.normal(scale=0.01, size=len(self._X.index))
        self._y.rename('y', inplace=True)

        self._linear = imm.LinearImpactModel(
            ensemble_size=self._k,
            random_state=97,
        )

        self.assertFalse(self._linear.is_fit)
        self._linear.fit(self._X, self._y)
        self.assertTrue(self._linear.is_fit)

    def test_coef(self):
        coef = self._linear.coefs
        for c in coef:
            self.assertAlmostEqual(1.0, c[0], delta=0.01)
            self.assertAlmostEqual(0.0, c[1], delta=0.01)
            self.assertAlmostEqual(3.0, c[2], delta=0.01)

    def test_intercept(self):
        intercepts = self._linear.intercepts
        for intercept in intercepts:
            self.assertAlmostEqual(-10.0, intercept, delta=0.1)

    def test_predict(self):
        df_y_hat = self._linear.predict(self._X)

        def _assert_y_close(df):
            y_hat_reindexed = df[['X_index', 'y_hat']].set_index('X_index')['y_hat']
            y_hat_reindexed.name = 'y'
            y_hat_reindexed.index.name = None
            pd.testing.assert_series_equal(self._y, y_hat_reindexed, atol=0.05)

        df_y_hat.groupby('estimator').apply(_assert_y_close)

    def test_impact(self):
        df_impact = self._linear.impact(self._X)

        y_mean = np.mean(self._y)

        def _assert_impact_sum(df):
            df_reindexed = df[['X_index', 'X0', 'X1', 'X2']].set_index('X_index')
            df_reindexed.index.name = None

            # Construct a series of zeroes with the same index
            df_zeros = pd.DataFrame(index=df_reindexed.index)
            df_zeros['X1'] = 0.0
            zeros_with_same_index = df_zeros['X1']

            # Close to zero impact from the feature that is ignored.
            pd.testing.assert_series_equal(zeros_with_same_index, df_reindexed['X1'], atol=0.1)

            y_impact = df_reindexed.sum(axis='columns') + y_mean
            y_impact.name = 'y'
            pd.testing.assert_series_equal(self._y, y_impact, atol=0.05)

        df_impact.groupby('estimator').apply(_assert_impact_sum)

        # Now check that the sum of the impacts and y_mean add up to the prediction.
        y_hat = self._linear.predict(self._X)
        impact_y_hat = df_impact[['X0', 'X1', 'X2']].sum(axis='columns') + y_mean
        impact_y_hat.name = 'y_hat'

        pd.testing.assert_series_equal(y_hat['y_hat'], impact_y_hat, atol=0.05)


class XgbTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._n = 100
        self._k = 10

        generator = np.random.default_rng(1619)
        self._X = pd.DataFrame(
            {
                'X0': generator.uniform(0, 1, self._n),
                'X1': generator.normal(0.5, 0.25, self._n),
                'X2': generator.uniform(0, 1, self._n),
                'X3': generator.uniform(0, 1, self._n),
            }
        )

        df_intermediate = pd.DataFrame()
        df_intermediate['X1'] = self._X['X1']
        df_intermediate['X2'] = self._X['X2']
        df_intermediate['a'] = 5 * (df_intermediate['X1'] - 0.3) + 1
        df_intermediate['lower_bound'] = 0.0
        df_intermediate['upper_bound'] = 1.0
        df_intermediate['b'] = df_intermediate[['a', 'lower_bound']].max(axis='columns')
        df_intermediate['c'] = df_intermediate[['b', 'upper_bound']].min(axis='columns')
        df_intermediate['d'] = pow(df_intermediate['X2'], 0.1)
        df_intermediate['y'] = df_intermediate['c'] * df_intermediate['d']

        self._y = df_intermediate['y']

        self._impact_model = imm.XGBoostImpactModel(
            ensemble_size=self._k,
            random_state=1619,
            estimator_kwargs={
                'n_estimators': 30,
                'max_depth': 3,
            }
        )

        self.assertFalse(self._impact_model.is_fit)
        self._impact_model.fit(self._X, self._y)
        self.assertTrue(self._impact_model.is_fit)

        ax = df_intermediate.plot.scatter('X1', 'y', figsize=(8, 6))
        ax.grid()

        plt.savefig('/var/tmp/c.png')

        #a = np.max(np.min(10 * (self._X['X1'] - 0.5)  1.0), 0.0)
        #b = self._X['X0'] * self._X['X3']

        #self._y = a * b

    def test_impact(self):
        df_impact = self._impact_model.impact(self._X)

        y_mean = np.mean(self._y)

        def _assert_impact_sum(df):
            df_reindexed = df[['X_index', 'X0', 'X1', 'X2', 'X3']].set_index('X_index')
            df_reindexed.index.name = None

            # Construct a series of zeroes with the same index
            df_zeros = pd.DataFrame(index=df_reindexed.index)
            df_zeros['zero'] = 0.0
            zeros_with_same_index = df_zeros['zero']

            # Close to zero impact from the features that are ignored.
            zeros_with_same_index.name = 'X0'
            pd.testing.assert_series_equal(zeros_with_same_index, df_reindexed['X0'], atol=0.05)
            zeros_with_same_index.name = 'X3'
            pd.testing.assert_series_equal(zeros_with_same_index, df_reindexed['X3'], atol=0.05)

            # Here we are checking to make sure the predictions
            # are not too terribly off from the training data.
            y_impact = df_reindexed.sum(axis='columns') + y_mean
            y_impact.name = 'y'
            pd.testing.assert_series_equal(self._y, y_impact, atol=0.25)

        df_impact.groupby('estimator').apply(_assert_impact_sum)

        # Now check that the sum of the impacts and y_mean add up to the prediction.
        y_hat = self._impact_model.predict(self._X)
        impact_y_hat = df_impact[['X0', 'X1', 'X2', 'X3']].sum(axis='columns') + y_mean
        impact_y_hat.name = 'y_hat'

        pd.testing.assert_series_equal(y_hat['y_hat'], impact_y_hat.astype('float32'), atol=0.005)


if __name__ == '__main__':
    unittest.main()
