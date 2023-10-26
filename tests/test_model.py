import unittest
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
from skimage.metrics import structural_similarity as ssim

import impactchart.model as imm


class ImpactChartTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Global set up once."""
        cls.shapefile_path = (
            Path(__file__).parent / "data" / "shapefiles" / "cb_2020_us_state_20m"
        )
        cls.expected_dir = Path(__file__).parent / "expected"

        # Create a clean output directory
        cls.output_dir = Path(__file__).parent / "_test_artifacts"

        rmtree(cls.output_dir, ignore_errors=True)
        cls.output_dir.mkdir(parents=True)

        plt.rcParams["figure.figsize"] = (8, 5)

    def assert_structurally_similar(
        self, file0: Path, file1: Path, threshold: float = 0.98
    ):
        """
        Assert that the images stored in two files are structurally similar.

        Parameters
        ----------
        file0
            An image file
        file1
            Another image file
        threshold
            Minimum structural similarity threshold.

        Returns
        -------
            None
        """
        image0 = skimage.io.imread(file0)
        image1 = skimage.io.imread(file1)

        for ii in range(len(image0[0, 0, :])):
            similarity = ssim(image0[:, :, ii], image1[:, :, ii])

            self.assertGreater(similarity, threshold)


class LinearModelTestCase(ImpactChartTestCase):
    def setUp(self) -> None:
        self._n = 50
        self._k = 50

        generator = np.random.default_rng(17)
        self._X = pd.DataFrame(
            {
                "X0": generator.uniform(0, 10, self._n),
                "X1": generator.uniform(0, 5, self._n),
                "X2": generator.uniform(0, 10, self._n),
            }
        )
        #
        # The function is
        #
        #     y = X0 + 3 * X2 - 10 + noise
        #
        # X1 has no effect on the value.
        #
        self._y = (
            self._X["X0"]
            + 3 * self._X["X2"]
            - 10
            + generator.normal(scale=0.01, size=len(self._X.index))
        )
        self._y.rename("y", inplace=True)

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
            y_hat_reindexed = df[["X_index", "y_hat"]].set_index("X_index")["y_hat"]
            y_hat_reindexed.name = "y"
            y_hat_reindexed.index.name = None
            pd.testing.assert_series_equal(self._y, y_hat_reindexed, atol=0.05)

        df_y_hat.groupby("estimator").apply(_assert_y_close)

    def test_impact(self):
        df_impact = self._linear.impact(self._X)

        y_mean = np.mean(self._y)

        def _assert_impact_sum(df):
            df_reindexed = df[["X_index", "X0", "X1", "X2"]].set_index("X_index")
            df_reindexed.index.name = None

            # Construct a series of zeroes with the same index
            df_zeros = pd.DataFrame(index=df_reindexed.index)
            df_zeros["X1"] = 0.0
            zeros_with_same_index = df_zeros["X1"]

            # Close to zero impact from the feature that is ignored.
            pd.testing.assert_series_equal(
                zeros_with_same_index, df_reindexed["X1"], atol=0.1
            )

            y_impact = df_reindexed.sum(axis="columns") + y_mean
            y_impact.name = "y"
            pd.testing.assert_series_equal(self._y, y_impact, atol=0.05)

        df_impact.groupby("estimator").apply(_assert_impact_sum)

        # Now check that the sum of the impacts and y_mean add up to the prediction.
        y_hat = self._linear.predict(self._X)
        impact_y_hat = df_impact[["X0", "X1", "X2"]].sum(axis="columns") + y_mean
        impact_y_hat.name = "y_hat"

        pd.testing.assert_series_equal(y_hat["y_hat"], impact_y_hat, atol=0.05)

    def test_impact_chart(self):
        charts = self._linear.impact_charts(self._X, self._X.columns)
        for feature, (fig, ax) in charts.items():

            png_file_name = f"impact_linear_{feature}.png"
            expected_file = self.expected_dir / png_file_name
            output_file = self.output_dir / png_file_name

            ax.set_ylim(-20, 20)
            fig.savefig(output_file)

            self.assert_structurally_similar(expected_file, output_file)

    def test_styled_impact_chart(self):
        charts = self._linear.impact_charts(
            self._X,
            self._X.columns,
            markersize=4,
            color="red",
            ensemble_markersize=20,
            ensemble_color="lightblue",
            subplots_kwargs=dict(figsize=(12, 6)),
            feature_names={'X0': "Name 0", 'X1': 'Foo', 'X2': 'Bar'},
            y_name="the Y",
        )
        for feature, (fig, ax) in charts.items():
            png_file_name = f"impact_linear_styled_{feature}.png"
            expected_file = self.expected_dir / png_file_name
            output_file = self.output_dir / png_file_name

            ax.set_ylim(-20, 20)
            fig.savefig(output_file)

            self.assert_structurally_similar(expected_file, output_file)


class XgbTestCase(ImpactChartTestCase):
    def setUp(self) -> None:
        self._n = 100
        self._k = 10

        generator = np.random.default_rng(1619)
        self._X = pd.DataFrame(
            {
                "X0": generator.uniform(0, 1, self._n),
                "X1": generator.normal(0.5, 0.25, self._n),
                "X2": generator.uniform(0, 1, self._n),
                "X3": generator.uniform(0, 1, self._n),
            }
        )

        df_intermediate = pd.DataFrame()
        df_intermediate["X1"] = self._X["X1"]
        df_intermediate["X2"] = self._X["X2"]
        df_intermediate["a"] = 5 * (df_intermediate["X1"] - 0.3) + 1
        df_intermediate["lower_bound"] = 0.0
        df_intermediate["upper_bound"] = 1.0
        df_intermediate["b"] = df_intermediate[["a", "lower_bound"]].max(axis="columns")
        df_intermediate["c"] = df_intermediate[["b", "upper_bound"]].min(axis="columns")
        df_intermediate["d"] = pow(df_intermediate["X2"], 0.1)
        df_intermediate["y"] = df_intermediate["c"] * df_intermediate["d"]

        self._y = df_intermediate["y"]

        self._impact_model = imm.XGBoostImpactModel(
            ensemble_size=self._k,
            random_state=1619,
            estimator_kwargs={
                "n_estimators": 30,
                "max_depth": 3,
            },
        )

        self.assertFalse(self._impact_model.is_fit)
        self._impact_model.fit(self._X, self._y)
        self.assertTrue(self._impact_model.is_fit)

        ax = df_intermediate.plot.scatter("X1", "y", figsize=(8, 6))
        ax.grid()

    def test_impact(self):
        df_impact = self._impact_model.impact(self._X)

        y_mean = np.mean(self._y)

        def _assert_impact_sum(df):
            df_reindexed = df[["X_index", "X0", "X1", "X2", "X3"]].set_index("X_index")
            df_reindexed.index.name = None

            # Construct a series of zeroes with the same index
            df_zeros = pd.DataFrame(index=df_reindexed.index)
            df_zeros["zero"] = 0.0
            zeros_with_same_index = df_zeros["zero"]

            # Close to zero impact from the features that are ignored.
            zeros_with_same_index.name = "X0"
            pd.testing.assert_series_equal(
                zeros_with_same_index, df_reindexed["X0"], atol=0.05
            )
            zeros_with_same_index.name = "X3"
            pd.testing.assert_series_equal(
                zeros_with_same_index, df_reindexed["X3"], atol=0.05
            )

            # Here we are checking to make sure the predictions
            # are not too terribly off from the training data.
            y_impact = df_reindexed.sum(axis="columns") + y_mean
            y_impact.name = "y"
            pd.testing.assert_series_equal(self._y, y_impact, atol=0.25)

        df_impact.groupby("estimator").apply(_assert_impact_sum)

        # Now check that the sum of the impacts and y_mean add up to the prediction.
        y_hat = self._impact_model.predict(self._X)
        impact_y_hat = df_impact[["X0", "X1", "X2", "X3"]].sum(axis="columns") + y_mean
        impact_y_hat.name = "y_hat"

        pd.testing.assert_series_equal(
            y_hat["y_hat"], impact_y_hat.astype("float32"), atol=0.005
        )

    def test_impact_chart(self):
        charts = self._impact_model.impact_charts(
            self._X,
            self._X.columns,
            feature_names=lambda x: f'Name of {x}',
        )

        for feature, (fig, ax) in charts.items():
            png_file_name = f"impact_xgb_{feature}.png"
            expected_file = self.expected_dir / png_file_name
            output_file = self.output_dir / png_file_name

            ax.set_ylim(-1, 1)
            fig.savefig(output_file)

            self.assert_structurally_similar(expected_file, output_file)


class KnnTestCase(ImpactChartTestCase):
    def setUp(self) -> None:
        self._n = 5

        # The point itself and the four at distance 1 in each direction
        # around it.
        self._k = 1  # 5

        generator = np.random.default_rng(997)

        self._X = pd.DataFrame(
            [[x1, x2] for x1 in range(self._n) for x2 in range(self._n)],
            columns=["X1", "X2"],
        )
        self._y = 100 * self._X["X1"] + generator.normal() * self._X["X2"]

        self._impact_model = imm.KnnImpactModel(
            estimator_kwargs=dict(n_neighbors=self._k),
            ensemble_size=2,
        )

    def test_knn_fit(self):
        self.assertFalse(self._impact_model.is_fit)
        self._impact_model.fit(self._X, self._y)
        self.assertTrue(self._impact_model.is_fit)

    def test_knn(self):
        self._impact_model.fit(self._X, self._y)

        charts = self._impact_model.impact_charts(self._X, self._X.columns, y_name="Prediction")

        for feature, (fig, ax) in charts.items():
            png_file_name = f"impact_knn_{feature}.png"
            expected_file = self.expected_dir / png_file_name
            output_file = self.output_dir / png_file_name

            ax.set_ylim(-250, 250)
            fig.savefig(output_file)

            self.assert_structurally_similar(expected_file, output_file)


if __name__ == "__main__":
    unittest.main()
