# impactchart

[![Hippocratic License HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-eco-extr-ffd-law-mil-sv.html)
![PyPI](https://img.shields.io/pypi/v/impactchart)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/impactchart)

![PyPI - Status](https://img.shields.io/pypi/status/impactchart?label=PyPI%20Status)
![PyPI - Format](https://img.shields.io/pypi/format/impactchart?label=PyPI%20Format)
![PyPI - Downloads](https://img.shields.io/pypi/dm/impactchart?label=PyPI%20Downloads)

![GitHub last commit](https://img.shields.io/github/last-commit/vengroff/impactchart)
[![Tests Badge](./reports/junit/tests-badge.svg)](https://vengroff.github.io/impactchart/)
[![Coverage Badge](./reports/coverage/coverage-badge.svg)](https://vengroff.github.io/impactchart/)

`impactchart` is a Python library for generating impact charts.

Impact charts make it easy to visualize the impact of one variable on another
in ways that techniques like scatter plots and linear regression don't, especially
when there are other variables involved. 

So instead of looking for the impact of `x_3` on `y` in a scatter plot like this

![A scatter plot of y vs. x_3](./images/y_vs_x3.png)

we can see it more directly in an impact chart like this

![An impact chart showing the impact of x_3 on y](./images/x3_impact.png)

In the scatter plot, we added regression curves for linear regression and
quadratic regression, but they did not tell us much more. The reason is that
in the data set we are looking at, `x_3` isn't the only feature that impacts
`y`. There are other `x_i` whose cumulative impact hides that of `x_3`.

The impact chart, on the other hand, we see the impact of `x_3` on `y` independent
of the effect of any other `x_i`. The green dots represent our best estimate of the 
impact. The grey dots around them represent the estimate of the impact based on many
(in this case 50) different machine learning models. When they are close to the green
dots, as on the left side of the chart it means there is strong agreement among the
models as to the impact. When they are farther apart, as on the left side of the chart,
there is less agreement.

The general shape of the curve of green dots, and the fact that the gray dots remain
rather tightly grouped around it, suggest that the impact of `x_3` on `y` is very limited
when `x_3` is negative. But as it becomes increasingly positive, it's impact grows more 
and more rapidly. It might even be exponential \[Spoiler alert: this is synthetic data and 
the impact is exponential.].

The reason impact charts like the one we are looking at are so powerful is that they 
very clealy and directly show us the impact of one feature in a data set on a target 
of interest.

Impact charts can find impact even though, unlike parametric techniques like linear and 
quadratic regression, they don't have any _a priori_ knowledge of the 
shape of the impacts they are looking for. For example, in the data set we have been
looking at, there is another feature `x_2` whose impact on `y` is sinusoidal. And the 
impact chart for it shows this clearly

![An impact chart showing the impact of x_2 on y](./images/x2_impact.png)






Please see [An Introduction to Impact Charts](https://datapinions.com/an-introduction-to-impact-charts/)
for an introduction to what they are all about.

Applications built on top of `impactchart` be found in the 
projects [evlcharts](https://github.com/vengroff/evlcharts) and
[rihcharts](https://github.com/vengroff/rihcharts).

An earlier version of the code that led to what is now
here produced the impact charts available at [http://rih.datapinions.com/impact.html](http://rih.datapinions.com/impact.html).
This work, and the motivation for the impact chart approach, is discussed at length in the blog post
[Using Interpretable Machine Learning to Analyze Racial and Ethnic Disparities in Home Values](https://datapinions.com/using-interpretable-machine-learning-to-analyze-racial-and-ethnic-disparities-in-home-values/).
