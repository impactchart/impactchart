# An Introduction to Impact Charts

## Abstract

Impact charts, as implemented in the [impactchart](https://github.com/vengroff/impactchart) package,
make it easy to take a data set and visualize the impact of one variable 
on another in ways that techniques like scatter plots and linear regression can't, 
especially when there are other variables involved.

In this talk, we will introduce impact charts, demonstrate how they find easter-egg impacts 
we embed in synthetic data, show how they can find hidden impacts in a real-world use case, 
show how you can create your first impact chart with just a few lines of code, 
and finally talk a bit about the interpretable machine learning techniques they are built upon.

Impact charts are primarily visual, so this talk will be too. 

## Description

If you are a data scientist to regularly does exploratory data analysis on new data sets, impact
charts are for you. If you are a social scientist or quantitative policy maker who regularly muddles 
through regression models to try to find out how one variable impacts another, impact charts are
for you too. 

Impact charts, as implemented in the [impactchart](https://github.com/vengroff/impactchart) package,
make it easy to take a data set and visualize the impact of one variable 
on another. Impact charts are easy to generate, easy to visually parse and understand,
and don't require any _a priori_ parametric hypotheses to uncover impact.

In this talk, we will 

- introduce impact charts with some visual examples; 
- demonstrate exactly what impact charts do by showing how they can find easter-egg impacts we embed in synthetic data;
- show how impact charts were used to find hidden impacts in a real-world use case involving race, ethnicity, income, and eviction;
- show how you can create your first impact chart on top of your own data set with just a few lines of code, using the 
  [impactchart](https://github.com/vengroff/impactchart) package;
- talk a bit (but without any gory mathematical details) about the interpretable machine learning techniques impact charts 
  are built upon and how and why they work

Only a basic understanding of data analysis tools and techniques (_i.e._ Python 3.x and pandas) is needed
to follow this talk. It will be mostly visual, because visual understanding of the impact of one variable
on another is what impact charts are all about.

## Notes

This talk will concentrate on introducing impact charts and showing the audience how they can integrate
them into their daily work. More detailed discussions and background can be found in
[this paper](https://datapinions.com/wp-content/uploads/2024/01/impactcharts.pdf)
and blog posts 
[here](https://datapinions.com/impact-chart-analysis-101/), 
[here](https://datapinions.com/an-introduction-to-impact-charts/), 
[here](https://datapinions.com/the-impact-of-demographics-and-income-on-eviction-rates/)
and the very earliest one
[here](https://datapinions.com/using-interpretable-machine-learning-to-analyze-racial-and-ethnic-disparities-in-home-values/). 
Note that some of these, particularly the earlier ones,
were written before the code was as accessible as it is today.
