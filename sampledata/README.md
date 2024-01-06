# Sample Data

This directory contains sample data for demonstrating the creation
of impact charts.

## 1. Eviction Data for Hudson County, NJ

File: `evl-34017.csv`

This is a combination of eviction data from the Eviction Lab at Princeton University
and U.S. Census demographic and income data. It was produced by the 
[evldata](https://github.com/datapinions/evldata) project.

Each row contains data for a particular tract in a particular year. The columns with
features we are
interested for building impact charts are `B25119_003E_2018`, which is the median income
of renters in the tract in 2018 dollars and the columns that begin with `frac_B25003[X]_`,
which represent the fraction of renters in living in the tract that identify as belonging
to each of various racial or ethnic groups. See https://api.census.gov/data/2018/acs/acs5/groups/B25119.html
for more on income and https://api.census.gov/data/2022/acs/acs5/groups/B25003A.html, 
https://api.census.gov/data/2022/acs/acs5/groups/B25003B.html, 
https://api.census.gov/data/2022/acs/acs5/groups/B25003C.html, etc... on race and ethnicity of
renters for more details
on the original sources of this data and their interpretations.

See also the [evldata](https://github.com/datapinions/evldata) project for more on
how the raw census data and Eviction Lab data were combined and this data set was generated.