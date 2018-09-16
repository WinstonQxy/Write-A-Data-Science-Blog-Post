# ___Write-A-Data-Science-Blog-Post
Udacity Data Science Nanodegree Project - Write A Data Science Blog Post


### Table of Contents - Changed

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using Airbnb data for Brussels, in order to better understand:

   - 1. Renting for a longer period is always cheaper? - When traveling for a longer period we are quite used to having a discount for a longer stay. We try to find out if the same trend can be seen for the properties rented in Brussels.
   - 2. Where can I rent a property? - With 19 different neighborhoods, having a strong link to both the French Community and the Flemish Community, when you want to rent a property is quite interesting to know which are the main profiles for the neighborhoods where a property is available.
   - 3. Which are the main amenities? - The list of the possible amenities that a property can have contains a catalog of 120 different items. We will try to find the main profiles that we can find in Brussels.
   - 4. Which are the main profiles for the people that propose to rent a property in Bruxelles?
   - 5. Are there groups that have the same characteristics? - For the properties available in Brussels it would be nice to group them together by similarities and find the most import feature bringing them together, or setting them apart.

The input files can be publicly downloaded at the following address:
http://insideairbnb.com/get-the-data.html


## File Descriptions <a name="files"></a>

The following files are available:
   - 1. a Jupyter Notebook in which all the above questions are being answered
   - 2. util.py - utility file containing procedure for the visualization, price computation, PCA execution, etc
   - 3. BruxellesNeighbourhoods.csv - a csv file mapping the correct neighbourhoods for Brussels
   - 4. mapCountry.json - a json file mapping the Host_Location_Country
   - 5. mapRegion.json - a json file mapping the Host_Location_State

## Results<a name="results"></a>

The main findings of the code can be found at the following post:
https://medium.com/@lisaro1982/brussels-airbnb-where-how-what-groups-413186e040f3

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to insideairbnb for the data and Udacity.
