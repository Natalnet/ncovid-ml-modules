# NCovid ML Modules
NCovid ML Modules is a standalone library for machine learning applications, compatible with receiving requests as web-point.

> The NCovid-ml-modules design goal is improving formatting and facilitating is
> to make ml pipelines readable and optimized. The idea is that a time-series dataframe 
> or a table data should be preprocessed and submitted to the use of 
> many applications to data regression problems.

## Tech
NCovid ML Modules uses a number of open source projects to work properly:

- [Python 3][Python] - a powerful general-purpose programming language
- [Pandas][Pandas] - fast, powerful, flexible and easy to use open source data analysis and manipulation tool
- [NumPy][NumPy] - comprehensive mathematical functions
- [Keras][Keras] - an open-source software library that provides a Python interface for artificial neural networks
- [Matplotlib][Matplotlib] - plotting library

And of course NCovid ML Modules itself is open source with a [public repository][ncovid] on GitHub.

This code runs on Python 3.7

## Installation

### Prerequisites

We have tested the library in Ubuntu 20.04, 19.04, 18.04, and 16.04, but it should be easy to compile on other platforms.
### Libraries


Please make sure that it has installed all the required dependencies. 
A list of items to be installed using pip install can be running as following:
```
pip install -r requirements.txt
```

## Folder structure
>Project Folder Structure and Files

* [src](src) : main Python package with source of the model.
* [dbs](src/dbs/) : used as a local path to store and load data and models.
* [docs](src/docs/) : contains documentation of the project.
* [jupyter-notebook](src/jupyter-notebook/) : contains jupyter notebooks evaluation and modeling experimentation.


## Usage


## Stakeholders
> People involved in this project

| Role                 | Responsibility         | Full name                | orcid        |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Scrum Master           | Davi Santos              | [ORCID][DAVI_ORCID]|
| Data Scientist       | Tech Leader            | Dunfrey P. Aragão        | [ORCID][DUNFREY_ORCID]|
| Data Scientist       | Developer              | Emerson Vilar            | [ORCID][EMERSON_ORCID]|

## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [ncovid]: <https://github.com/Natalnet/ncovid-ml-modules>
   [Python]: <https://www.python.org/>
   [Pandas]: <https://pandas.pydata.org/>
   [NumPy]: <https://numpy.org/>
   [Keras]: <https://keras.io/>
   [Matplotlib]: <https://matplotlib.org/>

   [DAVI_ORCID ]: <>
   [DUNFREY_ORCID]: <orcid.org/0000-0002-2401-6985>
   [EMERSON_ORCID]: <>
