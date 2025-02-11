# Recommender System

This repository contains a recommender system written in Python for the subject TDT4215.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project implements a recommender system that provides personalized recommendations based on user preferences and behavior. The system uses various algorithms to analyze data and generate recommendations.

## Installation

Before proceeding, ensure you have **Python `3.12.5`** installed. You can verify your Python version by running:

```bash
python --version
```

If you do not have Python 3.12.5 installed, please download it from the [official Python website](https://www.python.org/downloads/).

Alternatively, you can install Python 3.12.5 using [pyenv](
    https://github.com/pyenv/pyenv
) and [pyenv-win](https://github.com/pyenv-win/pyenv-win) for Windows.

It is recommended that you use a virtual environment, such can be instantiated by running the following command:
```bash
python -m venv venv
```

Activate the virtual environment by running the following command:
```bash
source venv/bin/activate
```

Once the virtual environment is activated, you can install the necessary dependencies.

To install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used in this project is the MIND dataset, which is a large-scale dataset for news recommendation. The dataset contains user interactions with news articles, and it is used to train and evaluate the recommender system. To run the project, you need to download the dataset from the [MIND dataset website](https://msnews.github.io/). Place the dataset in the `data` directory.

## Usage
To run the recommender system, execute the following command:
```bash
python main.py
```
Make sure to configure the necessary parameters in the `config.py` file before running the system.

## Contributing
Feel free to fork/clone the repository and create a pull request with your changes. If you encounter any issues, please create an issue in the repository.

## License
This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.