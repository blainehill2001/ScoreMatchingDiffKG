
## Project: Link Prediction via Score Matching
### Authors: Blaine Hill, Ruizhong Qiu, Hanghang Tong
### Spring 2024

## Comments and Asides

* When using pyproject.toml to configure your git hooks (such as when using pre-commit), ensure you run this command: `pip install Flake8-pyproject` so that [flake8](https://flake8.pycqa.org/en/latest/) can parse the TOML file.
* We make use of [pipreqsnb](https://pypi.org/project/pipreqsnb/) to generate our requirements.txt file. Take care to resolve any package versions it generates due to guesses (since sometimes it resolves the package version via PyPI).
* To run the ipynb files on a UIUC server, install the packages with pip and use conda to install jupyter and use `jupyter notebook --no-browser & disown` to run it in the background. Ensure you cancel the process later.
* Initialize pre-commit via `pre-commit install` in your home dir in order to utilize it.
