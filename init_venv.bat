if not exist virtual_environment\ (
	python -m venv virtual_environment
	virtual_environment\Scripts\activate.bat & pip install numpy pandas matplotlib seaborn jupyter tqdm plotly statsmodels
)
virtual_environment\Scripts\jupyter notebook

