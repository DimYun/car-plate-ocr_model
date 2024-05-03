install:
	pip install -r requirements.txt

train:
	DATA_PATH=data/ python train.py configs/config.yaml

lint:
	PYTHONPATH=. black train.py src
	PYTHONPATH=. nbstripout notebooks/*.ipynb
	PYTHONPATH=. tox