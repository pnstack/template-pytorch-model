package:
	pip freeze > requirements.txt
venv:
	source /mnt/d/ubuntu/env/mlenv/bin/activate
tensorboard:
	tensorboard --inspect --logdir logs/tensorboard