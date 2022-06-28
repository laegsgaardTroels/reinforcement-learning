SHELL := /bin/zsh

envs:
	conda env create --prefix envs/reinforcement-learning

.PHONY: configure_jupyter
configure_jupyter:
	source ~/.zshrc && conda activate envs/reinforcement-learning && python -m ipykernel install --user --name reinforcement-learning
