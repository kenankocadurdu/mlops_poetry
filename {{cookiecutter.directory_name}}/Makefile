install:
	@echo "Install dependencies..."
	poetry lock --no-update
	poetry run pre-commit install
	git add .
	git commit -m "poetry lock"


initialize_git:
	@echo "Initializing git"
	git init -b main
	git init && git symbolic-ref HEAD refs/heads/main
	git add .
	git commit -m "initial commit"
	git remote add origin https://github.com/kenankocadurdu/mlops.git
	git remote -v
	git push origin main

setup: initialize_git install
