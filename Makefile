# ----------------------------------
#        PACKAGE ACTIONS
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install -e . -U

reinstall_package:
	@pip uninstall -y baitwatch || :
	@pip install -e .

update:
	@git pull origin master
	@git merge master
	@pip install -r requirements.txt

run_preprocess:
	python -c 'from baitwatch.main import preprocess_dataset; preprocess_dataset()'
