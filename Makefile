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

run_dl_data:
	python -c 'from baitwatch.main import download_data; download_data()'

run_preprocess:
	python -c 'from baitwatch.main import preprocess_dataset; preprocess_dataset()'

run_train:
	python -c 'from baitwatch.main import train; train()'

run_evaluate:
	python -c 'from baitwatch.main import evaluate; evaluate()'

run_cycle_fonf:
	python -c 'from baitwatch.main import run_cycle; run_cycle("fonf")'

run_report_fonf:
	python -c 'from baitwatch.main import classification_report; classification_report("fonf")'
