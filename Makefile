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

run_preprocess_fonf:
	python -c 'from baitwatch.main import preprocess_data; preprocess_data("fonf")'

run_preprocess_ifsp:
	python -c 'from baitwatch.main import preprocess_data; preprocess_data("ifsp")'

run_augment_ifsp:
	python -c 'from baitwatch.main import save_augmented; save_augmented()'

run_train_fonf:
	python -c 'from baitwatch.main import train; train("fonf")'

run_train_ifsp:
	python -c 'from baitwatch.main import train; train("ifsp")'

run_train_ifsp_augmented:
	python -c 'from baitwatch.main import train; train("ifsp", augmented=True)'

run_evaluate_fonf:
	python -c 'from baitwatch.main import evaluate; evaluate("fonf")'

run_evaluate_ifsp:
	python -c 'from baitwatch.main import evaluate; evaluate("ifsp")'

run_cycle_fonf:
	python -c 'from baitwatch.main import run_cycle; run_cycle("fonf")'

run_cycle_ifsp:
	python -c 'from baitwatch.main import run_cycle; run_cycle("ifsp")'

run_report_fonf:
	python -c 'from baitwatch.main import classification_report; classification_report("fonf")'

run_report_ifsp:
	python -c 'from baitwatch.main import classification_report; classification_report("ifsp")'

run_api:
	uvicorn baitwatch.interfaces.api:app --reload
