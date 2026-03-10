# ----------------------------------
#        PACKAGE ACTIONS
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U -e

reinstall_package:
	@pip uninstall -y baitwatch || :
	@pip install -e .

update:
	@git pull origin master
	@git merge master
	@pip install -r requirements.txt
