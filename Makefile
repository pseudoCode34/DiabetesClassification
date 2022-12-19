init:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
	bash ~/miniconda.sh -b -p -f $HOME/miniconda
	#Setup a new conda environment
	source $HOME/miniconda/bin/activate
	# Activate a virtual environment from (base)
	conda create -n diabetes-dianogsis-env
	conda activate diabetes-dianogsis-env
	# Install necessary packages
	conda install --file requirements.txt

test:
	nosetests tests

.PHONY: init test
