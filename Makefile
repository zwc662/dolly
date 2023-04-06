CUDAR_VERSION=


venv:
	conda create -n $(VENV) python=3.9
	conda activate $(VENV)
	python -m pip install -r requirement.txt
	python -m pip install tensorboard 
	conda install -c "nvidia/label/cuda-$(CUDA_VERSION)" cuda-nvcc cuda-toolkit libcusparse-dev libcusolver-dev 
	ln -s $(CONDA_PREFIX)/lib/libcudart.so /usr/lib/libcudart.so
	ln -s $(CONDA_PREFIX)/lib/libcudart.a /usr/lib/libcudart.a
	ln -s $(CONDA_PREFIX)/lib/libcurand.so /usr/lib/libcurand.so
	ln -s $(CONDA_PREFIX)/lib/libcurand.a /usr/lib/libcurand.a
	python -m pip install torch==1.13 --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
	python -m pip install torchvision


