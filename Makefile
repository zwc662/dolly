VENV=venv
# Recommended ${CUDA_VERSION}=11.7.0
venv:
	conda create -n ${VENV} python=3.9
	conda activate ${VENV}

cuda:
	conda install -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-nvcc cuda-toolkit libcusparse-dev libcusolver-dev 
	ln -s ${CONDA_PREFIX}/lib/libcudart.so /usr/lib/libcudart.so
	ln -s ${CONDA_PREFIX}/lib/libcudart_static.a /usr/lib/libcudart.a
	ln -s ${CONDA_PREFIX}/lib/libcurand.so /usr/lib/libcurand.so
	ln -s ${CONDA_PREFIX}/lib/libcurand_static.a /usr/lib/libcurand.a
lib:
	python -m pip install tensorboard jupyter 
	python -m pip install -r requirements.txt
	#python -m pip install torch==1.13 --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116


