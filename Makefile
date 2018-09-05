check-gpu:
	docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

run-mx:
	./ndrun.sh -t mxnet

run-tf:
	./ndrun.sh -t tensorflow

run-cntk:
	./ndrun.sh -t cntk

pull-mx:
	docker pull honghu/keras:mx-latest

pull-tf:
	docker pull honghu/keras:tf-latest

pull-cntk:
	docker pull honghu/keras:cntk-latest