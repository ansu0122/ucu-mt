### Vast AI
### check public ssh
> .ssh % cat ~/.ssh/vsai.pub

### connect the instance from the terminal

> ssh -p 43279 root@93.99.191.159 -L 8080:localhost:8080 -i ~/.ssh/vsai

### Upload files to vast ai
> scp -r -P 43279 -i ~/.ssh/vsai . root@93.99.191.159:/root/

### Download files from vast ai
> scp -r -P 43279 -i ~/.ssh/vsai root@93.99.191.159:/root/results .


## Environment setup
To create conda env we can run
> conda env create -f conda.yaml

To activate conda env
> conda activate python3.10-la-proj

To update conda env
> conda env update --name python3.10-la-proj --file conda.yaml --prune

To add the conda env to Jupyter as a new kernel
> python -m ipykernel install --user --name python3.10-la-proj --display-name "python3.10-la-proj"

## Install
> chmod +x install.sh
> ./install.sh 

git config --global user.email "andriy.suh@gmail.com"
git config --global user.name "Andriy Sukh"

du -sh /root/.cache/huggingface/hub/*


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh

---------------------

## App Setup
1. On your Vast AI instance:
Run your FastAPI app (on any port, e.g., 7860):
> uvicorn main:app --host 127.0.0.1 --port 7860
We use 127.0.0.1 so it listens only on the internal loopback — not accessible publicly.
2. On your local machine, forward the port over SSH:
> ssh -N -L 7860:localhost:7860 root@<vast-instance-ip>
- N: don't execute commands remotely
- L 7860:localhost:7860: forward local port 7860 to remote port 7860
Replace vastuser@<vast-instance-ip> with your actual SSH login
Now you can access the FastAPI server on Vast at:
http://localhost:7860
This is securely tunneled over SSH and not exposed to the internet.
3. Run the application
> streamlit run src/client.py