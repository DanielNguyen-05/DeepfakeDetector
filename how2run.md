# Step 1: Create and activate the environment

```bash
python -m venv .venv
source .venv/bin/activate
```

# Step 2: Install the suitable libraries:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Step 3: Train the model

## 3.1: If you want to train the model 

```bash
python test.py --cfg configs/results_cifake_T_unfrozen.cfg
```

## 3.2: If you want to train it in background process
```bash
nohup python -u train.py --cfg configs/results_cifake_T_unfrozen.cfg > training_log.txt 2>&1 &
```

track the GPU and training process

```bash
nvidia-smi
tail -f training_log.txt
```

get the PID (if you want to stop the training)

```bash
ps -ef | grep train.py
```

force to stop the training (change the PID, default 1234567)

```bash
kill -9 1234567
```

## 3.1: If you want to test the pretrained model (change the config file)

```bash
python train.py --cfg configs/results_cifake_T_unfrozen.cfg
```