### Code for Phase of firing does not reflect temporal order in sequence memory of humans and recurrent neural networks
[Link to preprint](https://www.biorxiv.org/content/10.1101/2022.09.25.509370v1)


### Example usage
Install the conda environment and train an RNN model:
```
cd sequence-memory
conda env create -f sequence.yml
activate sequence
python rnn_model/rnn/run_single_model.py
```
Expected behavior: A trained RNN model performing a working-memory task should be obtained and saved. 

Expected run-time: One single RNN model took on average around 5-6 hours to train on a Nvidea 2080-ti GPU.

### Reproducing the paper Figures
Pull the model and data files from this repo, by first installing [git lfs](https://git-lfs.com/).
Alternatively, retrain new RNNs using
```
python rnn_model/rnn/run_single_model.py
```
or
```
wandb sweep rnn_model/rnn/sweep.yml
rnn_model/rnn/run_sweep.py # after adding sweep ID to top of file
```
and obtain summary statistics over multiple models using
```
rnn_model/rnn/run_summary.py
```

Either-way, the figures can then be recreated by running the notebooks in: `rnn_model/generate_figures`.

### Note
Code tested on a Ubuntu system with package versions given in the `environment.yml` file

