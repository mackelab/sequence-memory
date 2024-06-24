### Code for Phase of firing does not reflect temporal order in sequence memory of humans and recurrent neural networks
[Link to preprint](https://www.biorxiv.org/content/10.1101/2022.09.25.509370v1)


### example usage
Install the conda environment and train an RNN model:
```
cd sequence-memory
conda env create -f sequence.yml
activate sequence
python rnn_model/rnn/run_single_model.py
```
### Reproducing the paper Figures
One first needs [git lfs](https://git-lfs.com/) installed in order to pull the model files.
The figures can then be recreated by running the notebooks in: `rnn_model/generate_figures`.
