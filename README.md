# Transformer Translator from Scratch

A compact English-to-German neural machine translation project built with PyTorch. The Transformer encoder, decoder, attention layers, training loop, and greedy autoregressive decoding are implemented from scratch — no pretrained translation model is used.

The model is trained on the [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset.

## What is included

- Custom positional encoding
- Multi-head self-attention and causal decoder masking
- Encoder-decoder cross-attention
- Padding masks for batches with different sentence lengths
- Teacher forcing during training
- Greedy token-by-token decoding at inference time
- Saved trained weights in `en_de_transformer.pt`

## Project structure

```text
.
├── dataset.ipynb              # Data preparation, training, saving, and interactive translation
├── transformer_from_sc.ipynb  # Transformer implementation from scratch
├── en_de_transformer.pt       # Trained model checkpoint
└── requirements.txt
```

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/katr1nas/transformer-translator-from-scratch.git
cd transformer-translator-from-scratch

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open `dataset.ipynb` and run its cells from top to bottom. On the first run, the Multi30k dataset is downloaded automatically.

> `dataset.ipynb` imports the model classes from `transformer_from_sc.ipynb`, so both notebooks must remain in the same directory.

## Train the model

The training notebook builds separate English and German vocabularies, creates padded mini-batches, and trains with cross-entropy loss. The default model uses:

| Hyperparameter | Value |
| --- | ---: |
| Embedding size (`d_model`) | 256 |
| Attention heads | 8 |
| Feed-forward size | 1024 |
| Encoder / decoder layers | 4 |
| Dropout | 0.1 |
| Batch size | 32 |
| Learning rate | 1e-4 |

The notebook selects Apple Metal (`mps`) when available, then CUDA, otherwise CPU.

## Translate interactively

After training (or after loading the checkpoint), run the final cell in `dataset.ipynb` and enter an English sentence:

```text
English (exit for exit): A man is riding a bicycle
Deutsch: <model translation>
```

Type `exit` to leave the interactive prompt.

## Limitations and next steps

This is an educational implementation, so quality is limited by a small dataset and simple whitespace tokenization. Useful next improvements include:

- Save the best validation checkpoint and use early stopping
- Use `AdamW`, weight decay, and label smoothing to reduce overfitting
- Replace word-level tokenization with BPE or SentencePiece subwords
- Add BLEU evaluation on the test split
- Use beam search instead of greedy decoding

## License

This project is provided for learning and experimentation.
