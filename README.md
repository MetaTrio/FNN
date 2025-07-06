This is a FNN-based host depletion and classification tool capable of being trained on different host genomes. This tool allows users to train models from scratch,  deplete host reads from metagenomic samples, and evaluate classification performance.

## 📁 Project Structure

| File / Script | Description |
|---------------|-------------|
| `binary_train_val_diff_datasets_balanced.py.py` | Train model from scratch using separate training/validation sets |
| `binary_classification_imbalanced.py` | Run host depletion on a metagenomic dataset |
| `binary_evaluation.py` | Evaluate host depletion


## Usage Notes for AMAISE-MultiHost (CNN)

To train a new model for a new host, use the command
```bash
python3 binary_train_val_diff_datasets_balanced.py -ti <train_fasta> -tl <train_labels_csv> -vi <val_fasta> -vl <val_labels_csv> -m <output_model_path> -o <output_log_path> 
```

For host depletion, use the command
```bash
python3 binary_classification.py -i <input_file> -t <file_type> -k <kmer_input_file> -m <trained_model_path> -o <output_folder>
```
(File type is either fastq or fasta)

For evaluation, use the command
```bash
python3 evaluation_binary_true_pred_lables_no_clash.py -p <path_to_predictions/mlprobs.txt> -t <true_labels.csv> -o <output_log>
python3 binary_evaluation.py -p <path_to_predictions> -t <true_labels.csv> -lf <output_log> 
```
