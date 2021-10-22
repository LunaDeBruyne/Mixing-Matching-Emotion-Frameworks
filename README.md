# Mixing-Matching-Emotion-Frameworks
Code for the paper "Mixing and matching emotion frameworks: Investigating cross-framework transfer learning for Dutch emotion detection" in Electronics.

REQUIREMENTS:
- Python 3.6
- PyTorch with support for CUDA10.1
- pandas
- sklearn
- scipy
- matplotlib
- transformers


USAGE:
- The four folders in this project correspond to the four experimental settings in the paper: 1) the base model (Dutch transformer model RobbBERT), 2) multi-task learning, 3) the meta-learner and 4) the pivot method. For the base model and multi-task model, the file to run the scripts is pipeline.py. The folders of the meta-learner and pivot method only contain 1 file. However, they rely on the predictions of the base model (which should be saved when running).

- Your input files (gold data) should contain three named columns: id (a given sentence ID), text (the actual text), label (the correct label index for the given sentence). By default, the script expects a tab-separated file.

- Change the values in the pipeline.py files and in config_cat_default.json / config_vad_default.json / config_mt_default.json.
