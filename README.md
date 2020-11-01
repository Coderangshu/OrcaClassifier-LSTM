# OraClassifier using LSTM with DVC pipelining and GitHub action
<br><br>An approach is made to classify orca calls using recurrent neural networks instead of conventional convolutional neural networks with CI/CD in action using cml.
- tar.gz files are downloaded from remote storage which is data versioned by DVC
- dvc.yaml executed using `dvc repro` this runs the complete pipeline of execution from extracting tar files to their preprocessing and model training.
