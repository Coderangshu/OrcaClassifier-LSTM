# OrcaClassifier using LSTM with DVC pipelining and GitHub action
<br><br>An approach is made to classify orca calls using recurrent neural networks instead of conventional convolutional neural networks.

aws --no-sign-request s3 cp s3://acoustic-sandbox/labeled-data/detection/test/OrcasoundLab09272017_Test.tar.gz .
tar -xzvfOrcasoundLab09272017_Test.tar.gz
