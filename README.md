# OrcaClassifier using LSTM with DVC pipelining and GitHub action
aws --no-sign-request s3 cp s3://acoustic-sandbox/labeled-data/detection/test/OrcasoundLab09272017_Test.tar.gz .
tar -xzvfOrcasoundLab09272017_Test.tar.gz
