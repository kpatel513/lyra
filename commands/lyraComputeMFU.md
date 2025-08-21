Please create a pytorch lightning callback to measure FLOPS for the training script specified in $ARGUMENTS
Follow these steps:
1. Write a callback to measure FLOPS required for a training step following the example in https://github.com/Lightning-AI/pytorch-lightning/pull/20868/files#diff-e781c92f7e07c3d06866889ddcde21eb1c88d412aab109613eec1eca8504883e
2. Use the throughput information being logged in the ThroughputCallback to compute Model Flop Utilization
3. Log the model flop utilization 
