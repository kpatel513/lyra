Please create a pytorch lightning callback to measure throughput in tokens/sec for the training script specified in $ARGUMENTS
Follow these steps:
1. Carefully study the training script downstream_fine_tune.py and associated dataloader and generate a summary of the steps involved.
2. Write a callback to measure the training and inference throughput in tokens/sec, taking into account batch size.
3. Make sure all this information is being logged appropriately in the progress bar on the command line for progress tracking, while the model is training, not just at the end.
4. Make this feature configurable.

