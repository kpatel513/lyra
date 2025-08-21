
Please generate a profiling report using the PyTorch Advanced Profiler for the training script specified in $ARGUMENTS using pytorch advanced profiler.

Insert a lightweight profiling mode into this PyTorch Lightning training script. 
Cap training with max_steps (counting optimizer steps).
Disable sanity validation (num_sanity_val_steps=0).
Skip validation and test entirely (limit_val_batches=0, limit_test_batches=0).
Bound training batches (limit_train_batches = max_steps * accumulate_grad_batches).
Turn off extras (enable_checkpointing=False, logger=False, enable_progress_bar=False).
After .fit(), skip .test() and .predict().
Ensure the normal path is unchanged when profiling is off. Insert the changes concisely inside create_trainer and train_and_evaluate.


Follow these steps:
Modify the training code $TRAINING_SCRIPT to include AdvancedProfiling support using this class 
1) classlightning.pytorch.profilers.AdvancedProfiler(filename='lyra-profile.txt')
2) Remove the max_epochs setting from the lightning module definition if profiling is enabled
3) Limit the training to 5 steps and disable max_epochs setting
3) Pass the advanced profiler to the trainer, make sure max_epochs is not set
4) Do not make any code changes before walking me through the logic step by step
4) Run the training script with the pytorch Advanced profiler
