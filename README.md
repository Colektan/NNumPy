### Train the model.

Open test_train.py, modify parameters in the first few lines of the file and run it. You may need to change the `data_dir` if the data is stored in a different path.
 
If you want to train the model on your own dataset, you need to write up the `Dataset` class of your own dataset with `get_idx()` method defined.

### Test the model.

Open test_model.py, specify the saved model's path in `model_path` and the test dataset's path in `data_dir`, then run the script, the script will output the accuracy on the test dataset.



