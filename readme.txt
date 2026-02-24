1.You should first run the code in the create_train_val_data directory.Put the data in the “/Pointcept/data/teeth_land” directory.
2.Configure the environment according to the dockerfile
3.Run “sh Pointcept/scripts/train.sh -g 4 -d teeth_land -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base” to train the model.
4.After training the model, if you already have test data directly run “sh Pointcept/scripts/test.sh -g 1 -d teeth_land -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base “ to test the results. If you don't have it, you can just run process.py to process the input mesh data to construct the test data and auto-complete the test.

