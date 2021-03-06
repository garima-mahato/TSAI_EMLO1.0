import argparse
import os

# default pytorch import
import torch

# import lightning library
import pytorch_lightning as pl

# import trainer class, which orchestrates our model training
from pytorch_lightning import Trainer

# import our model class, to be trained
from CIFAR100Classifier import CIFAR100Classifier

# This is the main method, to be run when train.py is invoked
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpus', type=int, default=1) # used to support multi-GPU or CPU training

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-dr','--dir', type=str, default=os.environ['SM_CHANNEL'])
    #parser.add_argument('-te','--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print(args)

    # Now we have all parameters and hyperparameters available and we need to match them with sagemaker 
    # structure. default_root_dir is set to out_put_data_dir to retrieve from training instances all the 
    # checkpoint and intermediary data produced by lightning
    cifar100Resnet34Trainer=pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, default_root_dir=args.output_data_dir)

    # Set up our classifier class, passing params to the constructor
    model = CIFAR100Classifier(
        batch_size=args.batch_size, 
        data_dir=args.dir
        )
    
    # Runs model training 
    cifar100Resnet34Trainer.fit(model)

    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, 'cifar100Resnet34Model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)