'''
What This Script Does:
Setup: Configures a SageMaker session, IAM role, and the XGBoost container.
Hyperparameters: Uses standard settings for binary classification (objective='binary:logistic') and evaluates using AUC.
Inputs: Points to train.csv and validation.csv in S3.
Output: Saves the trained model artifact to s3://your-bucket/output/<job-name>/output/model.tar.gz which can be downloaded.
'''

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
import boto3

# Set up SageMaker session
session = sagemaker.Session()
role = 'arn:aws:iam::your-account:role/SageMakerRole'  # Ensure this role has SageMaker AND S3 permissions
region = 'eu-central-1'

# Define the XGBoost container
container = sagemaker.image_uris.retrieve(framework='xgboost', region=region, version='1.7-1')

# Set up the estimator
xgboost = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.medium',
    output_path='s3://your-bucket/output/',
    sagemaker_session=session
)

# Set hyperparameters
xgboost.set_hyperparameters(
    objective='binary:logistic',  # Binary classification
    num_round=100,               # Number of boosting rounds
    max_depth=5,                 # Max tree depth
    eta=0.2,                     # Learning rate
    eval_metric='auc'            # Evaluation metric for validation
)

# Define the training and validation inputs
train_input = TrainingInput(s3_data='s3://your-bucket/processed-movies/train.csv', content_type='csv')
validation_input = TrainingInput(s3_data='s3://your-bucket/processed-movies/validation.csv', content_type='csv')

# Train the model
xgboost.fit({'train': train_input, 'validation': validation_input})
