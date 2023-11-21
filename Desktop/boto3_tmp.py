import boto3

s3 = boto3.client('s3')
# s3.create_bucket(Bucket='my-boto3-bucket-takkou', CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'})
# print("Bucket created")
s3.upload_file("C:/Users/George Takkou/Pictures/Screenshots/Screenshot 2023-11-06 214717.png", 'my-boto3-bucket-takkou', 'takkos/newfile.png')

