import os


def create_file_in_s3(file_name, path, file):
    import boto3
    s3 = boto3.resource('s3')
    print("fileName = %s, path = %s, file = %s" % (file_name, path, type(file)))
    s3.Object('collect-data-for-machine-learning', file_name).put(Body=file)


def create_file_in_local(file_name, path, file):
    save_path = "images"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("fileName = %s, path = %s, file = %s" % (file_name, path, type(file)))
    file_path = os.path.join(save_path, file_name)
    file.save(str(file_path))


"""
bucket = s3.Bucket('collect-data-for-machine-learning')
exists = True
try:
    s3.meta.client.head_bucket(Bucket='collect-data-for-machine-learning')
except botocore.exceptions.ClientError as e:
    # If a client error is thrown, then check that it was a 404 error.
    # If it was a 404 error, then the bucket does not exist.
    error_code = int(e.response['Error']['Code'])
    if error_code == 404:
        exists = False
"""
