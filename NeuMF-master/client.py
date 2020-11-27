# client.py

import grpc
import basic_pb2
import basic_pb2_grpc
from tqdm import tqdm


def sendfile():
    channel = grpc.insecure_channel('localhost:80')

    stub = basic_pb2_grpc.FileStub(channel)
    def readfile():
        with open('Data/ml-1m.test.negative', 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.FileBody(file=line,name='test.negative')
        with open('Data/ml-1m.test.rating', 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.FileBody(file=line,name='test.rating')
        with open('Data/ml-1m.train.rating', 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.FileBody(file=line,name='train.rating')
    response = stub.upload(readfile())

    print('Upload Return: {}'.format(response))

# sendfile()

def train(guid):
    channel = grpc.insecure_channel('localhost:80')
    stub = basic_pb2_grpc.TrainStub(channel)
    response = stub.train(basic_pb2.GuidInfo(guid=guid))
    with open('./mlp.h5', 'ab') as f:
        for r in response:
                f.write(r.file)
        print('Successful acquisition model')      

# train('d0d15790')

def predict(guid):
    channel = grpc.insecure_channel('localhost:80')

    stub = basic_pb2_grpc.PredictResultStub(channel)
    def readfile():
        with open('Data/ml-1m.test.negative', 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.PredictBody(file=line,name='test.negative', guid=guid)
        with open('Data/ml-1m.test.rating', 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.PredictBody(file=line,name='test.rating', guid=guid)
        print('Upload completed, please wait for forecast result!')
    response = stub.predictResult(readfile())   
    with open('./{}.csv'.format(guid), 'ab') as f:
        for r in response:
            f.write(r.file)
        print('Successful!')      

predict('d0d15790')