# server.py

from concurrent import futures
import os
import time
import uuid
import grpc
import basic_pb2
import basic_pb2_grpc
from tqdm import tqdm
from MLP.MLP import MLP
from predict import Predict

class File(basic_pb2_grpc.FileServicer):

    def upload(self, request, context):
        if not os.path.isdir('./temp'):
            os.makedirs('./temp')
        guid = str(uuid.uuid1()).split('-')[0]
        for r in request:
            if not os.path.isdir('./temp/{}'.format(guid)):
                os.makedirs('./temp/{}'.format(guid))
            with open('./temp/{}/{}'.format(guid, r.name), 'ab') as f:
                f.write(r.file)
        return basic_pb2.Response(status='success', result=guid)

class Train(basic_pb2_grpc.TrainServicer):

    def train(self, request, context):
        guid = request.guid
        print(guid)
        m = MLP()
        mlpPath = m.mlp(guid)
        print(mlpPath)
        with open(mlpPath, 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.TrainResult(file=line)

class PredictResult(basic_pb2_grpc.PredictResultServicer):

    def predictResult(self, request, context):
        if not os.path.isdir('./PredictDir'):
            os.makedirs('./PredictDir')
        guid = ''
        for r in request:
            if not os.path.isdir('./PredictDir/{}'.format(r.guid)):
                os.makedirs('./PredictDir/{}'.format(r.guid))
            if guid == '':
                guid = r.guid
            with open('./PredictDir/{}/{}'.format(r.guid, r.name), 'ab') as f:
                f.write(r.file)
        p = Predict()
        predictPath = p.predict(guid)

        with open(predictPath, 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                yield basic_pb2.PredictResultFile(file=line)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    basic_pb2_grpc.add_FileServicer_to_server(File(), server)
    basic_pb2_grpc.add_TrainServicer_to_server(Train(), server)
    basic_pb2_grpc.add_PredictResultServicer_to_server(PredictResult(), server)
    server.add_insecure_port('[::]:80')
    server.start()

    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()