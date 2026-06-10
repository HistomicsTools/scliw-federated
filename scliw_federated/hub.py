#!/usr/bin/env python3
# /// script
# requires-python = '>=3.13'
# dependencies = [
#     'girder-client',
#     'xgboost',
# ]
# ///

import argparse
import os
import shutil
import tempfile
import time

import girder_client
import xgboost as xgb


class HubCoordinator:
    def __init__(self, girder_url, girder_token, work_path, epochs, num_clients):
        """
        Hub code
        """
        self.work_path = work_path
        self.epochs = epochs
        self.num_clients = num_clients
        self.gc = girder_client.GirderClient(apiUrl=girder_url)
        self.gc.token = girder_token
        self.workspace = self.gc.get('resource/lookup', parameters={'path': work_path})
        self.current_model = None
        self.tmpdir = tempfile.mkdtemp()
        self.epoch = 0

    def initial_model(self):
        params = {
            'max_depth': 5,
            'eta': 0.2,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
        dummy_data = [[0.0] * 8]  # this has to match the number of columns
        dtrain = xgb.DMatrix(dummy_data, label=[0])
        empty_model = xgb.train(params, dtrain, num_boost_round=0)
        new_model = os.path.join(self.tmpdir, f'model_epoch_{self.epoch}.json')
        empty_model.save_model(new_model)
        self.gc.uploadFileToFolder(
            self.workspace['_id'],
            new_model,
            new_model.split('/')[-1],
        )

    def wait_for_completions(self):
        completed = 0
        max_wait = time.time() + 3600
        while completed < self.num_clients and time.time() < max_wait:
            items = list(self.gc.listItem(self.workspace['_id']))
            completed = sum(1 for item in items if f'completed_{self.epoch}' in item['name'])
            if completed < self.num_clients:
                time.sleep(5)
        return completed == self.num_clients

    def merge_models(self, model_paths, output_path):
        # Use a round-robin approach to sequential train from each node
        # this is wasteful but demonstrates the process
        shutil.copy(model_paths[self.epoch % len(model_paths)], output_path)

    def aggregate_models(self):
        model_items = list(self.gc.listItem(self.workspace['_id']))
        model_data = []
        for item in model_items:
            if f'model_update_{self.epoch}' in item['name']:
                files = list(self.gc.listFile(item['_id'], limit=1))
                model_path = os.path.join(self.tmpdir, f'{item["name"]}')
                self.gc.downloadFile(files[0]['_id'], model_path)
                model_data.append(model_path)
        new_model = os.path.join(self.tmpdir, f'model_epoch_{self.epoch + 1}.json')
        self.merge_models(model_data, new_model)
        self.gc.uploadFileToFolder(
            self.workspace['_id'],
            new_model,
            new_model.split('/')[-1],
        )
        return new_model

    def create_trigger(self):
        self.gc.createItem(
            parentFolderId=self.workspace['_id'],
            name=f'trigger_{self.epoch}' if self.epoch < self.epochs else 'trigger_done',
            metadata={'epoch': self.epoch}
        )

    def run(self):
        print(f'Starting federated learning for {self.epochs} epochs')
        self.initial_model()
        for epoch in range(self.epochs):
            self.epoch = epoch
            print(f'Epoch {self.epoch + 1}/{self.epochs}')

            self.create_trigger()
            if not self.wait_for_completions():
                print(f'Warning: Not all clients completed epoch {self.epoch + 1}')
            self.current_model = self.aggregate_models()
        self.epoch = self.epochs
        self.create_trigger()
        print('Federated learning completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1')
    parser.add_argument('--girder-token', required=True)
    parser.add_argument('--work-path', required=True,
                        help='Hub Girder path to work folder')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to run')
    parser.add_argument('--clients', type=int, default=4,
                        help='Number of clients to expect')
    args = parser.parse_args()
    hub = HubCoordinator(
        girder_url=args.girder_url,
        girder_token=args.girder_token,
        work_path=args.work_path,
        epochs=args.epochs,
        num_clients=args.clients
    )
    hub.run()
