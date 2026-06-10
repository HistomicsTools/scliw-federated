#!/usr/bin/env python3
# /// script
# requires-python = '>=3.13'
# dependencies = [
#     'girder-client',
#     'pandas',
#     'xgboost',
# ]
# ///

import argparse
import os
import tempfile
import time

import pandas as pd
import xgboost as xgb
from girder_client import GirderClient


class ClientWorker:
    def __init__(self, client_id, girder_url, girder_token, hub_url,
                 hub_token, data_path, work_path):
        """
        Initialize client worker.
        """
        self.client_id = client_id
        self.data_path = data_path
        self.work_path = work_path
        self.gc_local = GirderClient(apiUrl=girder_url)
        self.gc_local.token = girder_token
        self.gc_hub = GirderClient(apiUrl=hub_url)
        self.gc_hub.token = hub_token
        self.workspace = self.gc_hub.get('resource/lookup', parameters={'path': work_path})
        self.tmpdir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.tmpdir, f'client_model_{client_id}.json')
        self.epoch = 0

    def get_data(self):
        """Get training data from local Girder."""
        data_item = self.gc_local.get('resource/lookup', parameters={'path': self.data_path})
        if data_item:
            files = list(self.gc_local.listFile(data_item['_id'], limit=1))
            if files:
                path = os.path.join(self.tmpdir, 'data.csv')
                self.gc_local.downloadFile(files[0]['_id'], path)
                return pd.read_csv(path)
        return None

    def get_model(self):
        """Download current model from hub."""
        items = list(self.gc_hub.listItem(self.workspace['_id']))
        for item in items:
            if f'model_epoch_{self.epoch}' in item['name']:
                files = list(self.gc_local.listFile(item['_id'], limit=1))
                self.gc_hub.downloadFile(files[0]['_id'], self.model_path)
                return self.model_path
        return None

    def train_model(self, df: pd.DataFrame, epochs=2, previous_model=None):
        """Train XGBoost model on local data."""
        target = 'diabetes'
        if 'diabetes' not in df.columns and 'Disease' in df.columns:
            target = 'Disease'
        y = df[target]
        X = df.drop(target, axis=1)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'max_depth': 5,
            'eta': 0.2,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'subsample': 0.6,
            'colsample_bytree': 0.8
        }
        bst = xgb.train(params, dtrain, num_boost_round=epochs, xgb_model=previous_model)
        return bst

    def upload_model(self, model):
        """Upload model to hub."""
        model_path = os.path.join(self.tmpdir, f'model_update_{self.epoch}_{self.client_id}.json')
        model.save_model(model_path)
        self.gc_hub.uploadFileToFolder(
            folderId=self.workspace['_id'],
            filepath=model_path,
            filename=model_path,
        )

    def mark_completion(self):
        """Mark task as completed on hub."""
        self.gc_hub.createItem(
            parentFolderId=self.workspace['_id'],
            name=f'completed_{self.epoch}_{self.client_id}',
            metadata={'status': 'completed'}
        )

    def wait_for_trigger(self):
        """Wait for new training trigger."""
        while True:
            items = list(self.gc_hub.listItem(self.workspace['_id']))
            for item in items:
                if f'trigger_{self.epoch}' in item['name']:
                    return True
                if 'trigger_done' in item['name']:
                    return False
            time.sleep(5)
        return False

    def run(self):
        """Main execution loop."""
        print(f'Client {self.client_id} starting...')
        while self.wait_for_trigger():
            df = self.get_data()
            if df is None:
                continue
            model_path = self.get_model()
            model = self.train_model(df, previous_model=model_path)
            self.upload_model(model)
            self.mark_completion()
            print(f'Client {self.client_id} completed round')
            self.epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True, help='Client identifier')
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1',
                        help='Local Girder URL')
    parser.add_argument('--girder-token', required=True,
                        help='Local Girder authentication token')
    parser.add_argument('--hub-url', default='http://hub.example.com/api/v1',
                        help='Hub Girder URL')
    parser.add_argument('--hub-token', required=True,
                        help='Hub authentication token')
    parser.add_argument('--data-path', required=True,
                        help='Local Girder path to csv data item')
    parser.add_argument('--work-path', required=True,
                        help='Hub Girder path to work folder')

    args = parser.parse_args()

    client = ClientWorker(
        client_id=args.client_id,
        girder_url=args.girder_url,
        girder_token=args.girder_token,
        hub_url=args.hub_url,
        hub_token=args.hub_token,
        data_path=args.data_path,
        work_path=args.work_path,
    )
    client.run()
