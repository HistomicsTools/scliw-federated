#!/usr/bin/env python3
# /// script
# requires-python = '>=3.10'
# dependencies = [
#     'girder-client',
#     'torch',
#     'pandas',
#     'scikit-learn',
# ]
# ///

import argparse
import os
import sys
import tempfile
import time

import girder_client
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class CardioDataset(torch.utils.data.Dataset):
    """Lightweight PyTorch Dataset for Cardio Neural Networks."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CardioNN(nn.Module):
    """
    3-layer Feed-forward Neural Network tailored for cardiovascular disease prediction.
    Mirrors the architecture originally seen in NVFlare cardio_nvflare examples.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


class FederatedCardioClient:
    """
    Client-side federated learning worker.
    Downloads global models from the Hub Girder workspace, trains strictly on
    private locally-isolated data, and uploads updated weights back.
    Communicates with Hub via identical polling-item-queue mechanism as scliw_federated.

    INSTRUCTIONS FOR USERS:
    - You must provide a partitioned CSV file containing your local private data.
    - The --data-path argument can point to a mounted Docker volume path
      (e.g., /mnt/data/train.csv) OR a local Girder item path.
    """

    def __init__(
        self, client_id, girder_url, girder_token, hub_url, hub_token,
        data_path, work_path,
    ):
        self.client_id = client_id

        # Connect to local Girder (where private training data lives)
        self.gc_local = girder_client.GirderClient(apiUrl=girder_url)
        self.gc_local.token = girder_token

        # Connect to Hub Girder for coordination and model transport
        self.gc_hub = girder_client.GirderClient(apiUrl=hub_url)
        self.gc_hub.token = hub_token

        # Resolve the Hub workspace folder (used for polls, download/push updates)
        resp = self.gc_hub.get('resource/lookup', parameters={'path': work_path})
        if not resp:
            raise FileNotFoundError(f"Central Hub workspace '{work_path}' not found in Girder.")

        self.workspace_folder_id = resp['_id']
        self.data_path = data_path
        self.local_model_path = None
        self.epoch = 0

    def download_global_model(self, target_epoch):
        """Checks Hub workspace for the exact global model file uploaded for this round."""
        items = list(self.gc_hub.listItem(self.workspace_folder_id))
        filename_needed = f'global_model_epoch_{target_epoch}.pt'

        for item in items:
            if item['name'] == filename_needed:
                files = list(self.gc_hub.listFile(item['_id'], limit=1))
                if files:
                    target_tmp = os.path.join(tempfile.gettempdir(), f'global_{target_epoch}.pt')
                    self.gc_hub.downloadFile(files[0]['_id'], target_tmp)
                    return torch.load(target_tmp, map_location='cpu')
        raise TimeoutError(
            f'{target_epoch} global model epoch was not found in Hub workspace.')

    def upload_local_update(self, state_dict):
        path = os.path.join(
            tempfile.gettempdir(), f'weights_epoch_{self.epoch}_{self.client_id}.pt')
        torch.save(state_dict, path)

        self.gc_hub.uploadFileToFolder(
            self.workspace_folder_id,
            path,
            os.path.basename(path)
        )

    def wait_for_trigger(self):
        """Polls Hub workspace for trigger items using the scliw_federated polling pattern."""
        while True:
            try:
                items = list(self.gc_hub.listItem(self.workspace_folder_id))

                # Check if we should stop training based on Hub signal
                if any('trigger_done' in item['name'] for item in items):
                    print("[CLIENT] Received 'trigger_done' signal. Shutting down worker.")
                    sys.exit(0)

                for item in items:
                    if f'trigger_{self.epoch}' in item['name']:
                        print(f'[CLIENT] Trigger found for epoch {self.epoch}.')
                        return

                time.sleep(5.0)
            except Exception as e:
                print(f'[CLIENT] Connection error checking trigger (will retry): {e}')
                time.sleep(5.0)

    def load_data(self):
        """
        Loads data from local filesystem or local Girder
        """
        if os.path.exists(self.data_path):
            return pd.read_csv(self.data_path, sep=';')
        local_item = self.gc_local.get(
            'resource/lookup', parameters={'path': self.data_path})
        if local_item:
            files = list(self.gc_local.listFile(local_item['_id'], limit=1))
            if files:
                tmp_data = os.path.join(tempfile.gettempdir(), 'local_client_data.csv')
                self.gc_local.downloadFile(files[0]['_id'], tmp_data)
                return pd.read_csv(tmp_data, sep=';')

        raise FileNotFoundError(f'Could not find data at {self.data_path}.')

    def run_training_round(self, global_weights):
        df = self.load_data()
        y = None
        X = None
        # Automatically detect target column and features (expecting 'cardio' as label)
        if 'cardio' in df.columns:
            y = df['cardio'].values.astype(int)
            X = df.drop(columns=['cardio']).values
        else:
            # Fallback if the user's partition does not have a 'cardio' label column
            y = df.iloc[:, -1].values.astype(int)
            X = df.iloc[:, :-1].values
        X_scaled = StandardScaler().fit_transform(X)
        local_X, local_y = X_scaled, y
        device = torch.device('cpu')
        train_loader = torch.utils.data.DataLoader(CardioDataset(
            local_X, local_y), batch_size=32, shuffle=True)
        model = CardioNN(input_size=local_X.shape[1]).to(device)
        model.load_state_dict(global_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for _epoch in range(3):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        state_dict = model.state_dict()
        self.upload_local_update(state_dict)
        # Signal completion
        self.gc_hub.createItem(
            parentFolderId=self.workspace_folder_id,
            name=f'completed_{self.epoch}_{self.client_id}',
            metadata={'status': 'done'}
        )

    def run(self):
        """Main client execution loop."""
        print(f'[CLIENT] Initializing Worker ID {self.client_id} connected to Hub Girder.')

        while True:
            # Wait for the next round to begin on the Hub side
            self.wait_for_trigger()
            # Download epoch global model from Hub workspace
            global_weights = self.download_global_model(self.epoch)
            # Train strictly on local data
            self.run_training_round(global_weights)
            print(f'[CLIENT] Finished round {self.epoch}. Waiting for next Hub trigger...')
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
    worker = FederatedCardioClient(
        client_id=args.client_id,
        girder_url=args.girder_url,
        girder_token=args.girder_token,
        hub_url=args.hub_url,
        hub_token=args.hub_token,
        data_path=args.data_path,
        work_path=args.work_path,
    )
    worker.run()
