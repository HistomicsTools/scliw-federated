#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "girder-client==2.4.0",
#     "torch>=2.0.0",
#     "pandas>=2.0.0",
#     "scikit-learn>=1.3.0",
#     "nvflare>=2.6.0",
# ]
# ///

import argparse
import os
import sys
import tempfile
import time
from typing import Dict, Optional

import girder_client
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

try:
    from .nvflare_bus import GirderEventBus
except Exception:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from nvflare_bus import GirderEventBus


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


class FedClientWorker:
    """
    A Federated Learning client strictly bound to local CSV data and
    synchronized over Girder via our custom Event Bus.
    """

    def __init__(self, client_id: str, hub_url: str, hub_token: str, workspace_path: str, data_path: str,
                 local_girder_url: str = None, local_girder_token: str = None):
        self.client_id = client_id
        print(f"[CLIENT] Initializing worker {client_id} connected to Hub Girder.")

        # Initialize our Girder-based event bus for communication
        self.bus = GirderEventBus(hub_url, hub_token, workspace_path)
        self.data_path = data_path
        self.epoch = 0

        # Store credentials for potential local Girder access on the spoke (client) side
        self.local_girder_url = local_girder_url
        self.local_girder_token = local_girder_token
        self.gc_local = None
        if self.local_girder_url:
            self.gc_local = girder_client.GirderClient(apiUrl=self.local_girder_url)
            if self.local_girder_token:
                self.gc_local.token = self.local_girder_token

    def load_model(self, global_weights: Dict[str, torch.Tensor]) -> nn.Module:
        model = CardioNN(input_size=next(iter(global_weights.values())).shape[1])
        if global_weights:
            model.load_state_dict(global_weights)
        return model

    def train_epoch(self, model: nn.Module, local_X: np.ndarray, local_y: np.ndarray) -> Optional[Dict]:
        """Executes localized training and returns the updated state dict."""
        train_loader = torch.utils.data.DataLoader(
            list(zip(local_X, local_y)), batch_size=32, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(3): # Fixed internal epochs per federated round
            model.train()
            for batch_X, batch_y in train_loader:
                X_batch = torch.tensor(batch_X[0], dtype=torch.float32)
                y_batch = torch.tensor(batch_y[1], dtype=torch.long)

                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def run(self):
        """Main client execution loop. Uses our Girder Bus for orchestration."""
        while True:
            try:
                task = self.bus.subscribe_item("global_model_epoch")
                if task['status'] == 'shutdown':
                    print("[CLIENT] Received shutdown signal from Hub.")
                    sys.exit(0)

                expected_epoch = task['epoch']
                print(f"[CLIENT] Epoch {expected_epoch} triggered. Downloading global model...")

                # Fetch the weights (The Hub pushes them via our bus logic)
                new_weights = self.bus.fetch_model(client_id='hub', expected_epoch=expected_epoch)
                if not new_weights:
                    raise ValueError("No global weights found in bus.")

                # Inject data from local CSV into the NVFlare-compatible model layer
                if os.path.exists(self.data_path):
                    df = pd.read_csv(self.data_path, sep=';')
                elif self.gc_local:
                    # Attempt to fetch via the client's local Girder instance if not a local file
                    try:
                        item = self.gc_local.get('resource/lookup', parameters={'path': self.data_path})
                        files = list(self.gc_local.listFile(item['_id'], limit=1))
                        tmp_data = os.path.join(tempfile.gettempdir(), 'local_client_data.csv')
                        self.gc_local.downloadFile(files[0]['_id'], tmp_data)
                        df = pd.read_csv(tmp_data, sep=';')
                    except Exception as e:
                        raise FileNotFoundError(f"Failed to fetch data '{self.data_path}' from local Girder: {e}")
                else:
                    raise FileNotFoundError(f"Data '{self.data_path}' not found locally and no local Girder credentials provided.")

                y = df['cardio'].values.astype(int)
                X = StandardScaler().fit_transform(df.drop(columns=['cardio']).values)

                # Train locally
                state_dict = self.train_epoch(self.load_model(new_weights), X, y)

                if not state_dict:
                    raise ValueError("Training failed or returned empty state dict.")

                # "Send" the update back to the Hub via our bus
                print(f"[CLIENT] Uploading local weights for epoch {expected_epoch + 1}...")
                self.bus.publish_model(
                    name=f'weights_epoch_{expected_epoch + 1}',
                    state_dict=state_dict,
                    client_id=self.client_id
                )

                # Prepare for the next round
                self.epoch += 1

            except TimeoutError:
                print("[CLIENT] Waiting Hub Trigger...")
                time.sleep(5.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NVFlare-enabled Federated Client Worker")
    parser.add_argument('--client-id', required=True, help="Worker ID string")
    parser.add_argument('--hub-url', default='http://hub.girder.com/api/v1')
    parser.add_argument('--hub-token', required=True)
    parser.add_argument('--workspace', required=True, help="Shared Hub Girder Folder path")
    # Added back the local Girder options for fetching private data on isolated spokes
    parser.add_argument('--local-girder-url', default=None, help="Local Girder URL on the spoke node for data access")
    parser.add_argument('--local-girder-token', default=None, help="Auth token for the local Girder instance")
    parser.add_argument('--data-path', required=True, help="Local CSV or local Girder Item path within container")

    args = parser.parse_args()

    FedClientWorker(
        client_id=args.client_id,
        hub_url=args.hub_url,
        hub_token=args.hub_token,
        workspace_path=args.workspace,
        data_path=args.data_path,
        local_girder_url=args.local_girder_url,
        local_girder_token=args.local_girder_token
    ).run()
