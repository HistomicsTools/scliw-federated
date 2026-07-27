#!/usr/bin/env python3
# /// script
# requires-python = '>=3.10'
# dependencies = [
#     'girder-client',
#     'torch',
# ]
# ///

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from scliw_federated.nvflare_bus import GirderBridge

import girder_client
import torch


class CardioNN(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


class FederatedCardioClient:
    """
    Client-side federated learning worker using NVFlare logic + Girder Bridge.
    Maintains strict data isolation while communicating via the Girder asset system.
    """

    def __init__(self, client_id: int, local_girder_url: str, local_token, hub_url: str, hub_token,
                 work_path: str, data_path: str):
        self.client_id = client_id
        self.data_path = data_path

        # Initialize connection to Hub Girder via the bridge
        self.girder_bridge = GirderBridge(
            girder_url=hub_url,
            girder_token=hub_token,
            work_path=work_path
        )

    def _load_data(self, hub_token: str, local_girder_url: str):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # Priority 1: Direct filesystem access
        if getattr(self, 'local_data_path', None) and os.path.exists(self.local_data_path):
            print(f"[*] Loading data directly from mounted filesystem: {self.local_data_path}")
            df = pd.read_csv(self.local_data_path, sep=',')
        else:
            # Priority 2: Download from local Girder (Original pattern)
            local_gc = girder_client.GirderClient(apiUrl=local_girder_url)
            if self.local_token:
                local_gc.token = self.local_token

            local_item = local_gc.get('resource/lookup', parameters={'path': self.data_path})
            if not local_item:
                raise FileNotFoundError(f"Data not found at '{self.data_path}' in filesystem or Girder.")

            files = list(local_gc.listFile(local_item['_id'], limit=1))
            import tempfile
            tmp_data = os.path.join(tempfile.mkdtemp(), 'client_data.csv')
            local_gc.downloadFile(files[0]['_id'], tmp_data)
            df = pd.read_csv(tmp_data, sep=',')

        if 'cardio' in df.columns:
            y = df['cardio'].values.astype(int)
            X = df.drop(columns=['cardio']).values.astype(float)
        else:
            y = df.iloc[:, -1].values.astype(int)
            X = df.iloc[:, :-1].values.astype(float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def run_loop(self, hub_token: str):
        import torch.nn as nn

        current_epoch = -1  # Track which round we are currently on
        print(f"[CLIENT] Starting loop for client ID {self.client_id}.")

        while True:
            # Determine the next epoch to wait for (Hub usually broadcasts + 1 of previous)
            target_round = current_epoch + 1
            marker_name = f'task_{target_round}_ready'

            print(f"[CLIENT] Waiting for Hub broadcast '{marker_name}'...")

            # Poll using the bridge specifically for this round's trigger
            ready = self.girder_bridge.wait_for_task_ready(
                round_num=target_round,
                timeout=300.0,
                poll_interval=2.0
            )

            if not ready:
                print(f"[CLIENT] Timeout waiting for epoch {target_round}!")
                continue

            global_weights = self.girder_bridge.read_task(target_round)

            if global_weights is None:
                print(f"[CLIENT] No model weights found for epoch {target_round}.")
                continue

            X_local, y_local = self._load_data(hub_token, self.local_girder_url)

            device = torch.device("cpu")
            model = CardioNN(input_size=X_local.shape[1]).to(device)
            model.load_state_dict(global_weights)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_local, y_local),
                batch_size=32, shuffle=True
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for _epoch in range(3): # Local epoch count
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

            updated_weights = model.state_dict()

            # Write result BACK to Girder using the SAME target_round
            print(f"[CLIENT] Uploading result for epoch {target_round}...")
            self.girder_bridge.write_result(
                client_id=self.client_id,
                round_num=target_round,
                payload=updated_weights
            )

            current_epoch = target_round # Advance our internal state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Identical arguments to the original setup
    parser.add_argument('--client-id', required=True, help='Client identifier')
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1', help='Local Girder URL')
    parser.add_argument('--girder-token', required=True, help='Local Girder authentication token')
    parser.add_argument('--hub-url', default='http://hub.example.com/api/v1', help='Hub Girder URL')
    parser.add_argument('--hub-token', required=True, help='Hub authentication token')
    parser.add_argument('--data-path', required=True, help='Local Girder path to csv data item')
    parser.add_argument('--work-path', required=True, help='Hub Girder path to work folder')

    args = parser.parse_args()

    worker = FederatedCardioClient(
        client_id=args.client_id,
        local_girder_url=args.girder_url,
        local_token=args.girder_token,
        hub_url=args.hub_url,
        hub_token=args.hub_token,
        work_path=args.work_path,
        data_path=args.data_path
    )
    worker.local_token = args.girder_token
    worker.local_girder_url = args.girder_url

    worker.run_loop(hub_token=args.hub_token)
