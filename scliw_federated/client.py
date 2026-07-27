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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nvflare_bus import GirderDropbox

import girder_client
import torch


class CardioNN(torch.nn.Module):
    """3-layer Feed-forward Neural Network tailored for cardiovascular disease prediction."""

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
    Client-side federated learning worker using NVFlare logic + Girder Bus.
    Maintains strict data isolation while communicating via the DropBox mechanism.
    """

    def __init__(self, client_id: int, local_girder_url: str, local_token, hub_url: str, hub_token,
                 work_path: str, data_path: str):
        self.client_id = client_id
        self.data_path = data_path

        # Initialize connection to Hub Girder via the DropBox bridge
        self.dropbox = GirderDropbox(
            girder_url=hub_url,
            girder_token=hub_token,
            work_path=work_path
        )

    def _load_data(self, hub_token: str, local_girder_url: str):
        """Loads strict private data. Falls back to Girder download if filesystem path is not found."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # Priority 1: Direct filesystem access (Docker mount pattern)
        if self.local_data_path and os.path.exists(self.local_data_path):
            print(f"[*] Loading data directly from mounted filesystem: {self.local_data_path}")
            df = pd.read_csv(self.local_data_path, sep=',')
        else:
            # Priority 2: Download from local Girder (Original scliw_federated pattern)
            local_gc = girder_client.GirderClient(apiUrl=local_girder_url)
            if self.local_token:
                local_gc.token = self.local_token

            local_item = local_gc.get('resource/lookup', parameters={'path': self.data_path})
            if not local_item:
                raise FileNotFoundError(f"Data not found at '{self.data_path}' in filesystem or Girder.")

            files = list(local_gc.listFile(local_item['id'], limit=1))
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
        """Main execution loop handling Round-Robin with the Hub."""
        import torch.nn as nn

        # Ensure the DropBox has the correct token for authentication
        self.dropbox.gc.token = hub_token

        while True:
            print(f"[CLIENT] Waiting for next round...")

            try:
                # Wait for the Hub to upload the new model weights
                self.dropbox.wait_for_task_ready(round_num=self.client_id, timeout=300.0)

                global_weights = self.dropbox.read_task(self.client_id)

                X_local, y_local = self._load_data(hub_token, self.local_girder_url)

                device = torch.device("cpu")
                model = CardioNN(input_size=X_local.shape[1]).to(device)
                model.load_state_dict(global_weights)

                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_local, y_local),
                    batch_size=32,
                    shuffle=True
                )
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

                updated_weights = model.state_dict()
                self.dropbox.write_result(round_num=self.client_id, payload=updated_weights)
                print(f"[CLIENT] Round {self.client_id} complete.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[CLIENT] Error: {e}")
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Exactly matching the original scliw_federated arguments
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
