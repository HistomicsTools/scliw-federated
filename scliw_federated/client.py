#!/usr/bin/env python3
# /// script
# requires-python = '>=3.10'
# dependencies = [
#     'girder-client',
#     'torch',
#     'pandas',
#     'scikit-learn',
#     'nvflare',
# ]
# ///

import argparse
import os
import sys
import torch


class FederatedCardioClient:
    """
    Client-side federated learning worker using NVFlare logic + Girder Bridge.
    Maintains strict data isolation while communicating via the Girder asset system.
    Handles dimension mismatches (e.g., local CSV having 13 features vs Hub expecting 21) via zero-padding.
    """

    def __init__(self, client_id: int, local_girder_url: str, hub_url: str, hub_token,
                 work_path: str, data_path: str):
        self.client_id = client_id
        self.data_path = data_path

        # Initialize connection to Hub Girder via the bridge
        from scliw_federated.nvflare_bus import GirderBridge

        self.girder_bridge = GirderBridge(
            girder_url=hub_url,
            girder_token=hub_token,
            work_path=work_path
        )

    def _load_data(self, hub_token: str, local_girder_url: str):
        """Loads CSV data. Uses Hub's global_weights dimensions if available to pad missing features."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        # Try direct filesystem access first (UCI Cardio typically at /dataset.csv or similar)
        if os.path.exists(self.data_path):
            print(f"[*] Loading data directly from mounted filesystem: {self.data_path}")
            # UCI datasets often use semicolon delimiters, try comma then semi
            for sep in [';', ',']:
                try:
                    df = pd.read_csv(self.data_path, sep=sep)
                    break
                except Exception: continue
        else:
            import girder_client

            # Priority 2: Download from local Girder (Original pattern)
            local_gc = girder_client.GirderClient(apiUrl=local_girder_url)
            if hasattr(self, 'local_token') and self.local_token:
                local_gc.token = self.local_token

            local_item = local_gc.get('resource/lookup', parameters={'path': self.data_path})
            if not local_item:
                import tempfile
                # Attempt to treat it as a file path if Girder returns nothing (local test fallback)
                if os.path.exists(self.data_path):
                     print(f"[*] Fallback to direct file access for '{self.data_path}'");
                     raise FileNotFoundError(...) # Handled by the block below anyway

            local_item = local_gc.get('resource/lookup', parameters={'path': self.data_path})
            if not local_item:
                raise FileNotFoundError(f"Data not found at '{self.data_path}' in filesystem or Girder.")

            files = list(local_gc.listFile(local_item['_id'], limit=1))
            tmp_data = os.path.join(tempfile.mkdtemp(), 'client_data.csv')
            local_gc.downloadFile(files[0]['_id'], tmp_data)

            for sep in [';', ',']:
                try:
                    df = pd.read_csv(tmp_data, sep=sep)
                    break
                except Exception: continue

        if df is None or df.empty:
            raise RuntimeError("Failed to load data from path or Girder.")

        # Extract features (X) and labels (y)
        if 'cardio' in df.columns:
            y = df['cardio'].values.astype(int)
            X = df.drop(columns=['cardio']).values.astype(float)
        else:
            y = df.iloc[:, -1].values.astype(int)
            X = df.iloc[:, :-1].values.astype(float)

        # --- Dimension Padding for Mismatched Schema ---
        import numpy as np

        target_dim = len(X[0])
        if hasattr(self, 'global_weights') and self.global_weights is not None and 'fc1.weight' in self.global_weights:
             target_dim = self.global_weights['fc1.weight'].shape[1]

        # Pad X with zeros to the expected target dimension
        if len(X[0]) < target_dim:
            print(f"[*] Padding X features from {len(X[0])} to {target_dim} to match Hub weights.")
            pad_width = target_dim - len(X[0])
            zeros_pad = torch.zeros(len(X), pad_width)
            x_tensor = torch.tensor(X).to(zeros_pad.device)
            X = torch.cat([x_tensor, zeros_pad], dim=1)

        scaler = StandardScaler()
        X_floats = np.array(X).astype(np.float32)
        X_scaled = scaler.fit_transform(X_floats)

        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def run_loop(self, hub_token: str):
        import torch.nn as nn
        from nvflare.apis.shareable import Shareable
        from nvflare.apis.fl_context import FLContext
        from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator

        current_round = 0
        total_epochs = 100
        print(f"[CLIENT] Starting loop for client ID {self.client_id}.")

        while current_round < total_epochs:
            marker_name = f'task_{current_round}_ready'

            print(f"[CLIENT] Waiting for Hub broadcast '{marker_name}'...")

            # Poll using the bridge specifically for this round's trigger (Girder-based transport)
            ready = self.girder_bridge.wait_for_task_ready(
                round_num=current_round,
                timeout=300.0,
                poll_interval=2.0
            )

            if not ready:
                print(f"[CLIENT] Timeout waiting for epoch {current_round}!")
                continue

            global_weights = self.girder_bridge.read_task(current_round)

            if global_weights is None:
                print(f"[CLIENT] No model weights found for epoch {current_round}.")
                continue

            self.global_weights = global_weights

            X_local, y_local = self._load_data(hub_token, self.local_girder_url)

            device = torch.device("cpu")
            model = CardioNN(input_size=X_local.shape[1]).to(device) # Will automatically match padded features
            model.load_state_dict(global_weights)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_local, y_local),
                batch_size=32, shuffle=True
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

            model.train()
            for _epoch in range(3): # Simulate 3 local epochs
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    loss = criterion(model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

            updated_weights = model.state_dict()

            print(f"[CLIENT] Uploading result for round {current_round}...")
            update_share = Shareable()
            # Use a standard key; NVFlare's aggregator will handle it via AGG_INDEX or list order
            update_share.set_data(key="WEIGHTS", data=updated_weights)

            self.girder_bridge.write_result(
                client_id=self.client_id,
                round_num=current_round,
                payload=updated_weights # GirderBridge handles torch.save internally
            )

            current_round += 1


class CardioNN(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2) # Assuming binary/multi-class output

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
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
        local_girder_url=args.girder_url,
        hub_url=args.hub_url,
        hub_token=args.hub_token,
        work_path=args.work_path,
        data_path=args.data_path
    )
    worker.local_token = args.girder_token
    worker.run_loop(hub_token=args.hub_token)
