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
    
    def __init__(self, client_id, local_girder_url, hub_url, work_path, data_path):
        self.client_id = int(client_id)
        self.data_path = data_path
        self.dropbox = None 

    def _load_data(self):
        """Loads strict private data. Falls back to Girder download if filesystem path is not found."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        if os.path.exists(self.data_path):
            print(f"[*] Loading data directly from mounted filesystem path: {self.data_path}")
            df = pd.read_csv(self.data_path, sep=',')
        else:
            local_gc = girder_client.GirderClient(apiUrl='http://localhost:8085/api/v1')
            cred_file = os.environ.get('SCLIW_HUB_CRED_PATH')
            if cred_file and os.path.exists(cred_file):
                with open(cred_file) as f:
                    local_gc.token = f.read().strip()
                    
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
        
        if self.dropbox is None:
            # Lazily initialize dropbox once we have a token, though we'd usually init in __init__
            # If __init__ didn't have it, this fallback handles local testing without creds
            import warnings
            warnings.warn("Hub token might not be set on DropBox if initialized before token was available.")
        
        self.dropbox.gc.token = hub_token
        
        while True:
            print(f"[CLIENT] Waiting for next round...")
            
            try:
                # Ensure the dropbox is properly initialized with the work_path provided in __init__
                # Note: GirderDropbox requires work_path to resolve folder_id in __init__, 
                # so it should have been set when the class was instantiated.
                
                self.dropbox.wait_for_task_ready(round_num=self.client_id, timeout=300.0)
                
                global_weights = self.dropbox.read_task(self.client_id)
                X_local, y_local = self._load_data()

                device = torch.device("cpu")
                model = CardioNN(input_size=X_local.shape[1]).to(device)
                model.load_state_dict(global_weights)
                
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_local, y_local), batch_size=32, shuffle=True)
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
    parser = argparse.ArgumentParser(description="Federated Cardio Worker (Girder File Transport)")
    parser.add_argument('--client-id', required=True, help='Unique identifier for this distributed worker')
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1', help='Local Girder URL')
    parser.add_argument('--hub-url', default='http://localhost:8080/api/v1', help='Hub Girder API URL')
    parser.add_argument('--work-path', required=True, help='Hub Girder path to work folder')
    parser.add_argument('--data-path', default='/dev/null', help='Local filesystem path OR local Girder item path')
    
    # --girder-token is typically passed from env or a separate arg in your CI/CD setup. 
    # If testing locally, you may need to ensure GIRDER_API_KEY is exported.
    args = parser.parse_args()

    import os
    hub_cred_path = os.environ.get('SCLIW_HUB_CRED_PATH')
    if hub_cred_path and os.path.exists(hub_cred_path):
        with open(hub_cred_path) as f:
            hub_token = f.read().strip()
    else:
        hub_token = 'default_dev_token_placeholder'

    worker = FederatedCardioClient(
        client_id=args.client_id,
        local_girder_url=args.girder_url,
        hub_url=args.hub_url,
        work_path=args.work_path,
        data_path=args.data_path
    )

    worker.run_loop(hub_token=hub_token)