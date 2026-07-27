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

from nvflare_bus import GirderBridge


class HubCoordinator:
    def __init__(self, girder_url: str, work_path: str, epochs: int, num_clients: int):
        self.work_path = work_path
        self.epochs = int(epochs)
        self.num_clients = int(num_clients)
        self.girder_url = None  # Initialized in _init_components
        self.gc = None  
        self.workspace = None
        self.folder_id = None
        self.girder_bridge = None

    def _init_components(self, girder_token: str):
        import girder_client
        
        # Initialize Girder client
        self.gc = girder_client.GirderClient(apiUrl=self.girder_url)
        self.gc.token = girder_token

        # Verify Hub Workspace
        self.workspace = self.gc.get('resource/lookup', parameters={'path': self.work_path})
        if not self.workspace:
            raise FileNotFoundError(f"Hub workspace '{self.work_path}' not found in Girder.")
            
        self.folder_id = self.workspace['_id']

        # Initialize the NVFlare Girder Bridge using explicit authentication (no env fallbacks)
        self.girder_bridge = GirderBridge(
            girder_url=self.girder_url,
            girder_token=girder_token,
            work_path=self.work_path
        )

    def run(self, girder_token):
        import torch
        
        # Ensure components are initialized with the provided hub token
        if not self.girder_bridge:
            self._init_components(girder_token)
            
        print(f'[HUB] Starting federated training with {self.epochs} epochs and {self.num_clients} clients.')

        default_feat_size = 21 
        initial_weights = {
            'fc1.weight': torch.zeros(64, default_feat_size),   
            'fc1.bias': torch.zeros(64),
            'fc2.weight': torch.zeros(32, 64),
            'fc2.bias': torch.zeros(32),
            'fc3.weight': torch.zeros(2, 32),
            'fc3.bias': torch.zeros(2)
        }

        self.girder_bridge.write_task(round_num=0, payload=initial_weights)

        for epoch in range(self.epochs):
            print(f'\n--- Coordinating Epoch {epoch + 1}/{self.epochs} ---')
            
            # Create trigger marker item via the bridge's synchronous poll mechanism
            self.girder_bridge._create_marker_item(f'trigger_{int(epoch)}')

            print(f'[HUB] Waiting for {self.num_clients} workers to complete round {epoch + 1}...')
            
            # Wait for clients via Girder Bridge protocol with explicit HTTP polling
            completed = self.girder_bridge.wait_for_clients_complete(
                round_num=int(epoch),
                total_clients=self.num_clients,
                timeout=600.0,
                poll_interval=2.0
            )

            if not completed:
                print(f'[HUB] Warning: Not all clients responded for epoch {epoch + 1}')

            # Aggregate weights via Girder Bridge protocol
            new_global_state = self._load_client_weights(epoch)
            
            try:
                self.girder_bridge.write_task(round_num=int(epoch) + 1, payload=new_global_state)
            except Exception as e:
                print(f"[HUB] Error writing aggregated weights: {e}")
            
        print('[HUB] Federated learning completed successfully.')

    def _load_client_weights(self, epoch):
        client_results = self.girder_bridge.read_all_results(epoch)
        
        if not client_results:
             raise RuntimeError(f"No client weights found in folder for epoch {epoch}")
             
        import torch
        
        aggregated_weights = None
        
        for result in client_results:
            if aggregated_weights is None:
                aggregated_weights = {k: v.clone() * 0.0 for k, v in result.items()}
                
            for key in aggregated_weights.keys():
                if key in result:
                    aggregated_weights[key].add_(result[key])
        
        for key in aggregated_weights.keys():
            aggregated_weights[key] /= len(client_results)
            
        return aggregated_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Exactly matching the original scliw_federated arguments
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1',
                        help='Hub Girder URL')
    parser.add_argument('--girder-token', required=True,
                        help='Hub Girder authentication token (B64-encoded JWT or API key)')
    parser.add_argument('--work-path', required=True,
                        help='Hub Girder path to work folder')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to run')
    parser.add_argument('--clients', type=int, default=4,
                        help="Number of clients to expect for aggregation")
            
    args = parser.parse_args()
    
    # Initialize the Hub Coordinator with workspace resolution immediately to ensure path exists
    hub = HubCoordinator(
        girder_url=args.girder_url,
        work_path=args.work_path, 
        epochs=args.epochs,
        num_clients=args.clients
    )
    hub.girder_url = args.girder_url

    hub.run(args.girder_token)