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


class HubCoordinator:
    def __init__(self, girder_url: str, work_path: str, epochs: int, num_clients: int, girder_token: str):
        self.girder_url = girder_url
        self.work_path = work_path
        self.epochs = int(epochs)
        self.num_clients = int(num_clients)
        self.token = girder_token

        import girder_client
        self.gc = girder_client.GirderClient(apiUrl=self.girder_url)
        self.gc.token = self.token

        self.workspace = self.gc.get('resource/lookup', parameters={'path': self.work_path})
        if not self.workspace:
            raise FileNotFoundError(f"Hub workspace '{self.work_path}' not found in Girder.")

        # Initialize the NVFlare DropBox bridge
        self.dropbox = GirderDropbox(
            girder_url=girder_url,
            girder_token=girder_token,
            work_path=work_path
        )

    def run(self):
        print(f'[HUB] Starting federated training with {self.epochs} epochs and {self.num_clients} clients.')
        import torch

        default_feat_size = 21 
        initial_weights = {
            'fc1.weight': torch.zeros(64, default_feat_size),   
            'fc1.bias': torch.zeros(64),
            'fc2.weight': torch.zeros(32, 64),
            'fc2.bias': torch.zeros(32),
            'fc3.weight': torch.zeros(2, 32),
            'fc3.bias': torch.zeros(2)
        }

        # Seed the first global model using DropBox
        self.dropbox.write_task(round_num=0, payload=initial_weights)

        for epoch in range(self.epochs):
            print(f'\n--- Coordinating Epoch {epoch + 1}/{self.epochs} ---')
            
            # Create trigger marker item
            self.gc.createItem(
                parentFolderId=self.workspace['_id'], 
                name=f'trigger_{int(epoch)}', 
                metadata={'type': 'train'}
            )

            print(f'[HUB] Waiting for {self.num_clients} workers to complete round {epoch + 1}...')
            
            # Wait for clients via DropBox protocol
            completed = self.dropbox.wait_for_clients_complete(
                round_num=int(epoch),
                total_clients=self.num_clients,
                timeout=600.0
            )

            if not completed:
                print(f'[HUB] Warning: Not all clients responded for epoch {epoch + 1}')

            # Aggregate weights via DropBox protocol
            new_global_state = self._load_client_weights(epoch)
            
            try:
                self.dropbox.write_task(round_num=int(epoch) + 1, payload=new_global_state)
            except Exception as e:
                print(f"[HUB] Error writing aggregated weights: {e}")
            
        print('[HUB] Federated learning completed successfully.')

    def _load_client_weights(self, epoch):
        """Downloads weight updates from all clients for a given epoch and averages them."""
        client_results = self.dropbox.read_all_results(epoch)
        
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
    parser = argparse.ArgumentParser(description="Hub Coordinator for Cardio NVFlare")
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1', help='Hub Girder URL')
    parser.add_argument('--work-path', required=True, help='Hub Girder path to work folder')
    parser.add_argument('--epochs', type=int, default=5, help='Number of federated training rounds')
    parser.add_argument('--clients', type=int, default=4, help='Number of distributed clients expected')
    parser.add_argument('--girder-token', required=True, help='Hub Girder authentication token')
    
    args = parser.parse_args()

    hub = HubCoordinator(
        girder_url=args.girder_url,
        work_path=args.work_path,
        epochs=args.epochs,
        num_clients=args.clients,
        girder_token=args.girder_token
    )
    
    hub.run()