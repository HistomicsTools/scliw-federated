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
import tempfile
import time

import girder_client
import torch


class HubCoordinator:
    def __init__(self, work_path, epochs, num_clients):
        """
        Central Coordinator for Cardio NVFlare-style Federated Learning.
        Communicates with distributed clients exclusively via the Girder workspace
        folder as a dropbox for model weights and task queueing.
        Mirrors the existing scliw_federated hub.py interface exactly.
        """
        self.work_path = work_path
        self.epochs = int(epochs)
        self.num_clients = int(num_clients)
        self.gc = None
        self.workspace = None
        self.folder_id = None
        self.tmpdir = tempfile.mkdtemp()

    def run(self):
        print(f'Starting federated learning for {self.epochs} epochs')
        # Initialize initial dummy weights (zeros) to seed the network.
        default_feat_size = 11
        initial_weights = {
            'fc1.weight': torch.zeros(64, default_feat_size),
            'fc1.bias': torch.zeros(64),
            'fc2.weight': torch.zeros(32, 64),
            'fc2.bias': torch.zeros(32),
            'fc3.weight': torch.zeros(2, 32),
            'fc3.bias': torch.zeros(2)
        }

        self.workspace = self.gc.get('resource/lookup', parameters={'path': self.work_path})
        if not self.workspace:
            raise FileNotFoundError(f"Hub workspace '{self.work_path}' not found in Girder.")

        self.folder_id = self.workspace['_id']
        save_path = os.path.join(self.tmpdir, 'global_model_epoch_0.pt')
        torch.save(initial_weights, save_path)
        self.gc.uploadFileToFolder(
            self.folder_id,
            save_path,
            'global_model_epoch_0.pt',
        )

        # Main Federated Learning Loop
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            # Create trigger item so workers know a new round is ready
            self.gc.createItem(
                parentFolderId=self.folder_id,
                name=f'trigger_{int(epoch)}',
                metadata={'epoch': epoch}
            )
            print(f'[HUB] Waiting for {self.num_clients} workers to complete round {epoch + 1}')
            max_wait = time.time() + 3600
            completed = 0

            while completed < self.num_clients and time.time() < max_wait:
                items = list(self.gc.listItem(self.folder_id))
                completed = sum(1 for item in items if f'completed_{int(epoch)}_' in item['name'])
                if completed < self.num_clients:
                    time.sleep(5)
            if completed < self.num_clients:
                print(f'[HUB] Warning: Only {completed} of {self.num_clients} clients responded.')

            # Aggregate models via Girder download/average across the arbitrary number of clients
            new_global_state = self.load_client_weights(epoch)

            save_path = os.path.join(self.tmpdir, f'global_model_epoch_{epoch + 1}.pt')
            torch.save(new_global_state, save_path)
            self.gc.uploadFileToFolder(
                self.folder_id,
                save_path,
                f'global_model_epoch_{epoch + 1}.pt',
            )

        print('[HUB] Federated learning completed successfully.')

    def load_client_weights(self, epoch):
        """Downloads weight updates from all clients for a given epoch and averages them."""
        items = list(self.gc.listItem(self.folder_id))
        aggregated_weights = None
        count = 0

        pattern = f'weights_epoch_{int(epoch)}_client'

        for item in items:
            if not item['name'].startswith(pattern):
                continue
            if not item['name'].endswith('.pt'):
                continue

            files = list(self.gc.listFile(item['_id'], limit=1))
            if not files:
                continue

            local_path = os.path.join(self.tmpdir, item['name'])
            self.gc.downloadFile(files[0]['_id'], local_path)

            client_state_dict = torch.load(local_path, map_location='cpu')

            # Initialize accumulator on first hit with zeroed tensors matching keys
            if aggregated_weights is None:
                aggregated_weights = {k: v.clone() * 0.0 for k, v in client_state_dict.items()}

            count += 1

            # Add tensors element-wise across all workers (FedAvg)
            for key in aggregated_weights.keys():
                if key in client_state_dict:
                    aggregated_weights[key].add_(client_state_dict[key])

        if count == 0 or aggregated_weights is None:
            raise RuntimeError(f'No client weights found in folder for epoch {epoch}')

        # Normalize by number of clients that contributed to the global average
        for key in aggregated_weights.keys():
            aggregated_weights[key] /= count

        return aggregated_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1',
                        help='Local Girder URL')
    parser.add_argument('--girder-token', required=True,
                        help='Local Girder authentication token')
    parser.add_argument('--work-path', required=True,
                        help='Hub Girder path to work folder')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to run')
    parser.add_argument('--clients', type=int, default=4,
                        help='Number of clients to expect')
    args = parser.parse_args()

    # Initialize Girder client and pass it to HubCoordinator
    api_client = girder_client.GirderClient(apiUrl=args.girder_url)
    api_client.token = args.girder_token

    hub = HubCoordinator(work_path=args.work_path, epochs=args.epochs, num_clients=args.clients)
    hub.gc = api_client
    hub.run()
