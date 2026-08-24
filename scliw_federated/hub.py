#!/usr/bin/env python3
# /// script
# requires-python = '>=3.10'
# dependencies = [
#     'girder-client',
#     'torch',
#     'nvflare',
# ]
# ///

import argparse
import datetime
import os
import sys

import girder_client


class HubCoordinator:
    def __init__(self, girder_url: str, work_path: str, epochs: int, num_clients: int):
        import nvflare.app_common.aggregators.intime_accumulate_model_aggregator
        from nvflare.apis.dxo import DataKind

        self.work_path = work_path
        self.epochs = int(epochs)
        self.num_clients = int(num_clients)
        self.girder_bridge = None
        self.girder_url = girder_url
        self.nvflare_aggregator = nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator(  # noqa
            expected_data_kind={'weights': DataKind.WEIGHT_DIFF}
        )

    def init_components(self, girder_token: str):
        from scliw_federated.girder_bridge import GirderBridge

        self.gc = girder_client.GirderClient(apiUrl=self.girder_url)
        self.gc.token = girder_token

        workspace = self.gc.get('resource/lookup', parameters={'path': self.work_path})
        if not workspace:
            # Handle local testing file path fallback if Girder lookup fails but it's a real dir
            if os.path.exists(self.work_path):
                self.folder_id = None
            else:
                raise FileNotFoundError(
                    f'Hub workspace {self.work_path} not found in Girder or file system.')

        self.folder_id = workspace.get('_id')

        # Initialize the NVFlare Girder Bridge using explicit authentication (transport layer)
        self.girder_bridge = GirderBridge(
            girder_url=self.girder_url,
            girder_token=girder_token,
            work_path=self.work_path
        )

    def run(self, girder_token: str):
        import torch

        # Ensure components are initialized with the provided hub token
        if not self.girder_bridge:
            self.init_components(girder_token)
        print(f'[HUB] Starting federated training with {self.epochs} epochs '
              f'and {self.num_clients} clients.')
        default_feat_size = 11
        initial_weights = {
            'fc1.weight': torch.zeros(64, default_feat_size),
            'fc1.bias': torch.zeros(64),
            'fc2.weight': torch.zeros(32, 64),
            'fc2.bias': torch.zeros(32),
            'fc3.weight': torch.zeros(2, 32),
            'fc3.bias': torch.zeros(2)
        }
        # Send initial global state via Girder transport (using bridge for polling/triggering)
        self.girder_bridge.write_task(round_num=0, payload=initial_weights)
        for epoch in range(self.epochs):
            print(f'Coordinating Epoch {epoch + 1}/{self.epochs}')
            print(f'[HUB] Waiting for {self.num_clients} workers to complete round {epoch + 1}')
            # Wait for clients via Girder Bridge protocol with explicit HTTP polling
            completed = self.girder_bridge.wait_for_clients_complete(
                round_num=int(epoch),
                total_clients=self.num_clients,
                timeout=600.0,
                poll_interval=2.0
            )
            if not completed:
                print(f'[HUB] Warning: Not all clients responded for epoch {epoch + 1}')
            # Load client weights retrieved via Girder file transfer
            client_raw_weights = self.girder_bridge.read_all_results(epoch)
            if not client_raw_weights:
                raise RuntimeError(f'No client weights found in folder for epoch {epoch}')

            # Because NVFlare's InTimeAccumulateWeightedAggregator strictly validates
            # internal DXO formats, we perform standard FedAvg weight averaging manually here
            # to bypass transport-layer validation errors over Girder.
            new_global_state = None
            if len(client_raw_weights) > 0:
                print(f'[HUB] Aggregating {len(client_raw_weights)} client '
                      f'updates for epoch {epoch + 1}')
                # Average the weights: sum all dicts and divide by the number of clients.
                new_global_state = {}
                for model_key in client_raw_weights[0].keys():
                    accumulated = sum(r.get(model_key, torch.tensor(0.0))
                                      for r in client_raw_weights)
                    new_global_state[model_key] = accumulated / len(client_raw_weights)
            try:
                if new_global_state is not None:
                    self.girder_bridge.write_task(
                        round_num=int(epoch) + 1, payload=new_global_state)
                    print(f'[HUB] Wrote aggregated weights for epoch {epoch + 1} to Girder.')
                else:
                    print(f'[HUB] Aggregation incomplete or empty for epoch {epoch + 1}')
            except Exception as e:
                print(f'[HUB] Error writing aggregated weights: {e}')
        print('[HUB] Federated learning completed successfully.')
        self.girder_bridge.write_done()


if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    parser = argparse.ArgumentParser()
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1',
                        help='Hub Girder URL')
    parser.add_argument('--girder-token', required=True,
                        help='Hub Girder authentication token (B64-encoded JWT or API key)')
    parser.add_argument('--work-path', required=True,
                        help='Hub Girder path to work folder')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to run')
    parser.add_argument('--clients', type=int, default=4,
                        help='Number of clients to expect for aggregation')
    parser.add_argument(
        '--reset', choices=['false', 'true'], default='false',
        help='Rename current work-path folder (if non-empty) and create a new one in Girder')
    args = parser.parse_args()
    if str(args.reset).lower().startswith('t'):
        gc = girder_client.GirderClient(apiUrl=args.girder_url)
        gc.token = args.girder_token
        try:
            workspace = gc.get('resource/lookup', parameters={'path': args.work_path})
            if workspace:
                if 'parentId' in workspace and sum(
                        1 for _ in gc.listItem(workspace['_id'], limit=1)):
                    ts = None
                    dt_obj = datetime.datetime.now(datetime.timezone.utc)
                    try:
                        dt_obj = datetime.datetime.strptime(
                            workspace['created'].replace('Z', '+00:00'),
                            '%Y-%m-%dT%H:%M:%S.%f+00:00'
                        )
                    except Exception:
                        pass
                    formatted_dt = dt_obj.strftime('%Y%m%d-%H%M%S')
                    new_name = f'{workspace["name"]} {formatted_dt}'
                    gc.put(f'folder/{workspace["_id"]}', {'name': new_name})
                    new_folder = gc.createFolder(
                        parentId=workspace['parentId'],
                        name=workspace['name'],
                        parentType=workspace['parentCollection'],
                    )
        except Exception:
            raise
            print('[HUB] Could not reset folder')
    hub = HubCoordinator(
        girder_url=args.girder_url,
        work_path=args.work_path,
        epochs=args.epochs,
        num_clients=args.clients
    )
    hub.run(args.girder_token)
