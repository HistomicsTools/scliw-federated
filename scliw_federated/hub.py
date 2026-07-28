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
import os
import sys


class HubCoordinator:
    def __init__(self, girder_url: str, work_path: str, epochs: int, num_clients: int):
        self.work_path = work_path
        self.epochs = int(epochs)
        self.num_clients = int(num_clients)
        self.girder_url = None

        # Initialize NVFlare's federated aggregator component for model aggregation
        from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
        self.nvflare_aggregator = InTimeAccumulateWeightedAggregator()

    def _init_components(self, girder_token: str):
        import girder_client
        from scliw_federated.nvflare_bus import GirderBridge

        self.girder_url = "http://localhost:8080/api/v1" # Default fallback logic if needed later

        self.gc = girder_client.GirderClient(apiUrl=self.girder_url)
        self.gc.token = girder_token

        workspace = self.gc.get('resource/lookup', parameters={'path': self.work_path})
        if not workspace:
            from pathlib import Path
            # Handle local testing file path fallback if Girder lookup fails but it's a real dir
            if os.path.exists(self.work_path):
                print(f"[HUB] Warning: Workspace '{self.work_path}' not in Girder, treating as local path.")
                self.folder_id = None # Use local logic if applicable
            else:
                raise FileNotFoundError(f"Hub workspace '{self.work_path}' not found in Girder or file system.")

        self.folder_id = workspace.get('_id')

        # Initialize the NVFlare Girder Bridge using explicit authentication (transport layer)
        self.girder_bridge = GirderBridge(
            girder_url=self.girder_url,
            girder_token=girder_token,
            work_path=self.work_path
        )

    def run(self, girder_token: str):
        import torch
        from nvflare.apis.fl_context import FLContext
        from nvflare.apis.shareable import Shareable
        from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator

        # Ensure components are initialized with the provided hub token
        if not self.girder_bridge:
            self._init_components(girder_token)

        print(f'[HUB] Starting federated training with {self.epochs} epochs and {self.num_clients} clients.')

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
            print(f'--- Coordinating Epoch {epoch + 1}/{self.epochs} ---')

            # Create trigger marker item via the bridge's synchronous poll mechanism
            self.girder_bridge._create_marker_item(f'trigger_{int(epoch)}')

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
                raise RuntimeError(f"No client weights found in folder for epoch {epoch}")

            # Delegate model aggregation to the NVFlare Aggregator component
            # Wrap raw tensors into standardized FL shares
            shareable_list = []
            for idx, w_dict in enumerate(client_raw_weights):
                share = Shareable()
                share.set_shareable_key(ShareableKey.AGGREGATION_INDEX, str(idx))
                share.set_data(key="WEIGHTS", data=w_dict)
                shareable_list.append(share)

            fl_ctx = FLContext()

            # Pass shares to NVFlare's federated averaging logic explicitly
            aggregated_share = self.nvflare_aggregator.aggregate(
                num_epochs=1,
                flctx=fl_ctx,
                result_shareables=shareable_list
            )

            new_global_state = aggregated_share.get_data("WEIGHTS")

            try:
                self.girder_bridge.write_task(round_num=int(epoch) + 1, payload=new_global_state)
            except Exception as e:
                print(f"[HUB] Error writing aggregated weights: {e}")

        print('[HUB] Federated learning completed successfully.')


if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
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
    args = parser.parse_args()
    hub = HubCoordinator(
        girder_url=args.girder_url,
        work_path=args.work_path,
        epochs=args.epochs,
        num_clients=args.clients
    )
    hub.run(args.girder_token)
