#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "girder-client==2.4.0",
#     "torch>=2.0.0",
#     "nvflare>=2.6.0",
# ]
# ///

import argparse
import os
import tempfile
import sys
from typing import Dict, Optional, Any

import torch
import girder_client

try:
    from .nvflare_bus import GirderEventBus
except Exception:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from nvflare_bus import GirderEventBus

from nvflare.app_common.aggregators.in_time_accumulate_weighted_aggregator import InTimeAccumulateWeightedAggregator


class FedHubCoordinator:
    """
    Central Coordinator for Cardio NVFlare-style Federated Learning.
    Communicates with distributed clients exclusively via the Girder workspace
    folder as a dropbox for model weights and task queueing.
    """
    def __init__(self, girder_url: str, girder_token: str, work_path: str, epochs: int, num_clients: int):
        self.epochs = int(epochs)
        self.num_clients = int(num_clients)

        # Initialize the bridge between NVFlare logic (averaging) and Girder Transport
        self.bus = GirderEventBus(girder_url, girder_token, work_path)

        # Set up NVFlare's standard aggregation engine for federated learning
        self.aggregator = InTimeAccumulateWeightedAggregator(
            expected_data_kind='WEIGHTS'
        )

    def run(self):
        print(f'[HUB] Starting Federated Learning across {self.num_clients} spokes for {self.epochs} epochs.')

        # Initial seed weights to distribute initially (NVFlare expects a starting point)
        initial_weights = torch.nn.Linear(10, 2).state_dict()
        self._seed_weights(initial_weights)

        for epoch in range(self.epochs):
            print(f'--- Epoch {epoch + 1}/{self.epochs} ---')

            # 1. Broadcast to all clients (via our event bus) instructing them to train this round
            self.bus.publish_item(
                name=f'global_model_epoch_{epoch}',
                data={'status': 'train_ready', 'epoch': epoch}
            )

            # 2. Wait for the Hub's aggregation engine and our Bus to collect from spokes
            gathered_weights = self.bus.fetch_model(
                client_id='hub', expected_epoch=epoch + 1
            )

            if not gathered_weights:
                print("[HUB] No clients responded this round. Proceeding with previous global state.")
                continue

            # 3. Use NVFlare's standard aggregator logic to combine the distributed weights
            aggregated_state = self.aggregator.aggregate(gathered_weights)

            # Persist the new global model by publishing it to the bus for the NEXT epoch download
            if aggregated_state:
                 print(f"[HUB] Successfully aggregated weights from {len(gathered_weights)} clients.")
                 self._seed_weights(aggregated_state)

        # Signal shutdown after final epochs
        self.bus.publish_item(name='global_model_shutdown', data={'status': 'shutdown'})
        print("[HUB] Federated Learning cycle completed successfully.")

    def _seed_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Uploads weights to the Hub's workspace for clients to poll via the Event Bus."""

        # Create a temporary object for NVFlare-like passing through PyTorch persistence
        path = os.path.join(tempfile.mkdtemp(), 'global_weights.pt')
        torch.save(state_dict, path)

        self.bus.gc.uploadFileToFolder(
            self.bus.folder_id,
            path,
            f'task_epoch_{self.epochs}_weights_hub_final.pt', # Final epoch marker
            metadata = {'type': 'seed'}
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--girder-url', default='http://localhost:8080/api/v1',
                        help='Local Girder URL')
    parser.add_argument('--girder-token', required=True,
                        help='Local Girder authentication token')
    parser.add_argument('--work-path', required=True,
                        help='Hub Girder path to work folder')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to run')
    parser.add_argument('--clients', type=int, default=4,
                        help='Number of clients to expect')

    args = parser.parse_args()

    # Initialize Hub Coordinator with Girder connection details
    hub = FedHubCoordinator(
        girder_url=args.girder_url,
        girder_token=args.girder_token,
        work_path=args.work_path,
        epochs=args.epochs,
        num_clients=args.clients
    )

    # Start the Federated Learning coordinator loop
    hub.run()
