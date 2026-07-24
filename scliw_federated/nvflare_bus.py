#!/usr/bin/env python3
"""
Girder-based Event Bus for NVFlare Federated Learning.
Implements the synchronization logic for a Hub and Client architecture where 
only file attachments (Girder Items) are exchanged between nodes.
"""
import os
import time
import json
import tempfile
from typing import Optional, Dict

import girder_client
import torch


class GirderEventBus:
    """
    A message bus that relies entirely on a central Hub Girder workspace folder 
    for synchronization across isolated environments.
   """
    def __init__(self, hub_url: str, hub_token: str, work_path: str):
        self.gc = girder_client.GirderClient(apiUrl=hub_url)
        self.gc.token = hub_token
        
        workspace_info = self.gc.get('resource/lookup', parameters={'path': work_path})
        if not workspace_info:
            raise FileNotFoundError(f"Girder Hub workspace '{work_path}' not found.")
            
        self.folder_id = workspace_info['_id']
        self.work_dir = tempfile.mkdtemp()

    def publish_item(self, name: str, data: Dict, metadata: Optional[Dict] = None):
        """Sends a command/trigger to the bus by creating a Girder item."""
        json_path = os.path.join(self.work_dir, f'{name}.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
            
        self.gc.uploadFileToFolder(
            self.folder_id, 
            json_path, 
            f'task_{os.path.basename(name)}_trigger.json', 
            metadata=metadata or {}
        )

    def publish_model(self, name: str, state_dict: Dict[str, torch.Tensor], client_id: str):
        """Sends updated model weights as a binary Girder item."""
        path = os.path.join(self.work_dir, f'{name}_{client_id}.pt')
        torch.save(state_dict, path)
        
        self.gc.uploadFileToFolder(
            self.folder_id, 
            path, 
            f'task_{os.path.basename(name)}_weights_{client_id}.pt',
            metadata={'status': 'ready', 'client_id': client_id}
        )

    def subscribe_item(self, expected_name: str, timeout: int = 3600) -> Dict:
        """Polls the workspace for an incoming task/trigger file."""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            items = list(self.gc.listItem(self.folder_id))
            
            matching_items = [
                i for i in items 
                if expected_name in i['name'] and 'trigger' in i['name']
            ]
            
            if matching_items:
                item = matching_items[0]
                
                files = list(self.gc.listFile(item['_id'], limit=1))
                if not files: 
                    time.sleep(2.0)
                    continue
                    
                target_path = os.path.join(self.work_dir, f'recv_{item["_id"]}.json')
                self.gc.downloadFile(files[0]['_id'], target_path)
                
                with open(target_path, 'r') as f:
                    data = json.load(f)
                    
                # Clean up the trigger item so it doesn't cause loops later
                self.gc.removeResource(item['_id'])

                return data
            
            time.sleep(2.0)
            
        raise TimeoutError(f"Timeout waiting for event '{expected_name}'.")

    def fetch_model(self, client_id: str, expected_epoch: int) -> Optional[Dict[str, torch.Tensor]]:
        """Hub side: Polls workspace to collect all weights from the latest epoch, then aggregates."""
        items = list(self.gc.listItem(self.folder_id))
        
        aggregated_weights = None
        received_clients = set()
        target_pattern = f'task_epoch_{expected_epoch}_weights_'

        for item in items:
            if not item['name'].startswith(target_pattern): 
                continue
                
            name_parts = item['name'].split('_')
            cid_part = name_parts[-1].replace('.pt', '')
            
            if cid_part != 'hub' and cid_part not in received_clients:
                received_clients.add(cid_part)
                
                files = list(self.gc.listFile(item['_id'], limit=1))
                path = os.path.join(self.work_dir, f"{item['name']}")
                self.gc.downloadFile(files[0]['_id'], path)
                
                state_dict = torch.load(path, map_location='cpu')
                
                # NVFlare-style aggregation accumulation
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * 0.0 for k, v in state_dict.items()}
                
                for key in aggregated_weights.keys():
                    if key in state_dict and aggregated_weights[key].shape == state_dict[key].shape:
                        aggregated_weights[key].add_(state_dict[key])

        # Normalize by the number of contributing clients
        if aggregated_weights and len(received_clients) > 0:
            divisor = float(len(received_clients))
            for key, tensor in aggregated_weights.items():
                tensor.div_(divisor)
                
        return aggregated_weights if aggregated_weights else {}
