#!/usr/bin/env python3

import os
import tempfile
import time
import json

import girder_client


class GirderBridge:
    """
    A bridge implementation that routes artifact transfer via a Girder workspace folder.
    Replaces the standard FLARE FileTransfer mechanism with a polling-based Girder interface.
    """

    def __init__(self, girder_url: str, girder_token: str, work_path: str):
        self.base_dir = work_path
        self.gc = girder_client.GirderClient(apiUrl=girder_url)
        self.gc.token = girder_token

        workspace = self.gc.get('resource/lookup', parameters={'path': work_path})
        if not workspace:
            raise FileNotFoundError(f"Girder Workspace '{work_path}' not found.")

        self.folder_id = workspace['_id']

    def _get_current_items(self) -> list:
        """Helper to get a fresh list of items from Girder."""
        return list(self.gc.listItem(self.folder_id))

    def _wait_for_marker(self, marker_name: str, timeout: float = 300.0, poll_interval: float = 2.0) -> bool:
        """Poll until a specific Marker Item exists in the workspace."""
        deadline = time.time() + timeout
        
        # Ensure we don't hammer the HTTP(S) endpoint
        last_poll_time = 0 
        while time.time() < deadline:
            if time.time() - last_poll_time >= poll_interval:
                items = self._get_current_items()
                for item in items:
                    if item['name'] == marker_name:
                        return True
                
                # If not found, enforce a pause before next HTTP request
                time.sleep(poll_interval) 
                last_poll_time = time.time()
            else:
                time.sleep(0.1)

        return False

    def _create_marker_item(self, marker_name: str) -> None:
        """Create a trigger/completed marker Item in Girder."""
        self.gc.createItem(parentFolderId=self.folder_id, name=marker_name, metadata={"type": "marker"})

    def write_task(self, round_num: int, payload):
        """Hub writes the global model (or trigger data) to Girder."""
        task_file_name = f'model_epoch_{int(round_num)}.pt'
        marker_name = f'task_{round_num}_ready'

        tmp_dir = tempfile.mkdtemp()
        source_path = os.path.join(tmp_dir, task_file_name)

        import torch
        if isinstance(payload, dict):
            torch.save(payload, source_path)
        else:
            with open(source_path, 'wb') as f:
                import pickle
                pickle.dump(payload, f)

        self.gc.uploadFileToFolder(self.folder_id, source_path, task_file_name)
        self._create_marker_item(marker_name)

    def wait_for_task_ready(self, round_num: int, timeout=300.0, poll_interval=2.0) -> bool:
        """Clients poll until the Hub has uploaded the task for a specific round."""
        marker = f'task_{round_num}_ready'
        return self._wait_for_marker(marker, timeout, poll_interval)

    def read_task(self, round_num: int):
        """Client reads the global model from Girder."""
        task_file_name = f'model_epoch_{int(round_num)}.pt'
        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, task_file_name)

        items = self._get_current_items()
        target_item = next((i for i in items if task_file_name in i.get('name', '')), None)

        if not target_item:
            return None
            
        files = list(self.gc.listFile(target_item['_id'], limit=1))
        if files:
            self.gc.downloadFile(files[0]['_id'], dest_path)
            import torch
            return torch.load(dest_path, map_location='cpu')

    def write_result(self, client_id: int, round_num: int, payload):
        """Client writes its update to Girder."""
        result_file_name = f'result_{int(client_id)}_epoch_{int(round_num)}.pt'

        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, result_file_name)

        import torch
        if isinstance(payload, dict):
            torch.save(payload, dest_path)
        
        self.gc.uploadFileToFolder(self.folder_id, dest_path, result_file_name)
        marker_name = f'completed_{client_id}_{round_num}'
        self._create_marker_item(marker_name)

    def wait_for_clients_complete(
            self, round_num: int, total_clients: int, timeout=300.0, poll_interval=2.0
    ) -> bool:
        """Hub polls until all clients have uploaded results."""
        deadline = time.time() + timeout
        completed_count = 0
        
        while (completed_count < total_clients) and (time.time() < deadline):
            items = self._get_current_items()
            
            for item in items:
                name = item['name']
                if f'completed_' in name and f'{round_num}' in name:
                    completed_count += 1
            
            if completed_count < total_clients:
                time.sleep(poll_interval)

        return completed_count >= total_clients

    def read_all_results(self, round_num: int) -> list:
        """Hub reads results from all clients."""
        items = self._get_current_items()
        results = []

        for item in items:
            if 'result_' in item['name'] and f'epoch_{int(round_num)}' in item['name']:
                tmp_dir = tempfile.mkdtemp()
                dest_path = os.path.join(tmp_dir, item['name'])
                files = list(self.gc.listFile(item['_id'], limit=1))
                if files:
                    self.gc.downloadFile(files[0]['_id'], dest_path)
                    import torch as torch_module
                    results.append(torch_module.load(dest_path, map_location='cpu'))
        return results

    def write_notification(self, msg_type: str):
        marker_name = f'notification_{msg_type}'
        self._create_marker_item(marker_name)

    def read_notifications(self) -> list:
        items = self._get_current_items()
        if any('notification_STOP' in item['name'] for item in items):
            return [True]
        return []