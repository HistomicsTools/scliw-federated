#!/usr/bin/env python3

import os
import tempfile
import time

import girder_client


class GirderBridge:
    """
    A bridge implementation that routes artifact transfer via a Girder
    workspace folder.  Replaces the standard FLARE FileTransfer mechanism
    with a polling-based Girder interface.
    """

    def __init__(self, girder_url: str, girder_token: str, work_path: str):
        self.base_dir = work_path
        self.gc = girder_client.GirderClient(apiUrl=girder_url)
        self.gc.token = girder_token
        workspace = self.gc.get('resource/lookup', parameters={'path': work_path})
        if not workspace:
            raise FileNotFoundError(f"Girder Workspace '{work_path}' not found in Girder.")
        self.folder_id = workspace['_id']

    def get_current_items(self) -> list:
        """Helper to get a fresh list of items from Girder."""
        return list(self.gc.listItem(self.folder_id))

    def wait_for_marker(self, marker: str | list, timeout: float = 300.0,
                         poll_interval: float = 2.0) -> bool | str:
        """Poll until a specific Marker Item exists in the workspace."""
        marker_list = [marker] if isinstance(marker, str) else marker
        deadline = time.time() + timeout
        while time.time() < deadline:
            items = self.get_current_items()

            for item in items:
                if item['name'] in marker_list:
                    return item['name']

            time.sleep(poll_interval)
        return False

    def create_marker_item(self, marker_name: str) -> None:
        """Create a trigger/completed marker Item in Girder."""
        try:
            self.gc.createItem(
                parentFolderId=self.folder_id, name=marker_name, metadata={'type': 'marker'})
        except Exception as e:
            print(f"[GirderBridge] Warning creating marker '{marker_name}': {e}")

    def write_done(self):
        self.create_marker_item('task_done')

    def write_task(self, round_num: int, payload):
        """Hub writes the global model (or trigger data) to Girder."""
        import torch

        task_file_name = f'model_epoch_{round_num}.pt'
        marker_name = f'task_{round_num}_ready'
        tmp_dir = tempfile.mkdtemp()
        source_path = os.path.join(tmp_dir, task_file_name)
        if isinstance(payload, dict):
            torch.save(payload, source_path)
        else:
            with open(source_path, 'wb') as f:
                import pickle
                pickle.dump(payload, f)
        self.gc.uploadFileToFolder(self.folder_id, source_path, task_file_name)
        # Verify the file is actually visible in Girder before signaling readiness
        items = list(self.gc.listItem(self.folder_id))
        if any(item['name'] == task_file_name for item in items):
            self.create_marker_item(marker_name)

    def wait_for_task_ready(self, marker: str, done_marker: str, timeout=300.0, poll_interval=2.0) -> bool | str:
        """Clients poll until the Hub has uploaded the task for a specific round."""
        return self.wait_for_marker([marker, done_marker], timeout, poll_interval)

    def read_task(self, round_num: int):
        """Client reads the global model from Girder."""
        task_file_name = f'model_epoch_{round_num}.pt'
        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, task_file_name)

        items = self.get_current_items()
        target_item = next((i for i in items if task_file_name in i.get('name', '')), None)
        if not target_item:
            return None
        files = list(self.gc.listFile(target_item['_id'], limit=1))
        if files:
            self.gc.downloadFile(files[0]['_id'], dest_path)
            import torch
            return torch.load(dest_path, map_location='cpu')

    def write_result(self, client_id, round_num: int, payload):
        """Client writes its update to Girder."""
        import torch

        safe_client_id = str(client_id)
        result_file_name = f'result_{safe_client_id}_epoch_{round_num}.pt'
        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, result_file_name)
        if isinstance(payload, dict):
            torch.save(payload, dest_path)
        self.gc.uploadFileToFolder(self.folder_id, dest_path, result_file_name)
        # Verify the file is actually visible in Girder before signaling completion
        items = list(self.gc.listItem(self.folder_id))
        if any(item['name'] == result_file_name for item in items):
            marker_name = f'completed_{safe_client_id}_{round_num}'
            self.create_marker_item(marker_name)

    def wait_for_clients_complete(
            self, round_num: int, total_clients: int, timeout=300.0, poll_interval=2.0
    ) -> bool:
        """Hub polls until all clients have uploaded results."""
        deadline = time.time() + timeout
        completed_count = 0

        while (completed_count < total_clients) and (time.time() < deadline):
            items = self.get_current_items()

            completed_count = 0
            for item in items:
                name = item['name']
                if 'completed_' in name and name.endswith(f'_{round_num}'):
                    completed_count += 1

            if completed_count < total_clients:
                time.sleep(poll_interval)

        return completed_count >= total_clients

    def read_all_results(self, round_num: int) -> list:
        """Hub reads results from all clients."""
        items = self.get_current_items()
        results = []
        seen = set()
        for item in items:
            name = item['name']
            if 'result_' in name and f'epoch_{round_num}.' in name:
                safe_name = self.sanitize_filename(name)
                if safe_name in seen:
                    continue
                seen.add(safe_name)
                tmp_dir = tempfile.mkdtemp()
                dest_path = os.path.join(tmp_dir, safe_name)
                files = list(self.gc.listFile(item['_id'], limit=1))
                if files:
                    import torch

                    self.gc.downloadFile(files[0]['_id'], dest_path)
                    loaded_data = torch.load(dest_path, map_location='cpu')
                    if loaded_data is not None:
                        results.append(loaded_data)
                        print(f'read {name}')
        return results

    def write_notification(self, msg_type: str):
        """Sends a stop signal or similar control message."""
        marker_name = f'notification_{msg_type}'
        self.create_marker_item(marker_name)

    def read_notifications(self) -> list:
        items = self.get_current_items()
        if any('notification_STOP' in item['name'] for item in items):
            return [True]
        return []

    def sanitize_filename(self, filename):
        """Sanitize filename for filesystem temporary storage."""
        return filename.replace(',', '_').replace('\\', '_')
