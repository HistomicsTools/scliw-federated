#!/usr/bin/env python3
# /// script
# requires-python = '>=3.10'
# dependencies = [
#     'girder-client',
#     'nvflare',
# ]
# ///

import os
import tempfile
import time
import json

import girder_client


class GirderDropbox:
    """
    A DropBox implementation that routes file transfers via a Girder workspace folder.
    This preserves the `FileTransferDropbox` requirements of the NVFlare
    architecture while leveraging your existing scliw_federated artifact exchange.
    """

    def __init__(self, girder_url: str, girder_token: str, work_path: str):
        self.base_dir = work_path
        self.gc = girder_client.GirderClient(apiUrl=girder_url)
        self.gc.token = girder_token

        workspace = self.gc.get('resource/lookup', parameters={'path': work_path})
        if not workspace:
            raise FileNotFoundError(f"Girder Workspace '{work_path}' not found in Girder.")

        self.folder_id = workspace['_id']

    def _upload_attachment(self, source_path: str, filename: str) -> None:
        """Helper to upload a file attachment to the workspace folder."""
        self.gc.uploadFileToFolder(
            self.folder_id,
            source_path,
            filename,
        )

    def _download_attachment(self, filename: str, dest_path: str) -> bool:
        """Helper to find an item by name and download its first attachment."""
        items = list(self.gc.listItem(self.folder_id))
        target_item = None

        for item in items:
            if filename in item['name']:
                target_item = item
                break

        if not target_item:
            return False

        files = list(self.gc.listFile(target_item['_id'], limit=1))
        if files:
            self.gc.downloadFile(files[0]['_id'], dest_path)
            return True
        return False

    def _create_marker_item(self, marker_name: str) -> None:
        """Create a trigger/completed marker Item in Girder."""
        self.gc.createItem(
            parentFolderId=self.folder_id,
            name=marker_name,
            metadata={"type": "marker", "source": "nvflare_bus"}
        )

    def _wait_for_marker(self, marker_name: str, timeout: float = 300.0, poll_interval: float = 1.0) -> bool:
        """Poll until a marker Item exists in the workspace."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            items = list(self.gc.listItem(self.folder_id))
            for item in items:
                if item['name'] == marker_name:
                    return True
            time.sleep(poll_interval)
        return False

    def write_task(self, round_num: int, payload):
        """Server/Hub writes the global model (or trigger data) to Girder."""
        task_folder_marker = f'task_{round_num}_ready'
        task_file_name = f'model_epoch_{int(round_num)}.pt'

        tmp_dir = tempfile.mkdtemp()
        source_path = os.path.join(tmp_dir, task_file_name)

        import torch
        if isinstance(payload, dict):
            torch.save(payload, source_path)
        else:
            with open(source_path, 'wb') as f:
                import pickle
                pickle.dump(payload, f)

        # 1. Upload the payload attachment
        self._upload_attachment(source_path, task_file_name)

        # 2. Create the marker Item to signal readiness to clients
        self._create_marker_item(task_folder_marker)

    def wait_for_task_ready(self, round_num: int, timeout=300.0, poll_interval=1.0) -> bool:
        """Clients poll until the Hub has uploaded the task."""
        ready_marker = f'task_{round_num}_ready'
        return self._wait_for_marker(ready_marker, timeout, poll_interval)

    def read_task(self, round_num: int):
        """Client reads the global model from Girder."""
        task_file_name = f'model_epoch_{int(round_num)}.pt'

        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, task_file_name)

        success = self._download_attachment(task_file_name, dest_path)
        if not success:
            raise RuntimeError(f"Could not read task {round_num} from Girder DropBox")

        import torch
        with open(dest_path, 'rb') as f:
            return torch.load(f, map_location='cpu')

    def write_result(self, client_id: int, round_num: int, payload):
        """Client writes its update to Girder."""
        result_file_name = f'result_{int(client_id)}_epoch_{int(round_num)}.pt'

        tmp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(tmp_dir, result_file_name)

        import torch
        if isinstance(payload, dict):
            torch.save(payload, dest_path)
        else:
            with open(dest_path, 'wb') as f:
                import pickle
                pickle.dump(payload, f)

        self._upload_attachment(dest_path, result_file_name)

        completion_marker = f'completed_{client_id}_{round_num}'
        self._create_marker_item(completion_marker)

    def wait_for_clients_complete(
            self, round_num: int, total_clients: int, timeout=300.0, poll_interval=1.0
    ) -> bool:
        """Hub polls until all clients have uploaded results."""
        deadline = time.time() + timeout
        completed_count = 0

        while (completed_count < total_clients) and (time.time() < deadline):
            items = list(self.gc.listItem(self.folder_id))

            for item in items:
                # Check if this item represents a completed result for ANY client
                if f'completed_' in item['name'] and f'{round_num}' in item['name']:
                    try:
                        parts = item['name'].split('_')
                        # 'completed_X_Y_roundZ' is the expected format from write_result
                        if len(parts) >= 3 and parts[0] == 'completed':
                            completed_count += 1
                    except Exception:
                        pass

            if completed_count < total_clients:
                time.sleep(poll_interval)

        return completed_count >= total_clients

    def read_all_results(self, round_num: int) -> list:
        """Hub reads results from all clients."""
        items = list(self.gc.listItem(self.folder_id))
        results = []

        for item in items:
            if 'result_' in item['name'] and f'epoch_{int(round_num)}' in item['name']:
                tmp_dir = tempfile.mkdtemp()
                dest_path = os.path.join(tmp_dir, item['name'])

                files = list(self.gc.listFile(item['_id'], limit=1))
                if files:
                    self.gc.downloadFile(files[0]['_id'], dest_path)

                    if item['name'].endswith('.pt'):
                        import torch as torch_module
                        results.append(torch_module.load(dest_path, map_location='cpu'))
                    else:
                        with open(dest_path, 'r') as f:
                            results.append(json.load(f))
        return results

    def write_notification(self, msg_type: str, data):
        """Sends a stop signal or similar control message."""
        marker_name = f'notification_{msg_type}'
        self._create_marker_item(marker_name)

    def read_notifications(self) -> list:
        """Polls for notifications (e.g., shutdown/stop)."""
        items = list(self.gc.listItem(self.folder_id))
        if any('notification_STOP' in item['name'] for item in items):
            return [True]
        return []


class GirderEventBus:
    """
    A bridge between NVFlare's internal messaging and our Girder-based DropBox.
    """

    def __init__(self, dropbox: GirderDropbox):
        self.dropbox = dropbox
        self.task_queue = []

    def get_event_bus(self):
        return None  # NVFlare uses the executor/dropbox for state sync in this setup.
