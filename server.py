import argparse
import json
import random
import socket
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches


Message = Dict[str, object]


def send_message(conn: socket.socket, message: Message) -> None:
    """Send a length-prefixed JSON message."""
    payload = json.dumps(message).encode("utf-8")
    header = struct.pack("!I", len(payload))
    conn.sendall(header + payload)


def _recv_exact(conn: socket.socket, num_bytes: int) -> Optional[bytes]:
    buffer = bytearray()
    while len(buffer) < num_bytes:
        chunk = conn.recv(num_bytes - len(buffer))
        if not chunk:
            return None
        buffer.extend(chunk)
    return bytes(buffer)


def receive_message(conn: socket.socket) -> Optional[Message]:
    """Receive a length-prefixed JSON message."""
    header = _recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    payload = _recv_exact(conn, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


@dataclass
class ClientEndpoint:
    client_id: int
    connection: socket.socket
    address: Tuple[str, int]


class FederatedServer:
    def __init__(self, host: str, port: int, rounds: int = 10) -> None:
        self.host = host
        self.port = port
        self.rounds = rounds
        self.clients: Dict[int, ClientEndpoint] = {}
        self.dataset_features: List[List[float]] = []
        self.dataset_labels: List[str] = []
        self.dt_features: List[List[float]] = []
        self.dt_labels: List[str] = []
        self.ds_features: List[List[float]] = []
        self.ds_labels: List[str] = []
        self.label_palette: Dict[str, Tuple[float, float, float, float]] = {}
        self.fig = None
        self.ax = None
        self.prediction_labels_added = set()
        self.class_legend_handles: List[patches.Patch] = []
        self.class_legend_artist = None

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(2)
            print(f"Server listening on {self.host}:{self.port}")
            self._accept_clients(server_socket)
            self._request_initial_dataset()
            self._split_dataset()
            self._broadcast_train_data()
            self._visualize_ds()
            self._inference_rounds()
            self._shutdown_clients()

    def _accept_clients(self, server_socket: socket.socket) -> None:
        for client_id in (1, 2):
            conn, address = server_socket.accept()
            endpoint = ClientEndpoint(client_id=client_id, connection=conn, address=address)
            self.clients[client_id] = endpoint
            print(f"Client {client_id} connected from {address}")
            send_message(conn, {"type": "client_id", "payload": {"id": client_id}})

    def _request_initial_dataset(self) -> None:
        for endpoint in self.clients.values():
            send_message(endpoint.connection, {"type": "request_dataset"})

        dataset_received = False
        for endpoint in self.clients.values():
            message = receive_message(endpoint.connection)
            if message is None:
                raise RuntimeError(f"Client {endpoint.client_id} disconnected before sending dataset info")
            msg_type = message.get("type")
            if msg_type == "dataset" and not dataset_received:
                payload = message["payload"]
                self.dataset_features = payload["features"]
                self.dataset_labels = payload["labels"]
                self._prepare_palette()
                self._visualize_full_dataset()
                dataset_received = True
                print(f"Dataset received from client {endpoint.client_id} with {len(self.dataset_features)} samples")
            elif msg_type == "dataset" and dataset_received:
                print(f"Additional dataset from client {endpoint.client_id} ignored (already have one)")
            elif msg_type == "no_dataset":
                print(f"Client {endpoint.client_id} did not provide a dataset")
            else:
                raise RuntimeError(f"Unexpected message {msg_type!r} from client {endpoint.client_id}")

        if not dataset_received:
            raise RuntimeError("No dataset received from any client")

    def _prepare_palette(self) -> None:
        unique_labels = sorted(set(self.dataset_labels))
        cmap = plt.get_cmap("tab10")
        self.label_palette = {
            label: cmap(index % cmap.N)
            for index, label in enumerate(unique_labels)
        }
        self.class_legend_handles = [
            patches.Patch(facecolor=self.label_palette[label], edgecolor='none', label=label)
            for label in unique_labels
        ]

    def _visualize_full_dataset(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        xs = [point[0] for point in self.dataset_features]
        ys = [point[1] for point in self.dataset_features]
        colors = [self.label_palette[label] for label in self.dataset_labels]
        self.ax.scatter(xs, ys, c=colors, s=30, alpha=0.7, edgecolors="none")
        self.ax.set_title("Full Dataset (All Samples)")
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        if self.class_legend_handles:
            self.class_legend_artist = self.ax.legend(
                handles=self.class_legend_handles,
                title="Classes",
                loc="upper right",
            )
        self._refresh_plot()

    def _split_dataset(self) -> None:
        total_samples = len(self.dataset_features)
        indices = list(range(total_samples))
        random.shuffle(indices)
        split_index = max(1, int(0.7 * total_samples))
        dt_indices = set(indices[:split_index])
        self.dt_features = [self.dataset_features[i] for i in dt_indices]
        self.dt_labels = [self.dataset_labels[i] for i in dt_indices]
        self.ds_features = [self.dataset_features[i] for i in indices[split_index:]]
        self.ds_labels = [self.dataset_labels[i] for i in indices[split_index:]]
        print(f"Training subset (dt): {len(self.dt_features)} samples")
        print(f"Holdout subset (ds): {len(self.ds_features)} samples")

    def _broadcast_train_data(self) -> None:
        message = {
            "type": "train_data",
            "payload": {
                "features": self.dt_features,
                "labels": self.dt_labels,
            },
        }
        for endpoint in self.clients.values():
            send_message(endpoint.connection, message)
        print("Training dataset broadcast to all clients")

    def _visualize_ds(self) -> None:
        if not self.ds_features:
            print("Warning: Holdout dataset ds is empty; skipping visualization")
            return
        xs = [point[0] for point in self.ds_features]
        ys = [point[1] for point in self.ds_features]
        colors = [self.label_palette[label] for label in self.ds_labels]
        self.ax.scatter(xs, ys, c=colors, s=60, alpha=0.4, edgecolors="k", linewidths=0.4, label="ds samples")
        self.ax.set_title("Holdout Dataset (ds)")
        if self.class_legend_artist is not None:
            self.ax.add_artist(self.class_legend_artist)
        self._refresh_plot()

    def _inference_rounds(self) -> None:
        if not self.ds_features:
            print("No holdout samples available for inference rounds")
            return

        for round_index in range(1, self.rounds + 1):
            sample_idx = random.randrange(len(self.ds_features))
            features = self.ds_features[sample_idx]
            true_label = self.ds_labels[sample_idx]
            request = {
                "type": "infer_request",
                "payload": {
                    "round": round_index,
                    "sample_index": sample_idx,
                    "features": features,
                },
            }
            print(f"Round {round_index}: dispatching sample {sample_idx} to clients")
            for endpoint in self.clients.values():
                send_message(endpoint.connection, request)

            predictions: Dict[int, str] = {}
            for endpoint in self.clients.values():
                response = receive_message(endpoint.connection)
                if response is None:
                    raise RuntimeError(f"Client {endpoint.client_id} disconnected during inference")
                if response.get("type") != "prediction":
                    raise RuntimeError(f"Unexpected message from client {endpoint.client_id}: {response}")
                payload = response["payload"]
                predictions[endpoint.client_id] = payload["label"]
                print(
                    f"Client {endpoint.client_id} predicted {payload['label']} for sample {sample_idx}"
                )

            self._visualize_predictions(features, true_label, predictions)
            remaining = self.rounds - round_index
            print("true_label was", true_label)
            input(f"Press Enter to continue (remaining rounds: {remaining})...")

    def _visualize_predictions(
        self,
        features: List[float],
        true_label: str,
        predictions: Dict[int, str],
    ) -> None:
        x, y = features
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        max_span = max(x_max - x_min, y_max - y_min)
        if max_span <= 0:
            square_size = 1.0
        else:
            square_size = max(max_span * 0.05, 0.05)
        half = square_size / 2.0

        square = patches.Rectangle(
            (x - half, y - half),
            square_size,
            square_size,
            facecolor='white',
            edgecolor='k',
            linewidth=1.2,
            alpha=0.85,
            zorder=5,
        )
        self.ax.add_patch(square)
        self.ax.plot(
            [x, x],
            [y - half, y + half],
            color='k',
            linewidth=1.0,
            zorder=6,
        )

        client_offsets = {1: -square_size / 4.0, 2: square_size / 4.0}
        for client_id, offset in client_offsets.items():
            label = predictions.get(client_id)
            if label is None:
                continue
            correct = label == true_label
            color = 'blue' if correct else 'red'
            self.ax.text(
                x + offset,
                y,
                label,
                ha='center',
                va='center',
                fontsize=9,
                color=color,
                fontweight='bold',
                zorder=7,
            )

        subtitle = f"Holdout Dataset (ds) - True label: {true_label}"
        self.ax.set_title(subtitle)
        if self.class_legend_artist is not None:
            self.ax.add_artist(self.class_legend_artist)
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        if self.fig is None:
            return
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

    def _shutdown_clients(self) -> None:
        for endpoint in self.clients.values():
            try:
                send_message(endpoint.connection, {"type": "shutdown"})
            except OSError:
                pass
            endpoint.connection.close()
        print("Server shutdown complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated classification demo server")
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind the server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server (default: 5000)")
    parser.add_argument("--rounds", type=int, default=10, help="Number of inference iterations (default: 10)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = FederatedServer(host=args.host, port=args.port, rounds=args.rounds)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer interrupted by user")


if __name__ == "__main__":
    main()