import argparse
import csv
import json
import socket
import struct
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# --- Hardcoded Model Import ---
from knn import KNN

# CLIENT.PY LOGIC #
###################

Message = Dict[str, object]

def send_message(conn: socket.socket, message: Message) -> None:
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
    header = _recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    payload = _recv_exact(conn, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


class FederatedClient:
    def __init__(self, host: str, port: int, dataset_path: Optional[Path]) -> None:
        self.host = host
        self.port = port
        self.dataset_path = dataset_path
        self.client_id: Optional[int] = None
        self.dataset_payload: Optional[Dict[str, List]] = None
        
        # --- Model Instantiation ---
        self.model = KNN(k=5)
        print(f"Using model: {self.model.__class__.__name__}")
        
        self.fallback_label: Optional[str] = None

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            print(f"Connected to server {self.host}:{self.port}")
            while True:
                message = receive_message(sock)
                if message is None:
                    print("Server closed the connection")
                    break
                msg_type = message.get("type")
                payload = message.get("payload", {})

                if msg_type == "client_id":
                    self.client_id = payload["id"]
                    print(f"Assigned client id: {self.client_id}")
                elif msg_type == "request_dataset":
                    self._handle_dataset_request(sock)
                elif msg_type == "train_data":
                    self._handle_train_data(payload)
                elif msg_type == "infer_request":
                    prediction = self._predict(payload)
                    response = {"type": "prediction", "payload": {"label": prediction}}
                    send_message(sock, response)
                elif msg_type == "shutdown":
                    print("Shutdown requested by server")
                    break
                else:
                    print(f"Unexpected message type received: {msg_type}")

    def _handle_dataset_request(self, sock: socket.socket) -> None:
        if self.dataset_path is None:
            send_message(sock, {"type": "no_dataset"})
            return

        if self.dataset_payload is None:
            self.dataset_payload = self._load_dataset()
        send_message(sock, {"type": "dataset", "payload": self.dataset_payload})
        print(f"Dataset with {len(self.dataset_payload['features'])} samples sent to server")

    def _load_dataset(self) -> Dict[str, List]:
        features: List[List[float]] = []
        labels: List[str] = []
        path = self.dataset_path
        if path is None:
            return {"features": features, "labels": labels}

        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # Skip header
            for row in reader:
                feature_values = [float(value) for value in row[:-1]]
                label = row[-1].strip()
                if label:
                    features.append(feature_values)
                    labels.append(label)
        return {"features": features, "labels": labels}
        
    def _handle_train_data(self, payload: Dict[str, List]) -> None:
        features = payload.get("features", [])
        labels = payload.get("labels", [])

        print(f"Training data received. Samples: {len(features)}")
        
        X_train_np = np.array(features)
        y_train_np = np.array(labels).astype(float)

        self.model.train(X_train_np, y_train_np)
        
        self.fallback_label = str(Counter(y_train_np).most_common(1)[0][0])

    def _predict(self, payload: Dict[str, object]) -> str:
        features_to_predict = payload.get("features")
        x_test_np = np.array(features_to_predict)
        
        prediction = self.model.predict(x_test_np)
        return str(prediction)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated classification demo client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to CSV dataset (only one client needs to provide this)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    client = FederatedClient(
        host=args.host,
        port=args.port,
        dataset_path=args.dataset,
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("Client interrupted by user")

if __name__ == "__main__":
    main()