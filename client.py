import argparse
import csv
import json
import socket
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# imports for decision tree logic
import math
import numpy as np

# Need column names for logic
AttributeNames: Optional[np.ndarray] = None

# MY FUNCTIONS #
################
def DEBUG(*args, **kwargs):
    return

def entropy(y):
    # Calculate Information Entropy for Distribution y
    # takes an array (y), gives 1 number (entropy). I think this is the log2 function thing
    # information gain/entropy = -p(x)log2(p(x)) summed over all possible values of the variable
    # ex. -.1*log2(.1) - .9*log2(.9)
    if len(y) == 0:
        return 0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, splits):
    # Calculate Information Gain, given a distribution y and a set of indicies splits
    # array of arrays or smtn. array of arrays of indicies for splits?
    # ah, so its like "I'll make a list out of the og one like this: first split has index 0,1,8 from og list, second has 2,3,9, and third has 4,5,6,7. what is the info gain if I split like this?"
    # so return just info gain number
    y_splits = [ [] for i in range(len(splits)) ]
    for i in range(len(splits)):
        for index in splits[i]:
            # MODIFIED: Removed float conversion to support string labels
            y_splits[i].append(y[index])
    
    total_samples = sum([len(x) for x in y_splits])
    if total_samples == 0:
        return 0
        
    children_entropy = 0
    for y_split in y_splits:
        children_entropy += (len(y_split) / total_samples) * entropy(y_split)

    parent_entropy = entropy(y)

    info_gain = parent_entropy - children_entropy

    return info_gain

def majority_label(y):
    if list(y) == []:
        print("ERROR: list y is empty, cannot make prediction")
        return -1
    return Counter(y).most_common(1)[0][0]

# Finding the best split
def best_split(X, y, split_features):
    DEBUG("")
    DEBUG("best_split() entry")
    n, m = X.shape # height, width
    best_gain, best_feature, best_threshold, best_splits = -1, None, None, None
    DEBUG("BEST SPLIT STUFF -- n =", n, "m =", m)

    for j in range(m): # for column (feature) index in <num collumns>
        DEBUG("")
        if AttributeNames[j] in split_features:
            DEBUG("we already did this feature (", AttributeNames[j], "), skipping...")
            continue
        DEBUG("hey were doing category", AttributeNames[j], "rn btw")
        col = X[:, j]
        DEBUG("col is", col)
        info_gain = -1
        thresh = None
        splits = None
        try: # NUMERICAL DATA
            col = col.astype(float)
            values = list(set(col.tolist()))
            values.sort()
            DEBUG("values are", values)
            for value in values:
                num_splits = [[], []]
                for i in range(len(col)): # split into  <= or > thresh
                    if col[i] <= value:
                        num_splits[0].append(i)
                    else:
                        num_splits[1].append(i)
                num_gain = information_gain(y, num_splits)
                if num_gain > info_gain:
                    info_gain = num_gain
                    thresh = value
                    splits = num_splits

        except ValueError as e: # CATEGORICAL DATA (bc we couldnt convert to int)
            # get array of arrays, sub-arrays contain indexes all datapoints with that feature value
            # ie, [[0,1,2],[3,4,5]] means feature values in column are like ['A','A','A', 'B','B','B']
            categories = list(set(col))
            categories.sort()
            DEBUG("categories are", categories)
            splits = [[] for i in range(len(categories))]
            for i in range(len(col)):
                splits[categories.index(col[i])].append(i)
            info_gain = information_gain(y, splits)
        except Exception as e:
            print("got exception:", e)

        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = AttributeNames[j]
            best_threshold = thresh
            best_splits = splits
            DEBUG("found new best split: best_gain = ", best_gain, " best_feature = ", best_feature, " best_threshold = ", best_threshold, " best_splits = ", best_splits )
        else:
            DEBUG("not a better split (only", info_gain, ")")


    return best_feature, best_threshold, best_gain, best_splits

# Tree Node
class Node:
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.branches = {}


# Recursive tree fitting
def fit_tree(X, y, depth=0, max_depth=None, split_features=[]):
    DEBUG("fit_tree() entry (depth", depth, ")")


    if len(set(y)) == 1 or len(y) == 0 or (max_depth and depth >= max_depth): # Max Depth or pure node, make leaf
        DEBUG("leaf time yippee we're done")
        node = Node(is_leaf = True, prediction = majority_label(y))
        return node

    # At this point, not a leaf, need to split more

    #Get the best split
    feat, thr, gain, splits = best_split(X, y, split_features)
    if feat is None or gain <= 1e-12: # no good split, just give up here 'cause it aint getting better
        DEBUG("the \"best split\" was mid af so we stopping here")
        node = Node(is_leaf = True, prediction = majority_label(y))
        return node
    DEBUG("best split returned: feat", feat, "thr", thr, "gain", gain, "splits", splits)

    # at this point, we *have* a split that is good enough
    # Todo: Create internal node and recurse for children
    node = Node(feature=feat, threshold=thr)
    DEBUG("RECURSION IS FUN!!! about to make node, and recurse on branches")
    if thr is not None:
        DEBUG("numerical data recursion")
        node.branches = {
            "<=": fit_tree(np.array([X[i] for i in splits[0]]), np.array([y[i] for i in splits[0]]), depth+1, max_depth, split_features),
            ">":  fit_tree(np.array([X[i] for i in splits[1]]), np.array([y[i] for i in splits[1]]), depth+1, max_depth, split_features)
        }
    else:
        pass # NO CATEGORICAL DATA IN TOURNAMENT
        # DEBUG("categorical data recursion")
        # values = list(set(X[:,(AttributeNames.tolist().index(feat))]))
        # values.sort()
        # node.branches = {value: fit_tree(np.array([X[i] for i in splits[values.index(value)]]), np.array([y[i] for i in splits[values.index(value)]]), depth+1, max_depth, split_features+[feat]) for value in values}

    return node

# Prediction
def predict_one(node, x):
    while not node.is_leaf:
        if node.threshold is not None:
            v = float(x[AttributeNames.tolist().index(node.feature)])
            key = "<=" if v <= node.threshold else ">"
        else:
            key = x[AttributeNames.tolist().index(node.feature)]
        node = node.branches.get(key, None)
        if node is None:
            break
    # MODIFIED: Removed int() conversion to return original label type
    return node.prediction if node else None

def predict(node, X):
    return np.array([predict_one(node, row) for row in X])

# Printing the tree
def plot_tree(node, attr, depth=0):
    if depth==0:
        print("root")
    if node.is_leaf:
        # MODIFIED: Handled potential non-numeric prediction for printing
        print(' | '*(depth) + " +-> " + "Predict:", node.prediction)
        return
    if node.threshold is not None:
        print(' | '*(depth) + " +-> " + node.feature + "<=" + str(node.threshold))
        plot_tree(node.branches["<="], attr, depth+1)
        print(' | '*(depth) + " +-> " + node.feature + ">" + str(node.threshold))
        plot_tree(node.branches[">"], attr, depth+1)
    else:
        for key in node.branches.keys():
            print(' | '*(depth) + " +-> " + node.feature + " is " + key)
            plot_tree(node.branches[key], attr, depth+1)

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
        self.train_features: List[List[float]] = []
        self.train_labels: List[str] = []
        self.majority_label: Optional[str] = None
        self.dataset_payload: Optional[Dict[str, List]] = None

        # TREE STRUCTURE
        self.decision_tree_root: Optional[Node] = None
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
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} does not exist")

        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            header_row_index = 0
            header: Optional[List[str]] = None
            for row_index, row in enumerate(reader, start=1):
                if not row or row[0].startswith("#"):
                    continue
                header = row
                header_row_index = row_index
                break

            if header is None:
                raise ValueError("Dataset is empty or only contains comments")
            feature_count = len(header) - 1
            if feature_count < 1:
                raise ValueError("Dataset must include at least one attribute column before the class column")

            for row_index, row in enumerate(reader, start=header_row_index + 1):
                if not row or row[0].startswith("#"):
                    continue
                if len(row) != len(header):
                    raise ValueError(f"Row {row_index} has {len(row)} columns; expected {len(header)} based on header")
                try:
                    feature_values = [float(value) for value in row[:-1]]
                except ValueError:
                    raise ValueError(f"Row {row_index} contains non numeric feature values") from None

                label = row[-1].strip()
                if not label:
                    raise ValueError(f"Row {row_index} has an empty class label")

                features.append(feature_values)
                labels.append(label)

        if not features:
            raise ValueError("Dataset is empty after parsing")
        if len(features) > 1000:
            raise ValueError("Dataset exceeds 1000 samples")
        if len(set(labels)) > 5:
            raise ValueError("Dataset exceeds 5 classes")

        return {"features": features, "labels": labels}
    def _handle_train_data(self, payload: Dict[str, List]) -> None:
        global AttributeNames
        self.train_features = payload.get("features", [])
        self.train_labels = payload.get("labels", [])
        if not self.train_features:
            print("Received empty training dataset")
            return
        print(
            "Training data received. Samples: "
            f"{len(self.train_features)}"
        )

        X_train = np.array(self.train_features)
        # MODIFIED: Removed .astype(float) to handle string labels
        y_train = np.array(self.train_labels)

        num_features = X_train.shape[1]
        AttributeNames = np.array([f"feature_{i+1}" for i in range(num_features)])

        self.fallback_label = str(Counter(y_train).most_common(1)[0][0]) # default to most common label

        print("training tree with", len(X_train), "samples...")

        self.decision_tree_root = fit_tree(X_train, y_train, max_depth=20)

    def _predict(self, payload: Dict[str, object]) -> str:
        features_to_predict = payload.get("features")

        prediction = None
        if self.decision_tree_root is not None:
            prediction = predict_one(self.decision_tree_root, features_to_predict)
        
        if prediction is not None:
            return str(prediction)
        
        print("WARNING!!! Looks like ur tree isn't trained or failed to predict. maybe fix it. gonna just do my best ig...")
        return self.fallback_label if self.fallback_label else "unknown"


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
    client = FederatedClient(host=args.host, port=args.port, dataset_path=args.dataset)
    try:
        client.run()
    except KeyboardInterrupt:
        print("Client interrupted by user")


if __name__ == "__main__":
    main()