import torch
import os
import time
import random
import pika
import pickle
import sys
import numpy as np
import copy
import src.Log
import src.Utils

from src.model.BERT import BERT
from src.model.GPT2 import GPT2
from src.val.get_val import get_val
from src.phase_manager import PhaseManager

class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model-name"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.cut_layers = config["server"]["cut-layers"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]

        # Clients
        self.total_block = config["server"]["model"][self.model_name]["n_block"]
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.weight_decay = config["learning"]["weight-decay"]
        self.control_count = config["learning"]["control-count"]
        self.clip_grad_norm = config["learning"]["clip-grad-norm"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
        self.refresh_each_round = self.data_distribution["refresh-each-round"]
        self.random_seed = config["server"]["random-seed"]
        self.label_counts = None

        # Fine tune config
        self.fine_tune_config = config['fine-tune']

        # Bottleneck + Quantization
        self.bottleneck_config = config['bottleneck']

        # Phase manager (CASL)
        phase_cfg = config.get('phase', {})
        thresh = phase_cfg.get('thresh_val_loss', 0.01)
        window = phase_cfg.get('window', 3)
        phase_enable = phase_cfg.get('enable', True)
        self.phase_manager = PhaseManager(
            thresh_val_loss=thresh,
            window=window,
            enable=phase_enable,
        )

        if self.random_seed:
            random.seed(self.random_seed)

        log_path = config["log_path"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.count_update = [0 for _ in range(len(self.total_clients))]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.count_notify = 0
        self.responses = {}
        self.list_clients = []
        self.round_result = True

        self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
        self.global_client_sizes = [[] for _ in range(len(self.total_clients))]
        self.avg_state_dict = []
        self.accumulated_comm_bytes = 0

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def distribution(self):
        if self.model_name == "BERT":
            if self.non_iid:
                label_distribution = np.array([[0.05, 0.05, 0.05, 0.85],
                                               [0.05, 0.05, 0.85, 0.05],
                                               [0.05, 0.85, 0.05, 0.05],
                                               [0.85, 0.05, 0.05, 0.05]
                                               ])
                self.label_counts = (label_distribution * self.num_sample).astype(int)

            else:
                self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)
        else:
            self.label_counts = [[self.num_sample] for _ in range(self.total_clients[0])]

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            if (str(client_id), layer_id) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")

                self.distribution()

                self.logger.log_info(f"Start training round 1")
                self.notify_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}

            self.count_notify += 1

            if self.count_notify == self.total_clients[0]:
                self.count_notify = 0
                src.Log.print_with_color(f"Received all the finish training notification", "yellow")

                for (client_id, layer_id) in self.list_clients:
                    self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            # self.distribution()
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")

            self.count_update[layer_id - 1] += 1
            if not result:
                self.round_result = False

            # Save client's model parameters
            if self.save_parameters and self.round_result:
                model_state_dict = message["parameters"]
                client_size = message["size"]
                self.global_model_parameters[layer_id - 1].append(model_state_dict)
                self.global_client_sizes[layer_id - 1].append(client_size)

            # If consumed all client's parameters
            if self.count_update == self.total_clients:
                src.Log.print_with_color("Collected all parameters.", "yellow")
                if self.save_parameters and self.round_result:

                    self.avg_all_parameters()
                    comm_cost_str, round_bytes = src.Utils.print_comm_costs_round(
                        self.model_name,
                        self.total_clients,
                        self.batch_size,
                        self.global_client_sizes,
                        self.avg_state_dict,
                        self.accumulated_comm_bytes,
                        phase=self.phase_manager.current_phase,
                        bottleneck_dim=self.bottleneck_config.get('bottleneck_dim', 128)
                    )
                    self.accumulated_comm_bytes += round_bytes
                    self.logger.log_info(comm_cost_str)
                    self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
                    self.global_client_sizes = [[] for _ in range(len(self.total_clients))]

                self.count_update = [0 for _ in range(len(self.total_clients))]
                # Test
                if self.save_parameters and self.validation and self.round_result:
                    state_dict_full = self.concatenate()

                    success, val_loss = get_val(self.model_name, state_dict_full, self.logger, self.num_sample)
                    if not success:
                        self.logger.log_warning("Training failed!")
                        self.round = 0
                    else:
                        # Log trang thai PhaseManager
                        src.Log.print_with_color(
                            self.phase_manager.status_str(), "yellow")
                        self.logger.log_info(self.phase_manager.status_str())

                        # Kiem tra chuyen pha (chi khi enable=True)
                        transitioned = self.phase_manager.update(val_loss)
                        if transitioned:
                            self.phase_manager.apply_phase2_config(
                                self.fine_tune_config, self.bottleneck_config)
                            msg = (
                                f"[PHASE TRANSITION] Pha I -> Pha II "
                                f"(thresh={self.phase_manager.thresh:.4f}, "
                                f"window={self.phase_manager.window} rounds)"
                            )
                            src.Log.print_with_color(msg, "green")
                            self.logger.log_info(msg)

                        # Save to files
                        torch.save(state_dict_full, f'{self.model_name}.pt')
                        self.round -= 1
                    self.avg_state_dict = []
                else:
                    self.round = 0

                # Start a new training round
                self.round_result = True

                if self.round > 0:
                    self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                    if self.save_parameters:
                        self.notify_clients()
                    else:
                        self.notify_clients(register=False)
                else:
                    self.logger.log_info("Stop training !!!")
                    self.notify_clients(start=False)
                    sys.exit()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True):

        # Send message to clients when consumed all clients
        if self.model_name == 'BERT':
            klass = BERT
        else:
            klass = GPT2
        stt = -1

        # ----------------------------------------------------
        # --- Load model files ONCE before the client loop ---
        # ----------------------------------------------------
        
        state_dict_layer1 = None
        state_dict_layer2 = None
        bottleneck_state_dict = None

        if start and self.load_parameters and register:
            filepath = f'{self.model_name}.pt'
            bottleneck_dim = self.bottleneck_config['bottleneck_dim']
            bottleneck_path = os.path.join(
                'bottleneck model pretrain',
                f'bottleneck{self.cut_layers}_{bottleneck_dim}.pt'
            )

            if os.path.exists(filepath):
                full_state_dict = torch.load(filepath, weights_only=True)

                # Build state_dict for layer 1 clients
                if self.bottleneck_config["enable"]:
                    model1 = klass(layer_id=1, n_block=self.cut_layers, reduce_comm=True,
                                   bottleneck_dim=self.bottleneck_config['bottleneck_dim'])
                else:
                    model1 = klass(layer_id=1, n_block=self.cut_layers)
                sd1 = model1.state_dict()
                for key in sd1.keys():
                    if key in full_state_dict:
                        sd1[key] = full_state_dict[key]
                state_dict_layer1 = sd1

                # Build state_dict for layer 2 clients
                if self.bottleneck_config["enable"]:
                    model2 = klass(layer_id=2, n_block=12 - self.cut_layers, reduce_comm=True,
                                   bottleneck_dim=self.bottleneck_config['bottleneck_dim'])
                else:
                    model2 = klass(layer_id=2, n_block=12 - self.cut_layers)
                sd2 = model2.state_dict()
                sd2 = src.Utils.change_keys(sd2, self.cut_layers, True)
                for key in sd2.keys():
                    if key in full_state_dict:
                        sd2[key] = full_state_dict[key]
                sd2 = src.Utils.change_keys(sd2, self.cut_layers, False)
                state_dict_layer2 = sd2
                src.Log.print_with_color(f"Load pretrain model successfully", "green")
            else:
                src.Log.print_with_color(f"File {filepath} does not exist. Starting from scratch.", "yellow")
                self.logger.log_info(f"File {filepath} does not exist.")

            if os.path.exists(bottleneck_path):
                bottleneck_state_dict = torch.load(bottleneck_path, weights_only=True,
                                                   map_location=torch.device("cpu"))
                src.Log.print_with_color(f"Load pretrain bottleneck model successfully", "green")
            else:
                src.Log.print_with_color(f"File {bottleneck_path} does not exist. Starting bottleneck from scratch.", "yellow")

        # --- Distribute to each client ---
        for (client_id, layer_id) in self.list_clients:
            if start:
                state_dict = copy.deepcopy(state_dict_layer1) if layer_id == 1 else copy.deepcopy(state_dict_layer2)

                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "model_name": self.model_name,
                            "parameters": state_dict,
                            "bottleneck": copy.deepcopy(bottleneck_state_dict),
                            "cut_layers": self.cut_layers,
                            "total_block": self.total_block,
                            "fine_tune_config": self.fine_tune_config,
                            "bottleneck_config": self.bottleneck_config,
                            }

                self.send_to_response(client_id, pickle.dumps(response))

            else:
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!"
                            }
                self.send_to_response(client_id, pickle.dumps(response))

        time.sleep(5)
        if start:
            for (client_id, layer_id) in self.list_clients:
                if layer_id == 1:
                    stt += 1
                response = {"action": "SYN",
                            "label_counts": self.label_counts,
                            "control_count": self.control_count,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "weight_decay": self.weight_decay,
                            "stt": stt,
                            "message": "Synchronize client devices",
                            }
                self.send_to_response(client_id, pickle.dumps(response))

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def avg_all_parameters(self):
        layer_sizes = self.global_client_sizes
        layer_params = self.global_model_parameters

        for layer_idx, list_state_dicts in enumerate(layer_params):
            list_sizes = layer_sizes[layer_idx]
            if not list_state_dicts or not list_sizes:
                self.avg_state_dict.append({})
                continue
            avg_sd = src.Utils.fed_avg_state_dicts(list_state_dicts, weights=list_sizes)
            self.avg_state_dict.append(avg_sd)

    def concatenate(self):
        avg_layers = self.avg_state_dict
        if not avg_layers:
            print(f"Warning: don't has averaged layers, skipping.")

        full_dict = {}
        for idx, layer_dict in enumerate(avg_layers):
            if idx == 0:
                sd = {
                    k: v
                    for k, v in layer_dict.items()
                    if not k.startswith("encoder.")
                }
                full_dict.update(copy.deepcopy(sd))
            else:
                sd = {
                    k: v
                    for k, v in layer_dict.items()
                    if not k.startswith("decoder.")
                }

                sd = src.Utils.change_keys(sd, self.cut_layers, True)

                full_dict.update(copy.deepcopy(sd))

        return full_dict
