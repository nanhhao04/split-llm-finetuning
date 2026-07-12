# SplitFedLLM
## Setup environment
When executing on DAI, access to a virtual environment is required.
```commandline
source sl/bin/activate
```
## Configuration
Application configuration is in the `config.yaml` file:
```yaml
name: SplitFedLLM
server:
  global-round: 1
  clients:
    - 1
    - 1
  cut-layers: 1
  model-name: Bert
  data-name: AGNEWS
  model:
    Bert:
      n_block: 12
  parameters:
    load: True
    save: True
  validation: True
  data-distribution:
    non-iid: False
    num-sample: 40
    num-label: 4
    dirichlet:
      alpha: 1
    refresh-each-round: True
  random-seed: 1

rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

log_path: .
debug_mode: True

learning:
  learning-rate: 0.00001
  weight-decay: 0.01
  batch-size: 2
  control-count: 1
  clip-grad-norm: 0.0

fine-tune:
  client: False
  server: False
  LoRA:
    r: 8
    alpha: 16

bottleneck:
  enable: True
  bottleneck_dim: 128

quantization:
  enable: True

```
## How to Run
### Server
```commandline
python server.py
```
### Client
```commandline
python client.py --layer_id 1
```
Where:
- `--layer_id` is the index of client's layer, start from 1

## Training Modes

The framework supports three training protocols, controlled entirely via `config.yaml`.

| Field | **SL** | **CE-SL** | **CA-SL** |
|---|---|---|---|
| `bottleneck.enable` | `False` | `True` | `True` |
| `fine-tune.client` | `True` | `False` | `False` → auto `True` (Phase II) |
| `fine-tune.server` | `True` | `True` | `True` |
| `phase.enable` | `False` | `False` | `True` |

### SL — Standard Split Learning
Full activations are transmitted across the split point. LoRA is applied on both client and server sides.
```yaml
fine-tune:
  client: True
  server: True
  LoRA:
    r: 8
    alpha: 16

bottleneck:
  enable: False
  bottleneck_dim: 128

phase:
  enable: False
  thresh_val_loss: 0.005
  window: 3
```

### CE-SL — Communication-Efficient Split Learning
Bottleneck compression is kept on from start to finish (no phase switching). The client trunk is frozen; only the server-side LoRA is updated.
```yaml
fine-tune:
  client: False
  server: True
  LoRA:
    r: 8
    alpha: 16

bottleneck:
  enable: True
  bottleneck_dim: 32   # smaller = more compression

phase:
  enable: False        # stay in Phase I forever
  thresh_val_loss: 0.005
  window: 3
```

### CA-SL — Communication-Aware Split Learning (2-phase)
Starts in Phase I (bottleneck on, client frozen). Automatically switches to Phase II (bottleneck off, full LoRA) once validation loss stays below `thresh_val_loss` for `window` consecutive rounds.
```yaml
fine-tune:
  client: False        # Phase I initial value; auto set to True at Phase II
  server: True
  LoRA:
    r: 8
    alpha: 16

bottleneck:
  enable: True         # Phase I initial value; auto set to False at Phase II
  bottleneck_dim: 32

phase:
  enable: True         # activate the phase-switching algorithm
  thresh_val_loss: 0.005   # epsilon threshold
  window: 3                # p = number of rounds to observe
```
