# MultiplexedLightDLGN Forward Compute Structure

`MultiplexedLightDLGN` evaluates the same shared logic network once per class by pairing each layer input with a learned per-layer class code, then reshaping the per-class outputs back into class logits.

Source references:

- [src/light_dlgn/model.py](/home/prof/lgnn3/src/light_dlgn/model.py:168)
- [src/light_dlgn/encoding.py](/home/prof/lgnn3/src/light_dlgn/encoding.py:10)

## Forward Graph

```mermaid
flowchart TD
    X["Input image batch x
    shape: [B, C, H, W]"] --> ENC["Thermometer encode
    encode(x)
    shape: [B, E]
    E = C*H*W*T"]

    CLOGITS["class_code_logits
    learned parameter
    shape: [N, K, D]"] --> EST["Apply estimator
    class_codes(discrete)
    shape: [N, K, D]"]

    ENC --> EXP1["Repeat encoded input across classes
    encoded.unsqueeze(1).expand(-1, K, -1)
    shape: [B, K, E]"]
    EST --> EXP2["Take layer-1 class codes and repeat across batch
    codes[0].unsqueeze(0).expand(B, -1, -1)
    shape: [B, K, D]"]

    EXP1 --> CAT1["Concatenate input bits and layer-1 class code
    cat(..., dim=-1)
    shape: [B, K, E + D]"]
    EXP2 --> CAT1

    CAT1 --> FLAT1["Fold batch and class into one axis
    reshape(B*K, E + D)
    shape: [B*K, E + D]"]
    FLAT1 --> L1["Shared InputWiseLogicLayer 1
    shape: [B*K, W1]"]

    L1 --> RESHAPE["Reshape per-class features
    shape: [B, K, W1]"]
    EST --> EXPNEXT["Take next layer's class codes
    repeat across batch
    shape: [B, K, D]"]
    RESHAPE --> CATNEXT["Concatenate previous layer output
    with next layer class code
    shape: [B, K, W1 + D]"]
    EXPNEXT --> CATNEXT

    CATNEXT --> L2["Shared InputWiseLogicLayer 2
    shape: [B*K, W2]"]
    L2 --> LDOTS["... repeat per layer ..."]
    LDOTS --> LN["Shared InputWiseLogicLayer N
    shape: [B*K, WN]"]

    LN --> GS["GroupSum(num_classes=1, tau)
    sum over final feature group
    shape: [B*K, 1]"]
    GS --> OUT["Reshape to class logits
    view(B, K)
    shape: [B, K]"]
```

## Shape Legend

- `B`: batch size
- `C, H, W`: image channels, height, width
- `T`: number of thermometer thresholds
- `E = C * H * W * T`: encoded input width
- `N = len(widths) - 1`: number of shared logic layers
- `K = num_classes`: number of classes
- `D = widths[0]`: learned class-code width for each layer
- `W1..WN = widths[1:]`: shared logic-layer widths

## Per-Class Shared Computation

For each class `c`, the model forms:

```text
z_c^1 = concat(encode(x), class_code[1, c])
```

and runs the shared logic stack with a fresh class code at every layer:

```text
h_c^1 = LogicLayer1(z_c^1)
h_c^2 = LogicLayer2(concat(h_c^1, class_code[2, c]))
...
h_c^N = LogicLayerN(concat(h_c^(N-1), class_code[N, c]))
logit_c = GroupSum(h_c^N)
```

Stacking all classes gives:

```text
logits(x) = [
  f_shared(encode(x), class_codes[:, 0, :]),
  f_shared(encode(x), class_codes[:, 1, :]),
  ...,
  f_shared(encode(x), class_codes[:, K-1, :])
]
```

## Inside One `InputWiseLogicLayer`

Each output feature chooses two input features and evaluates a learned soft logic table:

```text
g(p, q) =
  (1 - p)(1 - q) w00 +
  (1 - p) q       w01 +
  p (1 - q)       w10 +
  p q             w11
```

The weights `w00..w11` come from applying the selected estimator (`sinusoidal` or `sigmoid`) to learned logits.

## Width Interpretation

For a config like:

```text
widths = (256, 16000, 16000, 16000)
```

the forward structure is:

```text
encoded input [B, E]
  + layer-1 class codes [K, 256]
  -> [B*K, E+256]
  -> [B*K, 16000]
  + layer-2 class codes [K, 256]
  -> [B*K, 16000+256]
  -> [B*K, 16000]
  + layer-3 class codes [K, 256]
  -> [B*K, 16000+256]
  -> [B*K, 16000]
  -> [B*K, 1]
  -> [B, K]
```
