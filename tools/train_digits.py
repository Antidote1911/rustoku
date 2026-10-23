#!/usr/bin/env python3
"""Trains the digit classifier of `rustoku photo` (needs numpy only).

    python3 tools/train_digits.py digits.dat src/vision/digits.bin

`digits.dat` comes from `examples/gen_digits.rs`. The model is a small
multilayer perceptron: normalized 24x24 image + number of holes (one-hot)
-> 128 ReLU units -> 9 digits. 10 % of the fonts are kept apart to measure
the accuracy on fonts never seen during training.

Output format (little endian): b"RTKM", inputs, hidden, outputs (u32),
then W1 (inputs x hidden), b1, W2 (hidden x outputs), b2 as f32.
"""
import struct
import sys

import numpy as np

CANVAS = 24
HIDDEN = 128
EPOCHS = 20
BATCH = 128


def load(path):
    raw = np.fromfile(path, np.uint8).reshape(-1, 4 + CANVAS * CANVAS)
    labels = raw[:, 0].astype(np.int64) - 1
    holes = np.minimum(raw[:, 1], 2)
    fonts = raw[:, 2].astype(np.int64) | (raw[:, 3].astype(np.int64) << 8)
    x = np.concatenate([raw[:, 4:] / 255.0, np.eye(3)[holes]], axis=1).astype(np.float32)
    return x, labels, fonts


def forward(params, x):
    w1, b1, w2, b2 = params
    h = np.maximum(x @ w1 + b1, 0)
    z = h @ w2 + b2
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return h, p / p.sum(axis=1, keepdims=True)


def main():
    data, out = sys.argv[1], sys.argv[2]
    x, y, fonts = load(data)
    rng = np.random.default_rng(1)
    held_out = rng.random(fonts.max() + 1) < 0.1
    test = held_out[fonts]
    xtr, ytr, xte, yte = x[~test], y[~test], x[test], y[test]
    print(f"{len(xtr)} training samples, {len(xte)} test samples ({held_out.sum()} unseen fonts)")

    n_in = x.shape[1]
    params = [
        rng.normal(0, np.sqrt(2 / n_in), (n_in, HIDDEN)).astype(np.float32),
        np.zeros(HIDDEN, np.float32),
        rng.normal(0, np.sqrt(1 / HIDDEN), (HIDDEN, 9)).astype(np.float32),
        np.zeros(9, np.float32),
    ]
    # Adam
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step, lr, decay = 0, 2e-3, 1e-4
    for epoch in range(EPOCHS):
        order = rng.permutation(len(xtr))
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            xb, yb = xtr[idx], ytr[idx]
            h, p = forward(params, xb)
            dz = p.copy()
            dz[np.arange(len(yb)), yb] -= 1
            dz /= len(yb)
            dh = (dz @ params[2].T) * (h > 0)
            grads = [xb.T @ dh + decay * params[0], dh.sum(0), h.T @ dz + decay * params[2], dz.sum(0)]
            step += 1
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mh, vh = m[i] / (1 - 0.9**step), v[i] / (1 - 0.999**step)
                params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
        lr *= 0.9
        if epoch % 5 == 4 or epoch == EPOCHS - 1:
            acc = lambda xs, ys: (forward(params, xs)[1].argmax(1) == ys).mean()
            print(f"epoch {epoch + 1}: train {acc(xtr, ytr):.4f}  unseen fonts {acc(xte, yte):.4f}")

    with open(out, "wb") as f:
        f.write(b"RTKM" + struct.pack("<3I", n_in, HIDDEN, 9))
        for p in params:
            f.write(p.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
