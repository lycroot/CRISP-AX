# References for the CSI-only baselines

## EHUNAM Data Descriptor

de Armas, E., Diaz, G., Sobron, I., et al. EHUNAM, a WiFi CSI-based dataset for human and machine
sensing. *Scientific Data* **12**, 1950 (2025). https://doi.org/10.1038/s41597-025-06238-4

The EHUNAM technical validation uses three 2D convolutional layers (32 `5 × 5`, 64 `3 × 3` and
128 `3 × 3` kernels), dense layers of 256 and 128 units, Mish activations and Adam with a learning
rate of 0.001. The `cnn2d` baseline adopts these convolution widths and kernels, Mish and Adam, and
differs as follows:

- the input is the amplitude of the two receive chains; uncalibrated phase is not used;
- global average pooling replaces a large flatten layer, keeping the parameter count small for the
  996-subcarrier input;
- the input covers the full activity window, resampled to 256 packets;
- evaluation uses a subject-stratified in-domain split and subject-wise LOSO.

## FeitCSI data format

Hutar, M., Brida, P. & Machaj, J. FeitCSI, the 802.11 CSI tool (2023). https://feitcsi.kuskosoft.com/

- Format: https://feitcsi.kuskosoft.com/csi_format/
- Python example: https://feitcsi.kuskosoft.com/python/

Each record has a fixed 272-byte header followed by the CSI payload. Bytes 0–3 hold the payload size,
bytes 46 and 47 the numbers of RX and TX chains, and bytes 52–55 the number of subcarriers. Each
complex CSI value takes 4 bytes (signed `int16` real and imaginary parts), so the payload size is
`4 × RX × TX × subcarriers`. `axhome_csi/feitcsi.py` implements a streaming little-endian parser for
this layout and additionally rejects truncated records, inconsistent sizes and shape changes between
packets.

## Subject- and environment-independent evaluation

Meneghello, F., Garlisi, D., Dal Fabbro, N., Tinnirello, I. & Rossi, M. SHARP: Environment and person
independent activity recognition with commodity IEEE 802.11 access points. *IEEE Transactions on
Mobile Computing* **22**, 6160–6175 (2023). https://doi.org/10.1109/TMC.2022.3185681

## Mish activation

Misra, D. Mish: A self regularized non-monotonic activation function. *BMVC* (2020).
https://doi.org/10.5244/C.34.191

The implementation uses `torch.nn.Mish`.

## Class-weighted cross-entropy

PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

Balanced class weights are computed as

```text
weight_c = N / (C × count_c)
```

where `N` is the number of training samples, `C` the number of classes and `count_c` the number of
training samples of class `c`. Validation and test samples are not used.
