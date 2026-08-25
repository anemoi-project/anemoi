# Draft Attention

Anemoi's current Draft Attention backend has three pieces:

1. Reshape visual Q/K into the latent video grid and apply 2D average pooling.
2. Build a draft map with blockwise QK, online softmax statistics, and global top-k.
3. Reorder Q/K/V into spatial blocks and run Triton block-sparse attention.

The draft-map builder does not compute `Q @ K.T` as one giant matrix. It uses
blockwise QK tiles and online softmax row statistics, then makes a headwise
top-k block mask from the resulting softmax probabilities.

Only visual tokens enter draft-map guidance. Visual/text and text/text
interactions remain dense. The output is restored to the model's original token
order after sparse attention.

