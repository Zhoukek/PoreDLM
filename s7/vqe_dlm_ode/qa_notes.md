# QA notes

- 156 expected token records and embedding rows are present in identical order.
- Embedding matrix shape is 156 x 768; all values are finite float32 values.
- Every input sequence follows BOS + (64K VQE token + 128) + EOS; padding is masked.
- DLM inference used the README parameters: 2 ODE steps, start time 0.95, and self-conditioning CFG scale 0.0.
- Pooling uses only `ode_hidden_state` signal-token positions and excludes BOS, EOS, and padding.
- Ten primary aggregate rows are present; all groups contain both positive- and negative-strand reads.
- No individual read is treated as a primary classification sample.
- Uncertainty uses 1,000 strand-stratified read bootstraps; significance uses all 32 site-label assignments.
- Figure outputs use an opaque white background; SVG text remains editable and PDF text uses TrueType embedding.
- The PNG was visually inspected at original resolution. The exact-test annotation was moved into unused panel space to avoid collision with the model metric point.
