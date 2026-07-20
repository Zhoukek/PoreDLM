# QA notes

- 156 expected windows, token records, and embedding rows are present in identical order.
- Embedding matrix shape is 156 x 768; all values are finite.
- Raw VQE tokens range from 22 to 65,501 and remain within the 65,536-entry codebook.
- Shifted signal-token IDs remain within the BERT vocabulary of 65,664 entries.
- No Apple normalization was reapplied.
- Ten primary aggregate rows are present; all groups contain both positive- and negative-strand reads.
- No individual read is used as a primary classification sample.
- The VQE-BERT figure was visually inspected; panels, legends, labels, and uncertainty intervals do not overlap incoherently.
