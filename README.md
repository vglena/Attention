# 🧠 BERT Masked Language Model & Attention Visualization

This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to predict masked words and visualize **attention heads** — showing how BERT pays attention to different words when understanding language.  

It demonstrates how transformer-based models interpret context and relationships between words.

---

## 📂 File Structure

```text
bert-mask-attention/
│
├── mask.py              # Main script for masked word prediction and visualization
├── analysis.md          # Your analysis of attention patterns across layers and heads
├── requirements.txt     # Project dependencies
├── LICENSE              # MIT License
└── README.md            # Documentation (this file)
```
## 🧩 Understanding the Project

The program performs the following steps:
1. Takes user input — a sentence containing a `[MASK]` token.
2. Tokenizes the input using a BERT tokenizer.
3. Predicts the masked token using `TFBertForMaskedLM`.
4. Displays top predictions for the masked token.
5. Visualizes attention across all BERT layers and heads, generating attention heatmaps.
These visualizations provide insights into how BERT focuses attention between words in different contexts.

## ⚙️ Specifications
Functions to Implement
1. `get_mask_token_index(mask_token_id, inputs)`
  - Returns the index (0-based) of the `[MASK]` token in the input.
  - Returns `None` if the mask is not present.
2. `get_color_for_attention_score(score)`
  - Maps an attention score (0 → 1) to a grayscale RGB tuple `(R, G, B)`.
  - Example: `0.25 → (63, 63, 63)`.
3. `visualize_attentions(tokens, attentions)`
  - Generates attention diagrams for every layer and head.
  - Uses `generate_diagram(layer_number, head_number, tokens, matrix)` for visualization.
  - Layers and heads are 1-indexed in the visual output.

## 🧪 Example Usage
```python
$ python mask.py
Enter text: The scientist looked at the [MASK] carefully.
```
Output:
```python
Top Predictions:
1. experiment
2. data
3. results
4. sample
5. specimen
```
Then, multiple attention diagrams will be generated showing how words attend to one another across all BERT layers.

## 📊 Example Visualization
Attention heatmaps are produced for each attention head and layer, such as:
```python
Layer 3, Head 10 → words attend to the next token.
Layer 4, Head 11 → adverbs attend to verbs they modify.
```
You will identify two or more additional attention heads and explain their behaviors in `analysis.md`.

## 📚 Requirements

List of dependencies to install before running:
```bash
pip install -r requirements.txt
```
**requirements.txt**
```text
tensorflow
transformers
matplotlib
numpy
```
💡 Tip: You may also need nltk if tokenization or preprocessing is extended.

## 🧠 Example BERT Concepts
`[CLS]` → Marks the beginning of a sentence
`[SEP]` → Marks the end of a sentence
`[MASK]` → Placeholder for the word to be predicted

Each attention head focuses on different relationships between tokens — e.g.,
verbs and their objects, prepositions, pronouns, or determiners.

## 🧩 Analysis Task
Complete `analysis.md` by describing two unique attention heads and the relationships they appear to represent.
Example structure:
```text
### Layer 6, Head 5
Appears to track subject-verb agreement across clauses.
Examples:
- "The boy who plays soccer [MASK] fast."
- "The teacher that reads books [MASK] kind."
```
## 🧰 Tools Used

BERT (base-uncased) from Hugging Face

TensorFlow for model execution

Matplotlib for attention visualization

## 🪪 License

This project is licensed under the MIT License.
