

### **Cogni Map: Real-Time Detection of Cognitive Actions in Language Models Through Linear Probing**

**Ivan Chulo**
Harvard University, Cambridge, MA
ichulo@g.harvard.edu

***

### Abstract

Cogni Map is a novel tool for exploring and annotating cognitive actions in real-time as large language models generate text. Moving beyond prior work on user profiling, this research focuses on tracking the cognitive processes within the model-generated content itself, identifying actions such as *analyzing*, *reconsidering*, *divergent\_thinking*, and *self\_questioning*. By training linear probes on the internal activations from 30 layers of Gemma-3-4B, an average AUC-ROC of 0.78 was achieved across 45 distinct cognitive actions. The toolkit enables both quantitative and qualitative analysis through trained binary one-vs-rest probes and an interactive terminal user interface (TUI). This work, trained on a synthetic dataset of 31,500 examples, bridges the fields of mechanistic interpretability and cognitive science, offering a practical tool for understanding AI reasoning.

### Introduction

Understanding the internal reasoning of large language models (LLMs) is crucial for ensuring their safety, interpretability, and alignment. While previous studies have focused on profiling user attributes from conversations, less attention has been given to identifying the *cognitive actions* exhibited by the model during text generation.

This project introduces Cogni Map, a mechanistic interpretability tool designed to explore and annotate 45 cognitive actions spanning metacognitive, analytical, creative, and emotional categories. These actions, inspired by taxonomies from cognitive psychology, offer a detailed vocabulary for describing an AI's "thought processes," from *pattern\_recognition* and *hypothesis\_generation* to *emotional\_reappraisal*. Cogni Map supports both quantitative analysis of cognitive patterns and qualitative exploration via an interactive TUI. The methodology is built upon linear probing techniques to extract representations of cognitive actions from transformer activations, enabling researchers to observe the cognitive functions that are active during generation.

The main contributions of this work are:
1.  A synthetic dataset of 31,500 examples, with 700 examples for each of the 45 cognitive actions.
2.  A set of binary probes trained across 30 layers of Gemma-3-4B, achieving a 0.78 average AUC-ROC and identifying distinct layer specialization.
3.  A toolkit for both quantitative and qualitative analysis, including probe inference and an interactive TUI.
4.  A practical application demonstrated through the analysis of therapy transcripts, showcasing the tool's utility for downstream tasks.

### Methodology

**Cognitive Action Taxonomy**
A taxonomy of 45 cognitive actions was defined, organized into four categories:
*   **Metacognitive (13 actions):** e.g., *reconsidering*, *updating\_beliefs*, *meta\_awareness*.
*   **Analytical (12 actions):** e.g., *analyzing*, *evaluating*, *abstracting*.
*   **Creative (6 actions):** e.g., *divergent\_thinking*, *reframing*, *analogical\_thinking*.
*   **Emotional (14 actions):** e.g., *emotional\_reappraisal*, *emotion\_perception*.

**Activation Capture and Probing**
Following established probing methodologies, activations were extracted from 30 of the 35 layers of Gemma-3-4B using `nnsight`. A key technique employed was **augmented prompting**, which primes the model to encode the relevant cognitive information in the final token representation, creating a consistent extraction point.

A **one-vs-rest** strategy was used to train 45 independent binary linear probes, which allows for per-action interpretability and the flexibility to mix optimal layers for each action during inference. Training was performed with an AdamW optimizer, cosine annealing scheduler, and early stopping, using an AUC-ROC metric to handle the severe class imbalance.

**Data Generation**
The training dataset consists of 31,500 synthetic examples generated using GPT-3.5. This includes 700 examples for each of the 45 cognitive actions and 1,800 examples for sentiment analysis. The data's quality was validated with GPT-4, which showed 88-95% consistency.

### Results

**Probe Performance**
The binary probes demonstrated strong performance, achieving an average AUC-ROC of 0.78 and an average F1 score of 0.68 across all 45 cognitive actions. Top-performing probes included *suspending\_judgment* (0.988 AUC) and *counterfactual\_reasoning* (0.984 AUC), while more challenging actions included *emotion\_responding* (0.778 AUC). These results confirm that cognitive actions have linearly separable representations within Gemma-3-4B's activation space.

**Layer Specialization**
A distinct pattern of layer specialization was observed across the 30 analyzed layers. Layer 9 yielded the best average performance (AUC-ROC: 0.9481), with a strong performance envelope across layers 5-24. Performance degraded in the earliest and latest layers, suggesting that early layers focus on surface-level features, mid-layers capture high-level cognitive abstractions, and late layers optimize for next-token prediction, potentially overwriting these representations. Notably, different cognitive actions peaked at different layers—for instance, *divergent\_thinking* was best detected at layer 22, whereas *pattern\_recognition* peaked at layer 9.

**Application: Therapy Transcript Analysis**
Cogni Map was applied to analyze a therapy session transcript, comparing the cognitive actions of the therapist and the client. The analysis revealed that therapist-dominant actions included *perspective\_taking*, *accepting*, and *noticing*. In contrast, client-dominant actions were *reconsidering*, *emotion\_receiving*, and *self\_questioning*. These findings align with the principles of person-centered therapy, where the therapist fosters an environment for the client's self-exploration.

### Discussion and Future Work

Cogni Map builds upon prior work in interpretability but shifts the focus from modeling external users to tracking the internal cognitive processes of the model itself.

**Limitations** of this study include the reliance on synthetic data, the focus on a single model (Gemma-3-4B), the assumption of independence between cognitive actions, and potential artifacts from augmented prompting.

**Future directions** include extending the tool to larger models, training on human-annotated data, and exploring applications in AI safety, such as detecting deceptive reasoning, and in education for personalized cognitive feedback.

**Broader Impact**
This tool offers a means to explore AI reasoning patterns, which could help identify flawed logic, understand student thought processes in educational settings, and build more interpretable AI systems. The combination of quantitative probes and a qualitative TUI supports a wide range of use cases.

### Conclusion

Cogni Map is a practical tool for exploring and annotating 45 cognitive actions in language models. By using linear probes on the internal activations of Gemma-3-4B, this work successfully identified and analyzed these actions, revealing specialized layer-dependent representations. The toolkit, which supports both quantitative and qualitative analysis, has demonstrated its utility in a real-world application, bridging the gap between mechanistic interpretability and cognitive science. The full source code, trained probes, and datasets are available on GitHub.