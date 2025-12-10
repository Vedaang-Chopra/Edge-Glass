# Edge Glass System Architecture

The following diagram illustrates the two-stage training pipeline for the Edge Glass VLM.

```mermaid
graph LR
    %% Global Styles
    classDef frozen fill:#f5f5f5,stroke:#999,stroke-width:2px,stroke-dasharray: 5 5
    classDef trainable fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef input fill:#ffffff,stroke:#333,stroke-width:1px
    classDef module fill:#fff8e1,stroke:#f57f17,stroke-width:2px

    %% Stage 1: Alignment (Pre-training)
    subgraph S1 [Stage 1: Vision-Language Alignment]
        direction TB
        I1(Image):::input --> VE[Vision Encoder<br/>CLIP ViT-L/14]:::trainable
        VE --> |Patch Tokens| AP[Attn Pooling]:::trainable
        AP --> |4096 dim| LP[Projection]:::trainable
        LP --> MRL[MRL Loss]:::module
        T1(Text):::input --> TE[Text Encoder]:::trainable
        TE --> MRL
    end

    %% Stage 2: VLM Tuning (Instruction Tuning)
    subgraph S2 [Stage 2: VLM Instruction Tuning]
        direction TB
        I2(Image):::input --> VE_F[Vision Encoder<br/>Frozen S1 Weights]:::frozen
        VE_F --> |577 Tokens| VP[Adapter<br/>4096 to 1536]:::trainable
        
        Q(Question):::input --> Tok[Tokenizer]
        Tok --> |Text Emb| LLM_In
        VP --> |Visual Emb| LLM_In[Concat]
        
        LLM_In --> LLM[Qwen2.5-1.5B<br/>+ LoRA]:::trainable
        LLM --> A(Answer):::input
    end

    %% Cross-stage link
    S1 -.-> |Transfer Weights| S2
```

## Legend

| Visual Style | Component Status | Description |
| :--- | :--- | :--- |
| **Blue / Solid Border** | **Trainable** | Parameters are updated during this training stage. |
| **Grey / Dashed Border** | **Frozen** | Parameters are static (loaded from previous stage or base model). |
| **Orange** | **Loss Module** | Used for optimization objectives (e.g., Contrastive/MRL). |
| **White** | **Input / Output** | Data elements flowing through the system. |


