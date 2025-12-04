## SVM ABSA package (`sentalyzer/absa/SVMUtil`) - Training & inference

```mermaid
flowchart LR
    subgraph Data
        D1[SentFin CSV -train/test-]
        D2[load_sentfin_absa_samples \n List-ABSASample-]
    end

    subgraph Samples
        S1[ABSASample -text, aspect, label-]
        C1[default_setfit_combiner \n-text, aspect- → combined text]
    end

    subgraph Train
        CFG[SVMABSAConfig]
        T1[train_svm_absa\n TF-IDF + LinearSVC]
        MDIR[models/svm_sentfin]
    end

    subgraph Inference
        M1[SVMABSAModel.from_dir \n -load vectorizer + classifier-]
        P1[predict_samples \n-List-ABSASample-- → List-str-]
    end

    subgraph Reporting
        R1[write_svm_eval_report]
        R2[write_eval_report \n-absa/utils/reporting-]
    end

    D1 --> D2 --> S1
    S1 --> C1

    S1 -->|train_svm_absa\n+ combiner| T1 --> MDIR
    MDIR --> M1 --> P1

    S1 -->|gold labels| R1
    P1 -->|preds| R1 --> R2
```


