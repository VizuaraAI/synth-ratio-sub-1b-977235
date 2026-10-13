"""Sub-1B synthetic data ratio study.

Modules:
    config      load and validate YAML run configs
    model       four fixed LLaMA-style decoder sizes (70M to 830M)
    synthesize  build the rephrase / textbook / QA synthetic pools
    data        tokenize, cache and interleave real + synthetic streams
    train       one pretraining run for one (size, alpha) pair
    evaluate    held-out real-web loss plus zero-shot downstream accuracy
    analyze     fit the loss-vs-alpha curve, extract alpha*, build the table
"""

__version__ = "0.1.0"
