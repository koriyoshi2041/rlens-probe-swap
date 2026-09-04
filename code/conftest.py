import pytest
import torch


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    from rlens.paths import MODEL_DIR

    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


@pytest.fixture(scope="session")
def model():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from rlens.model import load_model

    return load_model()


@pytest.fixture(scope="session")
def lenses():
    from rlens.model import load_lenses

    return load_lenses()
