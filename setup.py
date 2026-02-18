from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='FlagEmbedding',
    version='1.3.5',
    description='FlagEmbedding: Dense Passage Retrieval and Reranking',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.44.2',
        'datasets>=2.19.0',
        'accelerate>=0.20.1',
        'sentence_transformers',
        'peft',
        'ir-datasets',
        'sentencepiece',
        'protobuf',
        'numpy',
        'pandas',
        'tqdm',
        'huggingface_hub',
        'regex',
        'packaging'
    ],
    extras_require={
        'finetune': ['deepspeed', 'flash-attn'],
        'eval': ['pytrec_eval', 'faiss-cpu'],
    },
)
