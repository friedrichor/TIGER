# Tiger
Tiger: A Unified Generative Model Framework on Multimodal Dialogue Response Generation

# Getting Start

## Installation

### 1. Prepare the code and the environment
```
git clone https://github.com/friedrichor/Tiger.git
cd Tiger/
conda env create -f environment.yml
conda activate tiger
```

### 2. Prepare the model weights

Please download our model weights from [here](https://drive.google.com/drive/folders/1ulc4X0yzJHQNFZJ2nyH5H9ZatZPkzZTC?usp=sharing). The final weights would be in a single folder in a structure similar to the following:

```
Tiger
├── model_weights
│   ├── tiger_t5_base_encoder.pth
│   └── tiger_dialogpt_medium.pth
```

For Text-to-Image Translator's weights, we have already uploaded it to Hugging Face, so you don't need to download it locally. More details can be sourced from [friedrichor/stable-diffusion-2-1-realistic](https://huggingface.co/friedrichor/stable-diffusion-2-1-realistic).

## Launching Demo Locally

```
python app.py --gpu_id 0
```


