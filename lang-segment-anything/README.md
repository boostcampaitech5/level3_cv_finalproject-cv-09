# Language Segment-Anything

Language Segment-Anything is an open-source project that combines the power of instance segmentation and text prompts to generate masks for specific objects in images. Built on the recently released Meta model, segment-anything, and the GroundingDINO detection model, it's an easy-to-use and effective tool for object detection and image segmentation.

![person.png](/assets/outputs/person.png)

## Features

- Zero-shot text-to-bbox approach for object detection.
- GroundingDINO detection model integration.
- Easy deployment using the Lightning AI app platform.
- Customizable text prompts for precise object segmentation.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- torch (tested 2.0)
- torchvision
- requirements.txt

### Installation
https://developer.nvidia.com/cuda-toolkit-archive  
nvidia-smi에 맞는 cuda-toolkit 다운로드  
Driver Version: 450.80.02   CUDA Version: 11.0  
이런식으로 나오는데 Driver Version이랑 젤 비슷한거로 깔면 됩니다.
```
# cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
sh cuda_11.0.3_450.51.06_linux.run

# cuda-toolkit 디렉토리 경로 확인
ls -lh /usr/local | grep cuda

# 환경변수 설정 (위에서 확인한 경로로 변경해서 지정해주세요)
export PATH=$PATH:/usr/local/cuda-11.0/bin
export CUDA_PATH=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
export CUDADIR=/usr/local/cuda-11.0
```

### GPU 버전 GrooundingDINO 설치
Grounded_SAM 클론 후에 GroundingDINO 설치해주시면 됩니다.  
이후 Grounded-Segment-Anything 디렉토리는 삭제해도 무방
```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e GroundingDINO
```

### Usage

To run the Lightning AI APP:

`lightning run app app.py`

Use as a library:

```python
from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
```

## Examples

![car.png](/assets/outputs/car.png)

![kiwi.png](/assets/outputs/kiwi.png)

![person.png](/assets/outputs/person.png)

## Roadmap

Future goals for this project include:

1. **FastAPI integration**: To streamline deployment even further, we plan to add FastAPI code to our project, making it easier for users to deploy and interact with the model.

1. **Labeling pipeline**: We want to create a labeling pipeline that allows users to input both the text prompt and the image and receive labeled instance segmentation outputs. This would help users efficiently generate results for further analysis and training.

1. **Implement CLIP version**: To (maybe) enhance the model's capabilities and performance, we will explore the integration of OpenAI's CLIP model. This could provide improved language understanding and potentially yield better instance segmentation results.

## Acknowledgments

This project is based on the following repositories:

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)

## License

This project is licensed under the Apache 2.0 License
