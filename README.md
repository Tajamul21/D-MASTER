
<div align="center">
<figure class="center-figure"> <img src="media/logo.png" width="85%"></figure>
</div>

<h1 align="left">
    D-MASTER: Mask Annealed Transformer for Unsupervised Domain Adaptation in Breast Cancer Detection from Mammograms
</h1>

<div align="left">

[![](https://img.shields.io/badge/website-stark-purple?style=plastic&logo=Google%20chrome)](https://stark.stanford.edu/)
[![](https://img.shields.io/badge/Dataset-yellow?style=plastic&logo=Hugging%20face)](https://huggingface.co/datasets/snap-stanford/stark)
[![](https://img.shields.io/badge/SKB_Explorer-online-yellow?style=plastic&logo=Hugging%20face)](https://stark.stanford.edu/skb_explorer.html)
[![](https://img.shields.io/badge/Arxiv-paper-red?style=plastic&logo=arxiv)](https://arxiv.org/abs/2404.13207)
[![](https://img.shields.io/badge/pip-stark--qa-brightgreen?style=plastic&logo=Python)](https://pypi.org/project/stark-qa/) 
[![](https://img.shields.io/badge/doc-online-blue?style=plastic&logo=Read%20the%20Docs)](https://stark.stanford.edu/docs/index.html)
[![](https://img.shields.io/badge/-Linkedin-blue?style=plastic&logo=Linkedin)](https://www.linkedin.com/posts/leskovec_reduce-llm-hallucinations-with-rag-over-textual-activity-7190745116339302401-da4n?utm_source=share&utm_medium=member_desktop) 
[![](https://img.shields.io/badge/-Twitter-cyan?style=plastic&logo=X)](https://twitter.com/ShirleyYXWu/status/1784970920383402433) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>



## NEWS
- **[Jun 2024]** We make our benchmark as a pip package [stark-qa](https://pypi.org/project/stark-qa/). You can directly load the data from the package now!
- **[Jun 2024]** We migrate our data to [Hugging Face](https://huggingface.co/datasets/snap-stanford/stark)! You don't need to change anything, the data will be automatically downloaded.
- **[May 2024]** We have augmented our benchmark with three high-quality human-generated query datasets which are open to access. See more details in our [updated arxiv](https://arxiv.org/abs/2404.13207)! 
- **[May 9th 2024]** We release [STaRK SKB Explorer](https://stark.stanford.edu/skb_explorer.html), an interactive interface for you to explore our knowledge bases!
- **[May 7th 2024]** We present STaRK in the [2024 Stanford Annual Affiliates Meeting](https://forum.stanford.edu/events/2024-annual-affiliates-meeting/day-3-ai-health-and-data-science-applications-workshop) and [2024 Stanford Data Science Conference](https://datascience.stanford.edu/2024-stanford-data-science-conference).
- **[May 5th 2024]** STaRK was reported on [Marketpost](https://www.marktechpost.com/2024/05/01/researchers-from-stanford-and-amazon-developed-stark-a-large-scale-semi-structure-retrieval-ai-benchmark-on-textual-and-relational-knowledge-bases/) and [智源社区 BAAI](https://hub.baai.ac.cn/paper/6841fd6f-1eca-41c4-a432-5f2d845ac167). Thanks for writing about our work!
- **[Apr 21st 2024]** We release the STaRK benchmark.

## What is D-MASTER?
STaRK is a large-scale Semi-structured Retrieval Benchmark on Textual and Relational Knowledge bases, covering applications in product search, academic paper search, and biomedicine inquiries.

Featuring diverse, natural-sounding, and practical queries that require context-specific reasoning, STaRK sets a new standard for assessing real-world retrieval systems driven by LLMs and presents significant challenges for future research.


🔥 Check out our [website](https://stark.stanford.edu/) for more overview!
<!-- 
<figure class="center-figure">
    <img src="media/overview.jpg" width="90%">
</figure>



## Why STaRK?
- **Novel Task**: Recently, large language models have demonstrated significant potential on information retrieval tasks. Nevertheless, it remains an open
question how effectively LLMs can handle the complex interplay between textual and relational
requirements in queries.

- **Large-scale and Diverse KBs**: We provide three large-scale knowledge bases across three areas, which are constructed from public sources.

    <figure class="center-figure"> <img src="media/kb.jpg" width="90%"></figure> 

- **Natural-sounding and Practical Queries**: The queries in our benchmark are crafted to incorporate rich relational information and complex textual properties, and closely mirror questions in real-life scenarios, e.g., with flexible query formats and possibly with extra contexts.

    <figure class="center-figure"> <img src="media/questions.jpg" width="95%"></figure>  -->

## What is RSBA-BSD1K data?
STaRK is a large-scale Semi-structured Retrieval Benchmark on Textual and Relational Knowledge bases, covering applications in product search, academic paper search, and biomedicine inquiries.

Featuring diverse, natural-sounding, and practical queries that require context-specific reasoning, STaRK sets a new standard for assessing real-world retrieval systems driven by LLMs and presents significant challenges for future research.


🔥 Check out our [website](https://stark.stanford.edu/) for more overview!

# Access benchmark data

## 1) Env Setup

### From pip (recommended)
With python >=3.8 and <3.12
```bash
pip install stark-qa
```

### From source
Create a conda env with python >=3.8 and <3.12 and install required packages in `requirements.txt`.
```bash
conda create -n stark python=3.11
conda activate stark
pip install -r requirements.txt
```

## 2) Data loading 

```python
from stark_qa import load_qa, load_skb

dataset_name = 'amazon'

# Load the retrieval dataset
qa_dataset = load_qa(dataset_name)
idx_split = qa_dataset.get_idx_split()

# Load the semi-structured knowledge base
skb = load_skb(dataset_name, download_processed=True, root=None)
```
The root argument for load_skb specifies the location to store SKB data. With default value `None`, the data will be stored in [huggingface cache](https://huggingface.co/docs/datasets/en/cache).


### Data of the Retrieval Task

Question answer pairs for the retrieval task will be automatically downloaded in `data/{dataset}/stark_qa` by default. We provided official split in `data/{dataset}/split`.


### Data of the Knowledge Bases

There are two ways to load the knowledge base data:
- (Recommended) Instant downloading: The knowledge base data of all three benchmark will be **automatically** downloaded and loaded when setting `download_processed=True`. 
- Process data from raw: We also provided all of our preprocessing code for transparency. Therefore, you can process the raw data from scratch via setting `download_processed=False`. In this case, STaRK-PrimeKG takes around 5 minutes to download and load the processed data. STaRK-Amazon and STaRK-MAG may takes around an hour to process from the raw data.



## 1. Installation

### 1.1 Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4

- Python >= 3.8

- torch >= 1.10.1, torchvision >= 0.11.2

- Other requirements

  ```bash
  pip install -r requirements.txt
  ```

### 1.2 Compiling Deformable DETR CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```



## 2. Usage

### 2.1 Data preparation

We provide the 2 benchmarks in our paper: 

- city2foggy: cityscapes dataset is used as source domain, and foggy_cityscapes(0.02) is used as target domain.
- sim2city: sim10k dataset is used as source domain, and cityscapes which only record AP of cars is used as target domain.
- city2bdd: cityscapes dataset is used as source domain, and bdd100k-daytime is used as target domain.

You can download the raw data from the official websites: [cityscapes](https://www.cityscapes-dataset.com/downloads/),  [foggy_cityscapes](https://www.cityscapes-dataset.com/downloads/),  [sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix), [bdd100k](https://bdd-data.berkeley.edu/). We provide the annotations that are converted into coco style, download from [here](https://drive.google.com/file/d/1LB0wK9kO3eW8jpR2ZtponmYWe9x2KSiU/view?usp=sharing) and organize the datasets and annotations as following:

```bash
[data_root]
└─ cityscapes
	└─ annotations
		└─ cityscapes_train_cocostyle.json
		└─ cityscapes_train_caronly_cocostyle.json
		└─ cityscapes_val_cocostyle.json
		└─ cityscapes_val_caronly_cocostyle.json
	└─ leftImg8bit
		└─ train
		└─ val
└─ foggy_cityscapes
	└─ annotations
		└─ foggy_cityscapes_train_cocostyle.json
		└─ foggy_cityscapes_val_cocostyle.json
	└─ leftImg8bit_foggy
		└─ train
		└─ val
└─ sim10k
	└─ annotations
		└─ sim10k_train_cocostyle.json
		└─ sim10k_val_cocostyle.json
	└─ JPEGImages
└─ bdd10k
	└─ annotations
		└─ bdd100k_daytime_train_cocostyle.json
		└─ bdd100k_daytime_val_cocostyle.json
	└─ JPEGImages
```

To use additional datasets, you can edit [datasets/coco_style_dataset.py](https://github.com/JeremyZhao1998/MRT-release/blob/main/datasets/coco_style_dataset.py) and add key-value pairs to `CocoStyleDataset.img_dirs` and `CocoStyleDataset.anno_files` .

### 2.2 Training and evaluation

As has been discussed in implementation details in the paper, to save computation cost, our method is designed as a three-stage paradigm. We first perform `source_only` training which is trained standardly by labeled source domain. Then, we perform `cross_domain_mae` to train the model with MAE branch. Finally, we perform `teaching` which utilize a teacher-student framework with MAE branch and selective retraining.

For example, for `city2foggy` benchmark, first edit the files in `configs/def-detr-base/city2foggy/` to specify your own `DATA_ROOT` and `OUTPUT_DIR`, then run:

```bash
sh configs/def-detr-base/city2foggy/source_only.sh
sh configs/def-detr-base/city2foggy/cross_domain_mae.sh
sh configs/def-detr-base/city2foggy/teaching.sh
```

We use `tensorboard` to record the loss and results. Run the following command to see the curves during training: 

```bash
tensorboard --logdir=<YOUR/LOG/DIR>
```

To evaluate the trained model and get the predicted results, run:

```bash
sh configs/def-detr-base/city2foggy/evaluation.sh
```



## 3. Results and Model Parameters

We conduct all experiments with batch size 8 (for source_only stage, 8 labeled samples; for cross_domain_mae and MRT teaching stage, 8 labeled samples and 8 unlabeled samples), on 2 NVIDIA A100 GPUs.

**inhouse2inbreast**: Inhouse → INBreast

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 29.5  | [logs](https://drive.google.com/file/d/1O-B-OXBf8clOSNMJLtJEPuNQvo5W2CuU/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1J6PpDsKvWvTJthwctFuYV8kUEnGGTVUk/view?usp=drive_link) |
| resnet50 | 6              | 6              | cross_domain_mae | 35.8  | [logs](https://drive.google.com/file/d/1gUYJDX9eE5FIKWMbR_tK6leMnM5q06dj/view?usp=sharing) & [weights](https://drive.google.com/file/d/1X-STx26799Q2vAUle1QjXj_1gzwvZrRk/view?usp=drive_link) |
| resnet50 | 6              | 6              | MRT teaching     | 51.2  | [logs](https://drive.google.com/file/d/1YwLUo3t2KJ1pjENFAr5vECZlrRFWwKG2/view?usp=sharing) & [weights](https://drive.google.com/file/d/1BooqcIdzP97I3ax7JN6ULZWoZcvRKLlm/view?usp=sharing) |

**inhouse2rsna**: Inhouse → RSNA-BSD1K

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 53.2  | [logs](https://drive.google.com/file/d/1qfdHLuUX8N3SRUTNmclf0Y3PJ-deOF4r/view?usp=sharing) & [weights](https://drive.google.com/file/d/1mkqKxrWannqJN1_tJdh76t7ZAGIzDsIs/view?usp=sharing) |
| resnet50 | 6              | 6              | cross_domain_mae | 57.1  | [logs](https://drive.google.com/file/d/1bDNux81HhHZhmuoABwU-N4ALZFjKQWHR/view?usp=drive_link) & [weights](https://drive.google.com/file/d/14cTFm8pM9DmN2UcV7NGaMJxOJVfOvANP/view?usp=sharing) |
| resnet50 | 6              | 6              | MRT teaching     | 62.0  | [logs](https://drive.google.com/file/d/1S_GiAb9Ujfndh6XHnBz6qmCawpEDY102/view?usp=sharing) & [weights](https://drive.google.com/file/d/1dsSuk24_jEq3k4DBpoPr4AH3mxL0DspP/view?usp=sharing) |

**ddsm2inhouse**: DDSM → Inhouse

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 29.6  | [logs](https://drive.google.com/file/d/1KIydqXkj0LIlDlHHDW4TfxIh3-rmaWQM/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1IAzbKozA_Rq-2H-KzdcvGp3LGJrZ4J5G/view?usp=drive_link) |
| resnet50 | 6              | 6              | cross_domain_mae | 31.1  | [logs](https://drive.google.com/file/d/1gUYJDX9eE5FIKWMbR_tK6leMnM5q06dj/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1X-STx26799Q2vAUle1QjXj_1gzwvZrRk/view?usp=drive_link) |
| resnet50 | 6              | 6              | MRT teaching     | 33.7  | [logs](https://drive.google.com/file/d/13jgRrsKVDap0O9rUiY-ZhZp-kL6di4EH/view?usp=sharing) & [weights](https://drive.google.com/file/d/1VRtNy_2bXdkpLr1h6v-ZusEuTR7hAu_v/view?usp=sharing) |

**ddsm2inhouse**: DDSM → Inhouse

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 29.6  | [logs](https://drive.google.com/file/d/1KIydqXkj0LIlDlHHDW4TfxIh3-rmaWQM/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1IAzbKozA_Rq-2H-KzdcvGp3LGJrZ4J5G/view?usp=drive_link) |
| resnet50 | 6              | 6              | cross_domain_mae | 31.1  | [logs](https://drive.google.com/file/d/1gUYJDX9eE5FIKWMbR_tK6leMnM5q06dj/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1X-STx26799Q2vAUle1QjXj_1gzwvZrRk/view?usp=drive_link) |
| resnet50 | 6              | 6              | MRT teaching     | 33.7  | [logs](https://drive.google.com/file/d/13jgRrsKVDap0O9rUiY-ZhZp-kL6di4EH/view?usp=sharing) & [weights](https://drive.google.com/file/d/1VRtNy_2bXdkpLr1h6v-ZusEuTR7hAu_v/view?usp=sharing) |


**city2foggy**: cityscapes → foggy cityscapes(0.02)

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 29.5  | [logs](https://drive.google.com/file/d/1O-B-OXBf8clOSNMJLtJEPuNQvo5W2CuU/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1J6PpDsKvWvTJthwctFuYV8kUEnGGTVUk/view?usp=drive_link) |
| resnet50 | 6              | 6              | cross_domain_mae | 35.8  | [logs](https://drive.google.com/file/d/1gUYJDX9eE5FIKWMbR_tK6leMnM5q06dj/view?usp=sharing) & [weights](https://drive.google.com/file/d/1X-STx26799Q2vAUle1QjXj_1gzwvZrRk/view?usp=drive_link) |
| resnet50 | 6              | 6              | MRT teaching     | 51.2  | [logs](https://drive.google.com/file/d/1YwLUo3t2KJ1pjENFAr5vECZlrRFWwKG2/view?usp=sharing) & [weights](https://drive.google.com/file/d/1BooqcIdzP97I3ax7JN6ULZWoZcvRKLlm/view?usp=sharing) |

**sim2city**: sim10k → cityscapes(car only)

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 53.2  | [logs](https://drive.google.com/file/d/1qfdHLuUX8N3SRUTNmclf0Y3PJ-deOF4r/view?usp=sharing) & [weights](https://drive.google.com/file/d/1mkqKxrWannqJN1_tJdh76t7ZAGIzDsIs/view?usp=sharing) |
| resnet50 | 6              | 6              | cross_domain_mae | 57.1  | [logs](https://drive.google.com/file/d/1bDNux81HhHZhmuoABwU-N4ALZFjKQWHR/view?usp=drive_link) & [weights](https://drive.google.com/file/d/14cTFm8pM9DmN2UcV7NGaMJxOJVfOvANP/view?usp=sharing) |
| resnet50 | 6              | 6              | MRT teaching     | 62.0  | [logs](https://drive.google.com/file/d/1S_GiAb9Ujfndh6XHnBz6qmCawpEDY102/view?usp=sharing) & [weights](https://drive.google.com/file/d/1dsSuk24_jEq3k4DBpoPr4AH3mxL0DspP/view?usp=sharing) |

**city2bdd**: cityscapes → bdd100k(daytime)

| backbone | encoder layers | decoder layers | training stage   | AP@50 | logs & weights                                               |
| -------- | -------------- | -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| resnet50 | 6              | 6              | source_only      | 29.6  | [logs](https://drive.google.com/file/d/1KIydqXkj0LIlDlHHDW4TfxIh3-rmaWQM/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1IAzbKozA_Rq-2H-KzdcvGp3LGJrZ4J5G/view?usp=drive_link) |
| resnet50 | 6              | 6              | cross_domain_mae | 31.1  | [logs](https://drive.google.com/file/d/1gUYJDX9eE5FIKWMbR_tK6leMnM5q06dj/view?usp=drive_link) & [weights](https://drive.google.com/file/d/1X-STx26799Q2vAUle1QjXj_1gzwvZrRk/view?usp=drive_link) |
| resnet50 | 6              | 6              | MRT teaching     | 33.7  | [logs](https://drive.google.com/file/d/13jgRrsKVDap0O9rUiY-ZhZp-kL6di4EH/view?usp=sharing) & [weights](https://drive.google.com/file/d/1VRtNy_2bXdkpLr1h6v-ZusEuTR7hAu_v/view?usp=sharing) |



## Reference 

Please consider citing our paper if you use our benchmark or code in your work:
```
@article{wu24stark,
    title        = {STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases},
    author       = {
        Shirley Wu and Shiyu Zhao and 
        Michihiro Yasunaga and Kexin Huang and 
        Kaidi Cao and Qian Huang and 
        Vassilis N. Ioannidis and Karthik Subbian and 
        James Zou and Jure Leskovec
    },
    eprinttype   = {arXiv},
    eprint       = {2404.13207},
  year           = {2024}
}
```
