# Retroformer
This is the directory of the work [Retroformer: Pushing the Limits of Interpretable End-to-end Retrosynthesis Transformer](https://arxiv.org/abs/2201.12475).



## Dependency:
Follow the below steps for dependency installation:
```
conda create -y -n retroformer tqdm
conda activate retroformer
conda install pytorch=1.10.1 torchvision cudatoolkit=11.0 -c pytorch
conda install -y rdkit -c conda-forge
```
Simply run ```pip install -e .``` for other dependencies. 

The `rdchiral` package is taken from [here](https://github.com/connorcoley/rdchiral) (no need to install it).


## Directory overview:
The overview of the full directory looks like this:
```
Retroformer/
└── retroformer/
    ├── rdchiral/
    ├── models/
    ├── utils/
    ├── dataset.py
    ├── train.py
    └── translate.py
└── data/
    ├── uspto50k/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── .../
└── intermediate/
    └── vocab_share.pk
└── checkpoint_untyped/
    └── model_best.pt
└── checkpoint_typed/
    └── model_best.pt   
├── result/
├── setup.py
├── start.sh
└── translate.sh
```

## Data:
Download the raw reaction dataset from [here](https://github.com/Hanjun-Dai/GLN) and put it into your data directory. One can also create your own reaction dataset as long as the data shares the same format (columns: `id`, `class`, `reactants>reagents>production`) and the reactions are atom-mapped. 

One can also download the processed USPTO50K data `cooked_*.lmdb` from [GoogleDrive](https://drive.google.com/drive/folders/1kiar6EhTInHBJpZLhPbrQ6dMcUuTfN39?usp=sharing) and put it into the corresponding data directory. If not, raw data will be processed and stored at the first time running the algorithm. The same applies to the built vocab file `vocab.pk`.

## Train:
One can specify different model and training configurations in `start.sh`. Below is a sample code that calls `train.py`. Simply run `./start.sh` for training.


```
python train.py \
  --num_layers 8 \
  --heads 8 \
  --max_step 100000 \
  --batch_size_token 4096 \
  --save_per_step 2500 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --device cuda \
  --known_class False \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <intermediate_folder> \
  --checkpoint_dir <checkpoint_folder> \
  --checkpoint <previous_checkpoint> 
```

## Test:
One can specify different translate configurations in `translate.sh` as the sample code below. Simply sun `./translate.sh` for inference. 

To replicate our results, download the pre-trained checkpoints from [GoogleDrive](https://drive.google.com/drive/folders/1kiar6EhTInHBJpZLhPbrQ6dMcUuTfN39?usp=sharing).

Special arguments:
- `stepwise`: determines whether to use _naive_ strategy or _search_ strategy.
- `use_template`: determines whether to use pre-computed reaction center to accelarate the _search_ strategy (corresponds to the reaction center retrieval setting in paper).  

```
python translate.py \
  --batch_size_val 8 \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <intermediate_folder> \
  --checkpoint_dir <checkpoint_folder> \
  --checkpoint <target_checkpoint> \
  --known_class False \
  --beam_size 10 \
  --stepwise False \
  --use_template False
```
