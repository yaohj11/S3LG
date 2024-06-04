# S3LG for Text to Gloss Translation

Official implementation of the [**ACL 2024** paper] Semi-Supervisied Spoken Language Glossification.

## Setup
### Installation
- Install the environment
    ```shell
  pip install -r requirement.txt
  ```

### Data Preparation
The data used in our experiments is placed in ./dataset. The structure is as following:
- ./dataset/phoenix/phoenix2014T/text: the plain-text version of the corpora for PHOENIX2014T dataset. 
- ./dataset/phoenix/external_corpus.txt: the monolingual data.
- .dataset/phoenix/rule_pseudo.txt: the rule-based synthetic data. Run the following command to generate rule-based pseudo glosses for unlabeled data.
   ```bash
   cd ./dataset/phoenix
   python rule_based_preprocess.py
   ```

### Training
To train the proposed model, please run the following command.
```bash
   cd ./project/translation
   python G2T_train.py -t ./../../dataset/phoenix/phoenix2014T/ -ec ./../../dataset/phoenix/external_corpus.txt -ge 70 -g 0
```

## Citation
If you find this repo useful in your research works, please consider citing:
```bibtex
@inproceedings{zhou2021improving,
  title={Improving sign language translation with monolingual data by sign back-translation},
  author={Zhou, Hao and Zhou, Wengang and Qi, Weizhen and Pu, Junfu and Li, Houqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1316--1325},
  year={2021}
}
```

Please note that the corpora ([PHOENIX2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)) have their own licenses and any use of them should be conforming with them and include the appropriate citations.
