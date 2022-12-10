# jtfst_implementation

## Reproducing
1. Download & extract the [Chinese Bamboo Flute](https://zenodo.org/record/5744336#.Y5FBb-zP1pQ) dataset

    ```
    ssh scripts/download.sh
    ```

2. Filter the dataset to remove non pitch evolution-based playing techniques. We are only looking at acciacatura, glissando, and portamento. In order to run on consumer compute we also segment all audio files and annotations to less thatn 60s.

    ```
    python python/data_preprocess.py
    ```

    Saves filtered and segmented dataset to `CBFdataset_PETS`
