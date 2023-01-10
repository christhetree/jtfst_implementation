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

3. Compute the joint time-frequency scattering transform on all audio files in the dataset.

    a) Matlab -- the original paper uses [ScatNet](https://www.di.ens.fr/data/software/scatnet/) and Matlab. The matlab executable can be called from the command line (requires an active matlab installation and the full path of the executable to be aliases to 'matlab'):

    ```
    cd matlab
    matlab -r dJTFS_acciacatura --nodisplay --nodesktop
    matlab -r dJTFS_portamento --nodisplay --nodesktop
    matlab -r dJTFS_glissando --nodisplay --nodesktop
    ```
