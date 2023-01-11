# jtfst_implementation

## Reproducing
1. Download & extract the [Chinese Bamboo Flute](https://zenodo.org/record/5744336#.Y5FBb-zP1pQ) dataset

    ```
    ssh scripts/download.sh
    ```

2. Filter the dataset to remove non pitch evolution-based playing techniques. We are only looking at acciacatura, glissando, and portamento. In order to run on consumer compute we also segment all audio files and annotations to less thatn 60s.

    ```
    python python/dataset_preprocess.py
    ```

    Saves filtered and segmented dataset to `CBFdataset_PETS`. Additionally, a text file called `file_names.txt` will be saved and used in subsequent steps to keep track of the order that audio files are processed.

3. Compute the joint time-frequency scattering transform on all audio files in the dataset.

    a) Matlab -- the original paper uses [ScatNet](https://www.di.ens.fr/data/software/scatnet/) and Matlab. The matlab executable can be called from the command line (requires an active matlab installation and the full path of the executable to be aliases to 'matlab'):

    ```
    cd matlab

    matlab -r dJTFS_acciacatura --nodisplay --nodesktop
    matlab -r dJTFS_portamento --nodisplay --nodesktop
    matlab -r dJTFS_glissando --nodisplay --nodesktop

    cd ..
    ```

    This will output `.mat` files for each technique. Defaults to outputting in the matlab directory with output like: `dJTFS_<technique>.mat`

4. Preprocess the extracted features prior to classification.

    ```
    python python/feature_preprocess.py matlab/dJTFS_acciacatura.mat file_names.txt acciacatura
    python python/feature_preprocess.py matlab/dJTFS_portamento.mat file_names.txt portamento
    python python/feature_preprocess.py matlab/dJTFS_glissando.mat file_names.txt glissando
    ```

    All processed features will be saved in `.npz` files in a directory called `features`.

5. Run binary classification 

    ```
    python python/svm_classifier.py features/acciacatura.npz
    python python/svm_classifier.py features/portamento.npz
    python python/svm_classifier.py features/glissando.npz
    ```

    Optionally run using a GPU by adding the flag: `--gpu 0` where 0 should be replaced with the desired gpu id.

    All results are stored in `.npz` files in a directory called `results`.
