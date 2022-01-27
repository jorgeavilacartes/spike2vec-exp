PARAMETERS = {
    "RANDOM_STATE": 42, # random state or seed for all experiments
    "PATH_DATA": "data",
    "PATH_MODELS": "models",
    "PATH_RESULTS": "results",
    "PATH_METADATA": "data/GISAID-2022-01-24/metadata_tsv_2022_01_24/metadata.tsv",
    "PATH_FASTA": "data/GISAID-2022-01-24/spikeprot0122/spikeprot0122.fasta",
    "HOST": "Human", 
    "PANGO_LINEAGES": [
                    "B.1.1.7",
                    "B.1.617.2",
                    "AY.4",
                    "B.1.2",
                    "B.1",
                    "B.1.177",
                    "P.1",
                    "B.1.1",
                    "B.1.429",
                    "AY.12",
                    "B.1.160",
                    "B.1.526",
                    "B.1.1.519",
                    "B.1.351",
                    "B.1.1.214",
                    "B.1.427",
                    "B.1.221",
                    "B.1.258",
                    "B.1.177.21",
                    "D.2",
                    "B.1.243",
                    "R.1"
                    ],
    "TRAIN_SIZE": 0.8,
    "RFF_SIZE": 500, # random fourier features dimensions
    "DOC2VEC_SIZE": 50, # doc2vec dimension
    "DOC2VEC_EPOCHS": 40, # doc2vec dimension
}