# Setup

- Clone compact_reconstruction
  from `https://github.com/losyer/compact_reconstruction/tree/a55627c99a7b17d556cc96275a4f41b6b93f8782` into the
  folder `compact_reconstruction/`.

  ```sh
  git submodule update --init
  ```


- Install dependencies

  ```sh
  pip install -r requirements.txt
  ```

# Usage

- To reproduce the word similarity results:

  ```sh
  python ws_exp_pbos.py
  python ws_exp_sasaki.py
  ```

  The results will be available at `results/ws/{target_vector_name}_{model_type}/result.txt`, where
  `target_vector_name` is in [`google`, `polyglot`], and `model_type` is in [`bos`, `pbos`, `sasaki`]


- To reproduce the multilingual word similarity results:

  ```sh
  python ws_multilingual_exp_pbos.py
  python ws_multilingual_exp_sasaki.py
  ```

  The results will be available at `results/ws_multi/{lang}_{model_type}/result.txt`, where
  `target_vector_name` is in [`de`, `en`, `it`, `ru`], and `model_type` is in [`bos`, `pbos`, `sasaki`]


- To reproduce the POS tagging results:

  ```sh
  python pos_exp.py
  python pos_exp_sasaki.py
  ```

  The results will be available in the `results/pos` folder.

  You can print out all the results with `python pos_gather_results.py`
  
  