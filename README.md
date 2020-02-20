# Usage

- Download datasets

```shell script
cd datasets && make
```

- Run n-shortest-path segmentation

```shell script
python n_shortest_path.py
```

- To replicate BoS results
```
python pbos_demo.py --boundary --sub_min_len 3 --model_path ./results/pbos/demo/model.bos --mock_bos
```

- To replicate PBoS results
```
python pbos_demo.py --boundary --sub_min_len 3 --model_path ./results/pbos/demo/model.pbos
```
