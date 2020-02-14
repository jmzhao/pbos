#!/usr/bin/env python3

from dag import DAG


if __name__ == "__main__":
    dag_segger = DAG(n_largest=5)
    dag_segger.load_data("../datasets/20k.txt")
    dag_segger.test(["pineappleanapplepie"])
