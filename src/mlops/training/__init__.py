"""Training layer: fit a model, evaluate it against a quality gate, and manage
it in the MLflow Model Registry (register, promote, rollback, load).

Public entry points:
* ``mlops.training.train.train_model`` - run one training job end-to-end
* ``mlops.training.evaluate.evaluate`` - metrics + quality-gate decision
* ``mlops.training.registry`` - thin wrapper over the MLflow registry
"""
