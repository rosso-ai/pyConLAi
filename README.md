# pyConLAi

## What's ConLAi?
Con(sensus)L(erning) Ai is a module for Ledger type federated learning.
Ledger type federated learning achieves federated learning in a way that feels like Git.

![features](https://github.com/rosso-ai/pyConLAi/blob/main/docs/images/conlai_features.png?raw=true)

## How to Install
from PyPi:  
```shell
pip install pyconlai
```

## How to Start
Here is how to run the CIFAR10 example:

This sample runs two client nodes in multi-process mode.  

```shell
cd examples/cifar10
python run.py conf/dsgd_cifar10.yml
```

For details, please see [CIFAR10 example README](https://github.com/rosso-ai/pyConLAi/tree/main/examples/cifar10).

### Server-side module
See also the server module README for more information.  
https://github.com/rosso-ai/conlai

## License
This software is licensed under the Apache-2.0 license.

## Authors
ConLAi is developed by [Rosso inc](https://www.rosso-tokyo.co.jp/).
