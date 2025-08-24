# pyConLAi
Client optimizer for use with ConLAi which is ledger type federated learning framework 


## What's ConLAi?
Con(sensus)L(erning) Ai is server module for Ledger type federated learning.
Ledger type federated learning achieves federated learning in a way that feels like Git.

## How to Start
Here is how to run the CIFAR10 example:

#### 1. Server-side execution
This Python module is a client module. The ConLAi service requires the server to be started.  

Download the server module from the URL below:  
https://github.com/rosso-ai/conlai


and start the server with the following command.
```shell
conlai
```

See also the server module README for more information.

#### 2. Client-side execution
Next, start the client side. This sample runs two client nodes in multi-process mode.  

```shell
cd examples/cifar10
python run.py conf/dsgd_cifar10.yml
```

## License
This software is Apache license.


## Authors
ConLAi is developed by [Rosso inc](https://www.rosso-tokyo.co.jp/).
