
Summary
=======

This repo contains the implementation of the model proposed in `Dynamic Relation Learning for Link Prediction in Knowledge Hypergraphs` for knowledge hypergraph embedding. 


_Note however that the code is designed to handle graphs with arity at most 6 (which is the case for JF17K)._

The software can be also used as a framework to implement new knowledge hypergraph embedding models.

## Dependencies

* `Python` version 3.7
* `Numpy` version 1.17
* `PyTorch` version 1.4.0

## Docker
We recommend running the code inside a `Docker` container. 
To do so, you will first need to have [Docker installed](https://docs.docker.com/).
You can then compile the image with:
```console
docker build -t hype-image:latest  .
```

and run using (replace the path to your local repo):
```console
docker run --rm -it -v {HypE-code-path}:/eai/project --user `id -u`:`id -g` hype-image /bin/bash
```

## Usage

The default values for most of these parameters are the ones that were used to obtain the results in the paper.




# RD--MPNN
# RD--MPNN
# random
