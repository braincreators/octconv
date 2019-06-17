# Image Classification Benchmarks

## Setup

- Install `octconv`:

   - (Option 1) From pip:
        
         pip install octconv

   - (Option 2) Locally: 
    
         pip install -e ..   
 
- Install remaining requirements

        pip install -r requirements.txt


## Training

### Single GPU

    python train.py -c configs/cifar10/oct-resnet20.yml --device cuda:0

### Multi-GPU

    NGPUS=4; python -m torch.distributed.launch --nproc_per_node ${NGPUS} train.py -c configs/cifar10/oct-resnet20.yml
