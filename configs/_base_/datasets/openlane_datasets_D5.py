# ---------------------------------------------------base config start---------------------------------------------------
_base_ = ['./openlane_datasets.py']
# ----------------------------------------------------base config end----------------------------------------------------


# ---------------------------------------------------D5 subset start---------------------------------------------------
# D5 means using one frame every five samples for quick experiments.
load_interval = 5
dataset_name = 'openlane_D5'

dataset_cfg = dict(
    dataset_name=dataset_name,
    load_interval=load_interval,
)

# Keep inexpensive D5 validation during training, but never downsample the
# final official test benchmark.  tools/test.py must evaluate all 42,001 test
# annotations, including each named scenario subset.
test_dataset_cfg = dict(dataset_cfg)
test_dataset_cfg['load_interval'] = 1

train_dataloader = dict(dataset=dict(dataset_cfg=dataset_cfg))
val_dataloader = dict(dataset=dict(dataset_cfg=dataset_cfg))
test_dataloader = dict(dataset=dict(dataset_cfg=test_dataset_cfg))
# ----------------------------------------------------D5 subset end----------------------------------------------------
