import json
import os


META_FLDR = "../../data/meta"
TRAIN_META = "fake_train.json"
VAL_META = "fake_val.json"

os.makedirs(META_FLDR, exist_ok=True)

stage_id_brain = "AD_pos/002_S_0729_I291876/lh"
train_list = []
val_list = []

# Train.
for i in range(100):
    train_list.append(stage_id_brain)

with open(os.path.join(META_FLDR, TRAIN_META), "w") as file:
    json.dump(train_list, file)

# Validation.
for i in range(12):
    val_list.append(stage_id_brain)

with open(os.path.join(META_FLDR, VAL_META), "w") as file:
    json.dump(val_list, file)





