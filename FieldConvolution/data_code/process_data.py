import os


def save_obj():
    """Save the obj file.
    """
    os.makedirs(OBJ_FLDR, exist_ok=True)
    for stage in STAGE_LIST:
        with open(os.path.join(META_FLDR, stage + ".json"), "r") as file:
            meta_list = json.load(file)
        for meta in tqdm(meta_list):
            os.makedirs(os.path.join(OBJ_FLDR, meta[0]), exist_ok=True)
            m_file = os.path.join(DATA_FLDR, meta[0], meta[1] + ".m")
            obj_file = os.path.join(OBJ_FLDR, meta[0], meta[1] + ".obj")
            if not os.path.exists(obj_file):
                try:
                    m2obj(m_file, obj_file)
                except ValueError:
                    print("File {} is broken".format(m_file))
                    continue

if __name__ == '__main__':
    save_obj()
