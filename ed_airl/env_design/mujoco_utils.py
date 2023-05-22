

def set_as_impossible(model):
    # Disable all controls
    model.actuator_ctrlrange[:] = 0. # buggy ?
    model.actuator_gear[:] = 0.
