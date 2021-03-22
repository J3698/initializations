import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def visualize_runs():
    ea = event_accumulator.EventAccumulator('./runs/Mar15_16-07-50_ip-172-31-71-255/events.out.tfevents.1615824470.ip-172-31-71-255.2289.0',
         size_guidance={ # see below regarding this argument
             event_accumulator.COMPRESSED_HISTOGRAMS: 0,
             event_accumulator.IMAGES: 0,
             event_accumulator.AUDIO: 0,
             event_accumulator.SCALARS: 0,
             event_accumulator.HISTOGRAMS: 0,
         })
    ea.Reload()
    frames = [(tag, pd.DataFrame(ea.Scalars(tag))) for tag in ea.Tags()['scalars']]
    for (tag, frame) in frames:
        columns = list(frame.columns)
        columns[-1] = str(tag)
        frame.columns = columns
        print(columns)
        frame.drop(columns = ["wall_time"], inplace = True)

    frames[0][1]
    frames[1][1]


