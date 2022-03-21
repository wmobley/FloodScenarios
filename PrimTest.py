import pandas as pd
import numpy as np
import dictionary as reference
import seaborn as sns
import time
from shapely.geometry import Point


from ema_workbench import (MultiprocessingEvaluator)
# import rhodium as rh
import ema_workbench
from ema_workbench import (Model,  Constant, CategoricalParameter, RealParameter, ScalarOutcome, ema_logging,
                           perform_experiments)
import matplotlib.pyplot as plt
import numba
import warnings
warnings.filterwarnings(action='once')
import prim

import matplotlib.pyplot as plt
import Mitigation as mi

start = time.time()

print(time.time() - start)
fh = r'./runsAll.tar.gz'
results = ema_workbench.load_results(fh)
experiments, outcomes = results
outcomes["BCR"]   =np.array([outcomes["Savings"][i] /  outcomes["Cost"][i] for i in range(len(outcomes["Savings"]))])
b = {"Structures":np.array([outcomes["Buyout_Structures"][i] +  outcomes["Elevation_Structures"][i] for i in range(len(outcomes["Savings"]))])
}
outcomes.update(b)
outcomes= pd.DataFrame(outcomes)

x = experiments
def test(x):
    if x['Elevation_BCR']< 3 and x['Elevation_Structures']<2000 and x['Buyout_Structures']<1200  and x['Buyout_BCR']<100:

        return 1
    else: return 0# Insert Key 1 Here


y = outcomes.apply(lambda x: test(x), axis =1)





p = prim.Prim(x, y, threshold=0, threshold_type=">")


box = p.find_box()
box.show_tradeoff()

plt.show()