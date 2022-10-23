# Kaissandra
Kaissandra is an Autonomous Trading Platfrom based on artificial intelligence (AI) that tracks, perdicts, and executes positions on potentially 
any financial asset. It currently operates in the Forex markets and relies on the MetaTrader 5 (MT5) software as broker platform for executing orders.

## Structure
Kaissandra consists of two packages: batch and online. In batch, all modules and functions related with the offline learning and parameter setting 
of the system are contained. The online package (kaissandra/prod) contains all functionalities related with live tracking and prediction of the market, and execution 
of orders. Back testing of strategies is also implemented in the online package.

## How to Use
1. Run setup.py
2. Launch MT5 with the MT5/FTR.mq5 file attached to target assets (e.g. EURUSD, GPBUSD, ...)
3. Lauch Kaissandra webapp as backend (see [kaissandra_webapp](https://github.com/mgutierrez09/kaissandra_webapp)). I recommend Heroku for hosting the backend
4. Run kaissandra/prod/run.py

Start a session:
```python
import os
import sys
this_path = os.getcwd()
# in linux
path = '/'.join(this_path.split('/')[:-1])+'/'
# in windows
path = '\\'.join(this_path.split('\\')[:-1])+'\\'
sys.path.insert(0, path)
from kaissandra.automate import *
from kaissandra.config import *
```

Shut down live session:
```python
import kaissandra.prod.communication as ct
ct.shutdown()
```
## Performance
Over a two-year period (from May 2019 until May 2021) Kaissandra obtained a return of investment (ROI) of close to 40%, outperforming all standard indicatiors like SPE500. 
![alt text](https://github.com/mgutierrez09/kaissandra/blob/master/ROI.png?raw=true)
## Author

[Miguel A. Gutierrez-Estevez] (https://www.linkedin.com/in/magutierrezestevez/)
