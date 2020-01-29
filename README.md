# Kaissandra

Kaissandra is an Autonomous Trading Platfrom based on artificial intelligence (AI) that tracks, perdicts, and executes positions on potentially 
any financial asset. It currently operates in the Forex markets and relies on the MetaTrader 5 (MT5) software as broker platform for executing orders.

## API

Example to change trading parameters:

`$ http PUT https://kaissandra-webapp.herokuapp.com/api/traders/sessions/change_params stoploss=<value> lots=<value> "Authorization:Bearer <token>"`

Start a session:

```python
import os
import sys
this_path = os.getcwd()
path = '/'.join(this_path.split('/')[:-1])+'/'
sys.path.insert(0, path)
from kaissandra.automate import *
from kaissandra.config import *
```
 
## Author

[Miguel A. Gutierrez-Estevez] (https://www.linkedin.com/in/magutierrezestevez/)
