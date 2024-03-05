import pandas as pd
from TableMiner.LearningPhase.Update import TableLearning,  updatePhase
Table = pd.read_csv("E:\Project\EDBTDemo\Datasets\T2DV2_122.csv") #125
print(Table, "\n")
tableLearning = TableLearning(Table)
tableLearning.table_learning()
updatePhase(tableLearning)
"""annotation = TA(table)
annotation.subcol_Tjs()
print( annotation.column_score)
"""


# annotation_entity = {0: 3.662611942715209, 3: 1.7804230716137477, 4: 1.5887996947126874, 5: 1.4407044164353653}

