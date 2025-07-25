#L1 English, L2 Dutch, Simultaneous
!pip install matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Your data as a CSV string
from io import StringIO

#input the relevant data information for l1/l2 accuracy 
data = """
    cond,checkpoint,l1,l2,l1_acc,l2_acc
    simultaneous, 0, eng, nld, 0.635065, 0.544402
    simultaneous, 10000, eng, nld, 0.918182, 0.606607
    simultaneous, 20000, eng, nld, 0.950649, 0.590734
    simultaneous, 30000, eng, nld, 0.964935, 0.638782
    simultaneous, 40000, eng, nld,0.95974, 0.624625
    simultaneous, 50000, eng, nld,0.963636, 0.633634
    simultaneous, 64000, eng, nld,0.971429, 0.608323
    simultaneous, 64010, eng, nld,0.968831, 0.661519
    simultaneous, 64020, eng, nld,0.966234, 0.66967
    simultaneous, 64030, eng, nld,0.962338, 0.681253
    simultaneous, 64040, eng, nld,0.961039, 0.699271
    simultaneous, 64050, eng, nld,0.958442, 0.706993
    simultaneous, 64060, eng, nld, 0.961039, 0.709567
    simultaneous, 64070, eng, nld,0.961039, 0.722866
    simultaneous, 64080, eng, nld,0.961039, 0.728014
    simultaneous, 64090, eng, nld,0.95974, 0.730159
    simultaneous, 64100, eng, nld,0.964935, 0.731875
    simultaneous, 64110, eng, nld,0.966234, 0.727156
    simultaneous, 64120, eng, nld, 0.967532, 0.732733
    simultaneous, 64130, eng, nld,0.967532, 0.73402
    simultaneous, 64140, eng, nld,0.968831, 0.743887
    simultaneous, 64150, eng, nld,0.967532, 0.73831
    simultaneous, 64160, eng, nld,0.968831, 0.736165
    simultaneous, 64170, eng, nld,0.968831, 0.73831
    simultaneous, 64180, eng, nld,0.966234, 0.744316
    simultaneous, 64190, eng, nld,0.961039, 0.744316
    simultaneous, 64200, eng, nld, 0.963636, 0.744316
    simultaneous, 64300, eng, nld,0.961039, 0.767053
    simultaneous, 64400, eng, nld,0.966234, 0.77692
    simultaneous, 64500, eng, nld,0.968831, 0.791506
    simultaneous, 64600, eng, nld,0.962338, 0.800944
    simultaneous, 64700, eng, nld,0.968831, 0.818104
    simultaneous, 64800, eng, nld,0.95974, 0.81982
    simultaneous, 64900, eng, nld,0.95974, 0.835693
    simultaneous, 65000, eng, nld, 0.958442, 0.844273
    simultaneous, 66000, eng, nld,0.958442, 0.873445
    simultaneous, 67000, eng, nld, 0.961039, 0.89275
    simultaneous, 68000, eng, nld,0.967532, 0.900901
    simultaneous, 69000, eng, nld,0.963636, 0.910768
    simultaneous, 70000, eng, nld,0.967532, 0.912913
    simultaneous, 80000, eng, nld,0.964935, 0.947233
    simultaneous, 90000, eng, nld,0.967532, 0.955384
    simultaneous, 100000, eng, nld,0.964935, 0.954097
    simultaneous, 110000, eng, nld,0.964935, 0.959674
    simultaneous, 120000, eng, nld, 0.97013, 0.96139
    simultaneous, 128000, eng, nld, 0.968831, 0.960532
"""



df = pd.read_csv(StringIO(data))
df_melted = df.melt(id_vars=["checkpoint"], value_vars=["l1_acc", "l2_acc"],
                    var_name="Language", value_name="Accuracy")
lang_map = {"l1_acc": "English (L1)", "l2_acc": "Dutch (L2)"} #edit for relevant languages 
df_melted["Language"] = df_melted["Language"].map(lang_map)

# Set font and colors
rcParams['font.family'] = 'Times New Roman'

# Custom color palette - edit for relevant languages 
palette = {
    "English (L1)": "#E86100",
    "Dutch (L2)": "#00124B",
}

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted, x="checkpoint", y="Accuracy",
             hue="Language", marker="o", palette=palette)
plt.title("B-GPT Accuracy Over Training (English-Dutch, Simultaneous)", fontsize=16, fontweight='bold')
plt.xlabel("Checkpoint", fontsize=14)
plt.ylabel("MultiBLiMP 1.0 Accuracy", fontsize=14)
plt.ylim(0.5, 1.0)
plt.xticks(rotation=45)
plt.legend(title="Language", title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.savefig("B-GPT_en_nl_simultaneous.png")
plt.show()



#L1 English, L2 Spanish, Simultaneous 
!pip install matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Your data as a CSV string
from io import StringIO

#input the relevant data information for l1/l2 accuracy 
data = !pip install matplotlib

#input the relevant data information for l1/l2 accuracy 
data = data = """checkpoint,l1_acc,l2_acc
0,0.650649,0.640299
10000,0.919481,0.687525
20000,0.948052,0.691067
30000,0.955844,0.691460
40000,0.964935,0.700118
50000,0.966234,0.702086
64000,0.972727,0.699331
64010,0.968831,0.723731
64020,0.970130,0.735144
64030,0.967532,0.733176
64040,0.968831,0.737898
64050,0.964935,0.744195
64060,0.968831,0.743015
64070,0.968831,0.750492
64080,0.970130,0.754034
64090,0.964935,0.756002
64100,0.968831,0.754821
64110,0.967532,0.759543
64120,0.970130,0.754427
64130,0.968831,0.766234
64140,0.968831,0.761511
64150,0.964935,0.765053
64160,0.963636,0.766234
64170,0.968831,0.766627
64180,0.963636,0.768201
64190,0.961039,0.772924
64200,0.964935,0.776072
64300,0.964935,0.801259
64400,0.964935,0.800472
64500,0.962338,0.810704
64600,0.963636,0.814640
64700,0.967532,0.821330
64800,0.967532,0.827627
64900,0.967532,0.828808
65000,0.963636,0.827627
66000,0.966234,0.855569
67000,0.970130,0.863046
68000,0.971429,0.874459
69000,0.959740,0.878788
70000,0.966234,0.886265
80000,0.964935,0.920110
90000,0.971429,0.927588
100000,0.970130,0.935458
110000,0.971429,0.934278
120000,0.970130,0.939787
128000,0.970130,0.936639
"""



df = pd.read_csv(StringIO(data))
df_melted = df.melt(id_vars=["checkpoint"], value_vars=["l1_acc", "l2_acc"],
                    var_name="Language", value_name="Accuracy")
lang_map = {"l1_acc": "English (L1)", "l2_acc": "Spanish (L2)"} #edit for relevant languages 
df_melted["Language"] = df_melted["Language"].map(lang_map)

# Set font and colors
rcParams['font.family'] = 'Times New Roman'

# Custom color palette - edit for relevant languages 
palette = {
    "English (L1)": "#E86100",
    "Spanish (L2)": "#00124B",
}

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted, x="checkpoint", y="Accuracy",
             hue="Language", marker="o", palette=palette)
plt.title("B-GPT Accuracy Over Training (English-Spanish, Simultaneous)", fontsize=16, fontweight='bold')
plt.xlabel("Checkpoint", fontsize=14)
plt.ylabel("MultiBLiMP 1.0 Accuracy", fontsize=14)
plt.ylim(0.5, 1.0)
plt.xticks(rotation=45)
plt.legend(title="Language", title_fontsize='13', fontsize='12')
plt.tight_layout()
plt.savefig("B-GPT_en_nl_simultaneous.png")
plt.show()
