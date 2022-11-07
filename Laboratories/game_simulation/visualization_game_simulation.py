#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("output_simulation.csv")
# df.head()
#%%
grouped_df = df.groupby(['initial_players', 'area', 'mobility_speed'], as_index=False)[["time", "avg_killed_opponents", "winner_killed_opponents"]].mean()
# grouped_df.info()
#%%
fig, ax = plt.subplots(3, 3, figsize=(20, 20))

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[0][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["time"].values, label="initial_players=3, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[0][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["time"].values, label="initial_players=7, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[0][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["time"].values, label="initial_players=11, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[0][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["time"].values, label="initial_players=3, mobility_speed=3")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[0][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["time"].values, label="initial_players=7, mobility_speed=3")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[0][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["time"].values, label="initial_players=11, mobility_speed=3")

ax[0][0].legend()
ax[0][0].set_xlabel("area")
ax[0][0].set_ylabel("time")
ax[0][0].grid()

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["area"] == 5) ].index.values
ax[0][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["time"].values, label="initial_players=3, area=5")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["area"] == 11) ].index.values
ax[0][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["time"].values, label="initial_players=7, area=11")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["area"] == 21) ].index.values
ax[0][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["time"].values, label="initial_players=11, area=21")

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["area"] == 5) ].index.values
ax[0][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["time"].values, label="initial_players=3, area=5")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["area"] == 11) ].index.values
ax[0][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["time"].values, label="initial_players=7, area=11")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["area"] == 21) ].index.values
ax[0][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["time"].values, label="initial_players=11, area=21")

ax[0][1].legend()
ax[0][1].set_xlabel("mobility_speed")
ax[0][1].set_ylabel("time")
ax[0][1].grid()

idx = grouped_df[ (grouped_df["area"] == 5) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[0][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["time"].values, label="area=5, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 11) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[0][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["time"].values, label="area=11, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 21) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[0][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["time"].values, label="area=21, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 5) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[0][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["time"].values, label="area=5, mobility_speed=3")

idx = grouped_df[ (grouped_df["area"] == 11) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[0][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["time"].values, label="area=11, mobility_speed=3")

idx = grouped_df[ (grouped_df["area"] == 21) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[0][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["time"].values, label="area=21, mobility_speed=3")

ax[0][2].legend()
ax[0][2].set_xlabel("initial_players")
ax[0][2].set_ylabel("time")
ax[0][2].grid()

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[1][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=3, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[1][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=7, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[1][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=11, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[1][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=3, mobility_speed=3")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[1][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=7, mobility_speed=3")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[1][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=11, mobility_speed=3")

ax[1][0].legend()
ax[1][0].set_xlabel("area")
ax[1][0].set_ylabel("avg_killed_opponents")
ax[1][0].grid()

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["area"] == 5) ].index.values
ax[1][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=3, area=5")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["area"] == 11) ].index.values
ax[1][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=7, area=11")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["area"] == 21) ].index.values
ax[1][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=11, area=21")

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["area"] == 5) ].index.values
ax[1][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=3, area=5")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["area"] == 11) ].index.values
ax[1][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=7, area=11")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["area"] == 21) ].index.values
ax[1][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="initial_players=11, area=21")

ax[1][1].legend()
ax[1][1].set_xlabel("mobility_speed")
ax[1][1].set_ylabel("avg_killed_opponents")
ax[1][1].grid()

idx = grouped_df[ (grouped_df["area"] == 5) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[1][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="area=5, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 11) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[1][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="area=11, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 21) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[1][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="area=21, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 5) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[1][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="area=5, mobility_speed=3")

idx = grouped_df[ (grouped_df["area"] == 11) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[1][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="area=11, mobility_speed=3")

idx = grouped_df[ (grouped_df["area"] == 21) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[1][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["avg_killed_opponents"].values, label="area=21, mobility_speed=3")

ax[1][2].legend()
ax[1][2].set_xlabel("initial_players")
ax[1][2].set_ylabel("avg_killed_opponents")
ax[1][2].grid()

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[2][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=3, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[2][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=7, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[2][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=11, mobility_speed=1")

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[2][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=3, mobility_speed=3")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[2][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=7, mobility_speed=3")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[2][0].plot(grouped_df.loc[idx]["area"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=11, mobility_speed=3")

ax[2][0].legend()
ax[2][0].set_xlabel("area")
ax[2][0].set_ylabel("winner_killed_opponents")
ax[2][0].grid()

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["area"] == 5) ].index.values
ax[2][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=3, area=5")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["area"] == 11) ].index.values
ax[2][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=7, area=11")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["area"] == 21) ].index.values
ax[2][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=11, area=21")

idx = grouped_df[ (grouped_df["initial_players"] == 3) & (grouped_df["area"] == 5) ].index.values
ax[2][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=3, area=5")

idx = grouped_df[ (grouped_df["initial_players"] == 7) & (grouped_df["area"] == 11) ].index.values
ax[2][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=7, area=11")

idx = grouped_df[ (grouped_df["initial_players"] == 11) & (grouped_df["area"] == 21) ].index.values
ax[2][1].plot(grouped_df.loc[idx]["mobility_speed"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="initial_players=11, area=21")

ax[2][1].legend()
ax[2][1].set_xlabel("mobility_speed")
ax[2][1].set_ylabel("winner_killed_opponents")
ax[2][1].grid()


idx = grouped_df[ (grouped_df["area"] == 5) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[2][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="area=5, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 11) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[2][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="area=11, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 21) & (grouped_df["mobility_speed"] == 1) ].index.values
ax[2][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="area=21, mobility_speed=1")

idx = grouped_df[ (grouped_df["area"] == 5) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[2][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="area=5, mobility_speed=3")

idx = grouped_df[ (grouped_df["area"] == 11) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[2][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="area=11, mobility_speed=3")

idx = grouped_df[ (grouped_df["area"] == 21) & (grouped_df["mobility_speed"] == 3) ].index.values
ax[2][2].plot(grouped_df.loc[idx]["initial_players"].values, grouped_df.loc[idx]["winner_killed_opponents"].values, label="area=21, mobility_speed=3")

ax[2][2].legend()
ax[2][2].set_xlabel("initial_players")
ax[2][2].set_ylabel("winner_killed_opponents")
ax[2][2].grid()

plt.savefig("visualization_game_simulation.png")
plt.show()
#%% md
