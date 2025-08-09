import pandas as pd
import numpy as np

TIMESTEPS = 20
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']

df = pd.read_csv("training-data.csv")
df = df.drop("Nome", axis=1)

class KickSplitCreator:
    def __init__(self, timesteps, num_boards, sensors, axes, dataframe, num_cols_out, counter):
        self.timesteps = timesteps
        self.num_boards = num_boards
        self.sensors = sensors
        self.axes = axes
        self.in_df = dataframe
        self.single_sample = num_boards * len(sensors) * len(axes)
        self.num_cols_out = num_cols_out
        self.counter = counter

    def generate_column_names(self):
        column_names = []
        for i in range(self.num_cols_out):
            suffix = f"_{i + 1}"
            for board_id in range(1, self.num_boards + 1):
                for sensor_type in self.sensors:
                    for axis in self.axes:
                        col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                        column_names.append(col_name)
        column_names.insert(0, "Class")
        return column_names

    def extract_from_row(self):
        for i in range(self.timesteps - 1):
            single_kick_data = self.array[self.counter * self.single_sample:(self.counter * self.single_sample + self.num_cols_out * self.single_sample)].astype(str)
            if self.counter == 0: classe = "Inizio"
            elif self.counter == 18: classe = "Fine"
            else: classe = "Calcio"
            single_kick_data = np.insert(single_kick_data, 0, classe)
            self.counter += 1
            self.out_df.loc[len(self.out_df)] = single_kick_data


    def run(self):
        self.out_df = pd.DataFrame(columns=self.generate_column_names())
        for i in range(len(self.in_df)):
            row = self.in_df.iloc[i]
            self.array = row.to_numpy()
            self.counter = 0
            self.extract_from_row()

        return self.out_df


if __name__ == "__main__":
    count = 0
    dc = KickSplitCreator(
        timesteps=TIMESTEPS,
        num_boards=NUM_BOARDS,
        sensors=SENSORS,
        axes=AXES,
        dataframe=df,
        num_cols_out=2,
        counter=count)

    out_df = dc.run()

    out_df.to_csv('kicksplit_dataset.csv', index=False)
