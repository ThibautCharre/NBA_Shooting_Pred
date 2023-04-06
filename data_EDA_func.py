import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class DataCollector:

    def __init__(self, df_cleaned):
        self.df = df_cleaned

    def var_boxplot(self, vert_var='Position', horiz_var='FG_player', bucket_type='made'):
        df_filtered = self.df[self.df['result_str'] == bucket_type]
        sns.boxplot(
            data=df_filtered,
            x=vert_var,
            y=horiz_var
        )

    def category_result(self, var_selected='Position'):
        category_table = pd.pivot_table(
            data=self.df,
            index=var_selected,
            columns='result_str',
            aggfunc='size', fill_value=0
        )
        category_table['FG%'] = round(
            100 * category_table['made'] / (category_table['made'] + category_table['missed']),
            2
        )
        return category_table

    def category_result_bar(self, var_selected='Position'):
        category_table = self.category_result(
            var_selected=var_selected
        )
        category_melt = pd.DataFrame(
            category_table
        ).reset_index()
        category_melt = pd.melt(
            category_melt,
            id_vars=[var_selected],
            value_vars=['made', 'missed'],
            value_name='Total'
        )

        plt.figure(figsize=(15, 8))
        plt.xticks(rotation=45)
        ax = sns.barplot(
            data=category_melt,
            x=var_selected,
            y='Total',
            hue='result_str'
        )
        ax2 = ax.twinx()
        sns.lineplot(
            data=category_table,
            x=var_selected,
            y='FG%', marker='o',
            color='crimson',
            lw=3,
            ax=ax2
        )
        plt.xticks(rotation=45)
        plt.show()

    def pivot_result(self, vert_var='Position', horiz_var='area_shot', bucket_type='made'):
        df_filtered = self.df[self.df['result_str'] == bucket_type]
        pivot_table = pd.pivot_table(
            df_filtered,
            index=[horiz_var],
            columns=[vert_var],
            aggfunc='size', fill_value=0
        )
        return pivot_table

    def pivot_heatmap(self, vert_var='Position', horiz_var='area_shot', bucket_type='made'):
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            data=self.pivot_result(
                vert_var=vert_var,
                horiz_var=horiz_var,
                bucket_type=bucket_type
            )
        )
        plt.show()

    def int_correlation_map(self, drop_variables='result'):
        df_int_corr = self.df.select_dtypes(include=['int64'])
        df_int_corr = df_int_corr.drop(drop_variables, axis=1)

        plt.figure(figsize=(15, 8))
        sns.heatmap(df_int_corr.corr(), annot=True)
        plt.xticks(rotation=45)
        plt.show()



