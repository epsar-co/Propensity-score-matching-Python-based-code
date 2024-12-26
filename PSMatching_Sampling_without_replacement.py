
# CAMARGO COHORT STUDY

##################################################################################

# Propensity score matching code
# Sampling without replacement
# Ratio 1:2, caliper = 0.20
# https://github.com/epsar-co/Propensity-Score-Matching-Python-based-code.git

##################################################################################



import logging
import pandas as pd
import pyreadstat
from statsmodels.api import Logit
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Load the .sav file
        logging.info('Loading data from SAV...')
        data, meta = pyreadstat.read_sav(r'C:\PATH_TO_YOUR_DATASET.sav')
        logging.info('Data loaded successfully.')
        
        # Standardize column names
        data.columns = data.columns.str.strip().str.upper()
        
        # Define variables in capital letters:
        ID = 'COV1'
        covs = 'COV2 COV3 COV4 COV5'  
        treat = 'COV6'
        covsC = covs_all = covs.split()
        
        # Verify treatment variable
        logging.info('Checking treatment variable...')
        if treat not in data.columns:
            raise ValueError(f"Treatment variable '{treat}' not found in dataset.")
        if data[treat].nunique() != 2:
            raise ValueError("Treatment variable must be binary (0/1).")
        logging.info('Treatment variable is valid.')

        # Calculate propensity scores
        logging.info('Fitting logistic regression model...')
        X = data[covs_all]
        y = data[treat]
        logit_model = Logit(y, X).fit(disp=0)
        data['Propensity score'] = logit_model.predict(X)
        logging.info('Logistic regression model fitted and propensity scores estimated.')

        # Separate treated and control groups
        treated_data = data[data[treat] == 1]
        control_data = data[data[treat] == 0]

        # Number of neighbors to match (1:N matching)
        N_neighbors = 2  # Replace with the desired number of controls per treated case

        # Perform nearest neighbor matching with 1:N ratio and a caliper
        logging.info(f'Performing nearest neighbor matching without replacement, 1:{N_neighbors} ratio and caliper=0.20')
        nn = NearestNeighbors(n_neighbors=N_neighbors, algorithm='ball_tree')
        nn.fit(control_data[['Propensity score']])

        # Find the nearest neighbors for each treated unit
        distances, indices = nn.kneighbors(treated_data[['Propensity score']])

        # Apply caliper and perform matching without replacement
        caliper = 0.20
        matched_indices = []
        used_controls = set()  # Track already matched control indices

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            valid_mask = dist <= caliper
            treated_idx = treated_data.index[i]
            valid_controls = [
                control_idx for control_idx in control_data.iloc[idx[valid_mask]].index
                if control_idx not in used_controls
            ]
            
            # Select up to N_neighbors controls with replacement
            selected_controls = valid_controls[:N_neighbors]
            if selected_controls:
                matched_indices.extend([(treated_idx, ctrl_idx) for ctrl_idx in selected_controls])
                used_controls.update(selected_controls)

        # Combine matched data
        if matched_indices:
            treated_indices, control_indices = zip(*matched_indices)  # Unpack treated and control indices
            matched_data = pd.concat([
                data.loc[list(treated_indices)],
                data.loc[list(control_indices)]
            ])
            logging.info('Matching completed successfully.')
        else:
            raise ValueError("No matches were found. Check your caliper setting or data.")

        # Save matched pairs with IDs
        matched_pairs = pd.DataFrame(matched_indices, columns=['Treated_Index', 'Control_Index'])
        matched_pairs['Treated_ID'] = data.loc[matched_pairs['Treated_Index'], ID].values
        matched_pairs['Control_ID'] = data.loc[matched_pairs['Control_Index'], ID].values
        matched_pairs['Treated_PS'] = data.loc[matched_pairs['Treated_Index'], 'Propensity score'].values
        matched_pairs['Control_PS'] = data.loc[matched_pairs['Control_Index'], 'Propensity score'].values

        # Save matched pairs to CSV
        output_folder = r"C:\PATH_TO_YOUR_FOLDER"
        os.makedirs(output_folder, exist_ok=True)
        matched_pairs_path = os.path.join(output_folder, f"Matched_pairs_1to{N_neighbors}_without_replacement.csv")
        matched_pairs.to_csv(matched_pairs_path, index=False)
        logging.info(f'Matched pairs table saved to "{matched_pairs_path}".')
        
        
        
          # Plot propensity score distributions
        logging.info('Generating propensity score distribution plot: density')
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data['Propensity score'], color='orange', label='Before Matching', fill=True)
        sns.kdeplot(matched_data['Propensity score'], color='green', label='After Matching', fill=True)
        plt.grid(linestyle=':')
        plt.title('Propensity Score Distribution: Density')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"Propensity_Score_Distribution_Density.png"))
        plt.tight_layout()
        plt.show()
        

        # Plot standardized mean differences before and after matching
        logging.info('Generating SMD barplot')
        before_std_diff = (treated_data[covsC].mean() - control_data[covsC].mean()) / data[covsC].std()
        after_std_diff = (matched_data[matched_data[treat] == 1][covsC].mean() -
                          matched_data[matched_data[treat] == 0][covsC].mean()) / data[covsC].std()

        std_diff_df = pd.DataFrame({
            'Covariate': covsC,
            'SMD Before Matching': before_std_diff.values,
            'SMD After Matching': after_std_diff.values
        }).melt(id_vars='Covariate', var_name='Matching Status', value_name='Standardized Mean Difference')



        plt.figure(figsize=(10, 6))
        sns.barplot(data=std_diff_df, x='Standardized Mean Difference', y='Covariate', hue='Matching Status')
        plt.title('Standardized Mean Differences Before and After Matching')
        plt.axvline(0, color='gray', linestyle='solid')
        plt.axvline(-0.1, color='black', linestyle=':')
        plt.axvline(0.1, color='black', linestyle=':')
        plt.grid(color='grey', linestyle=':')
        plt.savefig(os.path.join(output_folder, "SMD_Before_and_After_Matching_barplot.png"))
        plt.legend(loc='best', fontsize='8')
        plt.tight_layout()
        plt.show()


     
        # Calculate SMD
        def calculate_smd(data, covariate, treat):
            treated_mean = data[covariate][data[treat] == 1].mean()
            untreated_mean = data[covariate][data[treat] == 0].mean()
            pooled_sd = np.sqrt(
                ((data[covariate][data[treat] == 1].std() ** 2) +
                 (data[covariate][data[treat] == 0].std() ** 2)) / 2
            )
            return (treated_mean - untreated_mean) / pooled_sd

        smd_pre_matching = {cov: calculate_smd(data, cov, treat) for cov in covsC}
        smd_post_matching = {cov: calculate_smd(matched_data, cov, treat) for cov in covsC}

        # SMD plot
        logging.info('Generating SMD lineplot')
        smd_df = pd.DataFrame({
        'Covariate': list(smd_pre_matching.keys()) * 2,
        'SMD': list(smd_pre_matching.values()) + list(smd_post_matching.values()),
        'Status': ['All Data'] * len(covsC) + ['Matched Data'] * len(covsC)
        })

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=smd_df, x='Status', y='SMD', hue='Covariate', marker='o', linestyle='solid')
        plt.title('Standardized Mean Differences')
        plt.axhline(0, color='black', linestyle=':')
        plt.axhline(-0.1, color='black', linestyle='--')
        plt.axhline(0.1, color='black', linestyle='--')
        plt.xlabel('STATUS', fontsize= '10')
        plt.ylabel('SMD', fontsize= '10')
        plt.savefig(os.path.join(output_folder, "Standardized_Mean_Differences_lineplot.png"))
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='8')
        plt.grid(color='grey', linestyle= ':')
        plt.tight_layout()
        plt.show()


        # Separate treated and control groups
        treated = treated_data = data[data[treat] == 1]
        control = control_data = data[data[treat] == 0]

       
        # Visualize propensity scores
        logging.info('Generating propensity score histplot: Frequency')
        plt.figure(figsize=(8, 6))
        plt.hist(data['Propensity score'], bins=20, alpha=0.5, label='All', color='grey')
        plt.hist(treated['Propensity score'], bins=20, alpha=0.5, label='Treated', color='red')
        plt.hist(control['Propensity score'], bins=20, alpha=0.5, label='Control', color='orange')
        plt.legend()
        plt.title("Propensity Score Distribution: Frequency")
        plt.xlabel("Propensity Score")
        plt.ylabel("Frequency")
        plt.grid(linestyle=':')
        plt.savefig(os.path.join(output_folder, "Propensity_Score_Distribution_Frequency.png"))
        plt.tight_layout()
        plt.show()


      
        logging.info('Plots generated successfully.')

    except Exception as e:
        logging.error(f'An error occurred: {e}')


# Run the main function
if __name__ == '__main__':
    main()
      
      

    
