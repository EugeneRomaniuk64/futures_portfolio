import pandas as pd
import os

class DynamicRoll:
    def import_futures_data_from_csv(self, file_name: str) -> pd.DataFrame:
        try:
            file_path = os.path.join(os.getcwd(), file_name)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at: {file_path}. Ensure '{file_name}' is in the same directory.")
                
            df = pd.read_csv(file_path, sep=r',', engine='python', header=None)
            

            df.columns = ['TICKER', 'DATE', 'PRICE']

            
            
            df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
            df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
            print(df.head())
            df = df.dropna(subset=['PRICE', 'TICKER']).reset_index(drop=True)
            df = df[df['PRICE'] != 0]
            
            df = df.sort_values('DATE').reset_index(drop=True)
            
            print(f"Successfully loaded {len(df)} contracts from '{file_name}'.")
            
            return df
            
        except FileNotFoundError as fnfe:
            print(fnfe)
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred during CSV loading: {e}")
            return pd.DataFrame()
        
    def dynamic_roll_optimizer(self, df_curve: pd.DataFrame) -> dict:
        if df_curve.empty or len(df_curve) < 2:
            return {"Status": "Error", "Message": "Insufficient data to calculate roll yield."}
        
        front_contract = df_curve.iloc[0]
        P_front = front_contract['PRICE']
        T_front = front_contract['DATE']
        
        roll_results = []

        for i in range(1, len(df_curve)):
            roll_in_contract = df_curve.iloc[i]
            P_roll_in = roll_in_contract['PRICE']
            T_roll_in = roll_in_contract['DATE']
            

            time_diff_days = (T_roll_in - T_front).days
            if time_diff_days <= 0: continue
            
            Delta_t = time_diff_days / 365
            

            implied_roll_yield = (P_front - P_roll_in) / P_roll_in / Delta_t
            
            roll_results.append({
                'RollInTicker': roll_in_contract['TICKER'],
                'ImpliedYield': implied_roll_yield,
                'DeltaT_Years': Delta_t,
                'FrontPrice': P_front,
                'RollInPrice': P_roll_in,
                'Description': roll_in_contract['DATE'].strftime("%b-%y")
            })

        df_results = pd.DataFrame(roll_results)
        
        optimal_roll = df_results.loc[df_results['ImpliedYield'].idxmax()]
        
        return {
            "FrontContract": front_contract['TICKER'],
            "OptimalRollIn": optimal_roll['RollInTicker'],
            "MaxYield": optimal_roll['ImpliedYield'],
            "FullResults": df_results
        }



if __name__ == "__main__":
    roller = DynamicRoll()

    curve_data = roller.import_futures_data_from_csv("./data/RB.csv")
    optimization_result = roller.dynamic_roll_optimizer(curve_data)
    print("\n--- S&P GSCI Dynamic Roll Recommendation ---")
    print(f"Current Front Contract: **{optimization_result['FrontContract']}**")
    print(f"Optimal Roll-In Contract: **{optimization_result['OptimalRollIn']}**")
    print(f"Maximum Implied Roll Yield (Annualized): **{optimization_result['MaxYield']:.2%}**") 
    print("\n--- Full Roll Yield Comparison ---")
    display_cols = ['RollInTicker', 'Description', 'ImpliedYield', 'RollInPrice', 'DeltaT_Years']
    

    df_display = optimization_result['FullResults'][display_cols].copy()
    df_display['ImpliedYield'] = (df_display['ImpliedYield'] * 100).round(2).astype(str) + '%'
    
    print(df_display.to_markdown(index=False, floatfmt=".2f"))