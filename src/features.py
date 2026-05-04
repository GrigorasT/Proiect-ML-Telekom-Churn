def add_custom_features(df):
    """
    PART 4: FEATURE ENGINEERING
    Adaugă variabile derivate pentru a îmbunătăți performanța modelului.
    """
    df_mod = df.copy()
    # Feature nou: cost mediu pe lună raportat la tenure
    df_mod['AvgCostPerMonth'] = df_mod['TotalCharges'] / (df_mod['tenure'] + 1)
    # Feature nou: client nou (primele 6 luni)
    df_mod['IsNewCustomer']   = (df_mod['tenure'] <= 6).astype(int)
    
    return df_mod
