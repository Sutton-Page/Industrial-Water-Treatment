import warnings
# Remove anoying warning
warnings.filterwarnings("ignore")

import scipy
import numpy as np
import pandas as pd
from setup import setup_data
from scipy.integrate import odeint
from scipy.interpolate import interp1d


def extract_inital_data(design_assumptions,variables):

    inital_values = {}

    for item in variables:

        if item !="default":

            l = ['low','expected','high']
            temp = {}

            for values in l:

                temp[values] = round(design_assumptions.loc[item][values],3)
            
            inital_values[item] = temp
    

    return inital_values
     


def group_data(params,data):

    holder = {}
    for item in params:

            temp = {}
            l = ['low','expected','high']
            found_null = False

            for inputs in l:

                check_for_null = data[item+"_"+inputs]
                if check_for_null == None:

                        found_null = True
                        continue
                
                else:
                    try:
                        temp[inputs] = float(check_for_null)

                    except ValueError:
                         
                         found_null = True 

            if found_null == False:

                item = item.replace("default_",'')
                holder[item] = temp
    
    return holder

# requires a special 
def run_pond1B(data):

    M = data['M']
    elevation_a  = data['elevation_a']
    R = data['R']
    d_e = data['d_e']
    Pa = data['Pa']
    C20 = data['C20']
    Foul_factor = data['Foul_factor']
    beta = data['beta']
    C12 = data['C12']
    S_NH3 = data['S_NH3']
    KBTHETA = data['KBTHETA']
    YBNSD = data['YBNSD']
    S_DO1 = data['S_DO1']

    Diffuser_depth = data['Diffuser_depth']

    
    IN_NH4 = data['IN_NH4']
    
    IN_Q = data['IN_Q']
    V = data['V']
   
    kd = data['kd']
    Y = data['Y']
   
    math_exp = data['math_exp']
    
    Hydrau_Reten = data['Hydrau_Reten']
  
    
  
    Coeff_Power = data['Coeff_Power']
    
    cBOD5_Multiplier = data['cBOD5_Multiplier']
   

   
  
   
    IN_NH3_z = data['IN_NH3_z']


    
    TSSi = data['TSSi']
   
    Influent_BOD = data['Influent_BOD']
   
    aeration_1B = data['aeration_1B']
  
  
    Organic_nitrogen1  = data['Organic_nitrogen1']
    f = data['f']

    max_growth_coefficient = data['max_growth_coefficient']
    Volume_1B = data['Volume_1B']

    Sup_NH3 = data['Sup_NH3']

    Ti = data['Ti']
    air_temp_1b = data['air_temp_1b']

    b_20  = data['b_20']
    K_NH4 = data['K_NH4']
    S_DO = data['S_DO']
    K_AOB = data['K_AOB']
    bh = data['bh']
    YH = data['YH']
    fd_o = data['fd_o']

    IN_PO4_z = data['IN_PO4_z']
    Sup_PO4 = data['Sup_PO4']

    Area_1A = data['Area_1A']
    Area_1B = data['Area_1B']
    air_temp = data['air_temp']

    Ti_1B = data['Ti_1B']
    g = data['g']

    EFF_PO4 = data['EFF_PO4']
    CBOD5 = data['CBOD5']

    eff_nh3_isr = data['eff_nh3_isr']
    Eff_TSS = data['Eff_TSS']

    Flow_rate_3 = data['Flow_rate_3']

    # %% Aerated Stabilization Basin

    # process models for ASB


    # Flow in Aerated Stablization Basin is set to the Flow volume from the sedimentation Tank




    Q = IN_Q * 3785.4118

    Influent_BOD1 = Influent_BOD 

    Benchmark_efficiency = 10

    VSS = TSSi * 0.80

    IN_PO4 = IN_PO4_z + Sup_PO4

    IN_NH3 = IN_NH3_z + Sup_NH3

    Area_ft2_1A = Area_1A * 10.7639

    Area_ft2_1B = Area_1B * 10.7639

    #Temperature of Pond1A

    Ta = (Area_ft2_1A * 0.000012 * air_temp + IN_Q * Ti ) / ((Area_ft2_1A * 0.000012) + 1)

    T = ((Ta - 32) * 5) / 9

    #Temperature of Pond1B

    Tai = (Area_ft2_1B * 0.000012 * air_temp_1b + IN_Q * Ti_1B ) / ((Area_ft2_1B * 0.000012) + 1)

    T_1B = ((Tai - 32) * 5) / 9



    def get_theta(temp):
        return np.where((temp >= 5) & (temp <= 15), 1.109, 
                        np.where((temp > 15) & (temp <= 30), 1.042, 0.967))

    # Assuming T and T_1B are already defined NumPy arrays
    theta_T = get_theta(T)
    theta_T1_B = get_theta(T_1B)

    # Aerated Stabilization Basin 2

    Px_1B = aeration_1B / Volume_1B

    Power_1B = Px_1B * Volume_1B

    K1 = 2.5 * (EFF_PO4 / (0.053 + EFF_PO4)) * (theta_T1_B ** (T_1B - 20))

    Effluent_BOD51 = CBOD5

    Effluent_BOD5_1B = 1 / (1 + (K1 * Hydrau_Reten)) * Effluent_BOD51    


    BOD5_removal_1B = CBOD5 - Effluent_BOD5_1B    #mg/


    x_numerator_1B = Y * (CBOD5 - Effluent_BOD5_1B)

    x_denomenator_1B = 1 + (kd * Hydrau_Reten)

    x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

    Px_1B = (x_1B * Q) / 1000  # kg/d

    Px_O2_1B = 1.42 * Px_1B

    O2_requirement_1Bisr = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d

    oxygen_supply1B = ((aeration_1B * 24) / 1000) * Coeff_Power

    BOD_Effluent_1B = ((oxygen_supply1B -  Px_O2_1B) / Q ) * (f * 1000)

    BOD5_Effluent_1B = CBOD5 - Effluent_BOD5_1B

    BOD5_Effluentnp_1B = np.percentile(BOD5_Effluent_1B, 50)


    x_numerator_1B = Y * (CBOD5 - Effluent_BOD5_1B)

    x_denomenator_1B = 1 + (kd * Hydrau_Reten)

    x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

    Px_1B = (x_1B * Q) / 1000  # kg/d

    Px_O2_1B = 1.42 * Px_1B

    O2_requirement_1B = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d


    Power_req_bod_1B = O2_requirement_1B / (Coeff_Power)  # kW


    # Calculation of oxygen transfer rate under standard conditions

    pressure = -(g * M * (elevation_a - 0)) / (R * (273.15 + 12))

    relative_pressure = math_exp ** pressure

    C_20 = 1 + d_e * (Diffuser_depth / Pa)

    Oxygen_Con_20 = C20 * C_20

    O2_requirement_h1 = oxygen_supply1B / 24

    SOTR1_isr = O2_requirement_h1 / (0.50 * Foul_factor)

    SOTR2_isr = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

    SOTR3_isr = Oxygen_Con_20 / SOTR2_isr

    SOTR4_isr = SOTR3_isr * (1.024) ** (20 - T_1B)

    SOTR5_isr = SOTR1_isr * SOTR4_isr                       #kg O2/ h

    SOTR6_isr0 = SOTR5_isr * 26.4                        #kg O2/ d

    #For oxygen requirement

    O2_requirement_h_isr1 = O2_requirement_1Bisr / 24

    SOTR1_isr1 = O2_requirement_h_isr1 / (0.50 * Foul_factor)

    SOTR2_isr1 = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

    SOTR3_isr1 = Oxygen_Con_20 / SOTR2_isr1

    SOTR4_isr1 = SOTR3_isr1 * (1.024) ** (20 - T_1B)

    SOTR5_isr1 = SOTR1_isr1 * SOTR4_isr1                       #kg O2/ h

    SOTR6_isr1 = SOTR5_isr1 * 26.4                        #kg O2/ d

    K_isr_0 = (EFF_PO4 / (0.053 + EFF_PO4)) * (theta_T1_B ** (T_1B - 20))

    Effluent_BOD_trials2_isr = 1 + (K_isr_0 * Hydrau_Reten) * (SOTR6_isr0 / SOTR6_isr1) 

    Trial_new1 = CBOD5 / Effluent_BOD_trials2_isr

    # STEP 19- Ox supply- Calculate available oxygen supply

    Power_req_sbod_1B = Power_1B - Power_req_bod_1B

    DO_supply_1B = Power_req_sbod_1B * (Coeff_Power)  # kg/d


    # STEP 13: ΔSBOD- Calculate cell effluent SBOD5

    SBOD_removal_1B = CBOD5 - Effluent_BOD5_1B  # mg/L or g/m3

    SBOD_rem_g_d_1B = SBOD_removal_1B * Q  # g/d


    SBOD_Yield_1B = BOD5_removal_1B / SBOD_rem_g_d_1B  # g VSS / sBOD


    SBOD_oxygen_used_1B = (Effluent_BOD5_1B * Q) - \
        (SBOD_removal_1B * Q)  # g O2/ g sBOD

    Oxygen_per_unit_SBOD_1B = SBOD_oxygen_used_1B / SBOD_rem_g_d_1B

    SBOD_P_1B = CBOD5 - Effluent_BOD5_1B

    Aerobic_SBOD_1b = SBOD_removal_1B - SBOD_P_1B

    # STEP 20- DOΔSBOD- Calculate SBOD5 removal supported by oxygen supply




    # BOD concentration at the specified time


    aerobic_biomass_yield = 0.5




    bh_t = bh * (1.04 ** (T_1B - 20))

    Px_bio1B = ( Q * YH * (CBOD5 - Effluent_BOD5_1B )) / 1000

    Px_bio2B = (1 + bh_t * Hydrau_Reten)

    Px_bio3B = Px_bio1B / Px_bio2B

    Px_bio4B = (fd_o * bh_t * Q * YH * (CBOD5 - Effluent_BOD5_1B ) * Hydrau_Reten) / 1000

    Px_bio5B = 1 + bh_t * Hydrau_Reten 

    Px_bio6B = Px_bio4B / Px_bio5B

    Px_bioB = Px_bio3B + Px_bio6B



    #Nitrification 

    spec_growth_AOB1 = max_growth_coefficient * (1.072 ** (T_1B - 20))

    spec_endo_decay1 = b_20 * (1.029 ** (T_1B - 20))

    growth_NH41 = spec_growth_AOB1 * (S_NH3  / (S_NH3  + K_NH4))

    spec_growth_NH41 = growth_NH41 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay1

    SRT1 = ( 1 / spec_growth_NH41) * 1.5 

    EFF_NH_ISR = spec_growth_AOB1 * (S_DO1 / (S_DO1 + K_AOB))

    EFF_NH1_ISR = eff_nh3_isr * (1 + spec_endo_decay1 * SRT1) / (SRT1 * (EFF_NH_ISR - spec_endo_decay1) - 1) 

    S_NH1 = K_NH4 * (1 + (spec_endo_decay1 * SRT1))

    growth_AOB_DO1 = spec_growth_AOB1 * ((S_DO) / (S_DO + K_AOB))

    Effluent_NH4_N_numenator1 = (SRT1 * (growth_AOB_DO1 - spec_endo_decay1)) - 1.0

    Effluent_NH4_N1 = (S_NH1 / Effluent_NH4_N_numenator1 )

    r_NH = (spec_growth_AOB1 / 0.15 ) * (eff_nh3_isr / (eff_nh3_isr + K_NH4))

    r_NH12 = r_NH * (S_DO / (S_DO + K_AOB)) * 30

    NOX1 = r_NH12 * 1.1 

    X_AOB = Q * 0.15 * NOX1 * SRT1 

    X_AOB1 = X_AOB / (V * (1 + spec_endo_decay1 * SRT1))

    Px_NH4 = (Q * 0.15 * (NOX1)) / 1000

    Px_NH41 = 1 + (0.315) * Hydrau_Reten 

    Px_NH42 = Px_NH4 / Px_NH41

    Px_bio_isr = (Px_bio3B  + Px_bio6B ) / 0.80

    Px_bio_mg1 =  (Px_bio_isr * 1000000 )/ (Q * 1000 )     #mg/L

    Total_VSS_1A = Eff_TSS * 0.8

    TSS_pro = Px_bio_isr + ((Q * (Eff_TSS - Total_VSS_1A)) / 1000)

    TSS_pro1isr = TSS_pro 

    TSS_pro2 = (TSS_pro1isr * 1000000 )/ (Q * 1000 ) + Eff_TSS

    Benchmark_efficiency1 = 84 - (10.6 * (aeration_1B / Volume_1B))

    TRY1 = Px_bio_mg1 + Total_VSS_1A

    Settled_solids1B =  (( TRY1 / 100) * Benchmark_efficiency1)

    Settling_solids_1B = (Settled_solids1B / 100) * TSS_pro2

    Eff_TSS_1B =  TRY1 - Settled_solids1B


    # Calculation of feedback1

    YBNSD1 = (0.0593 * (Settled_solids1B * 0.8 ) * 2.6 ) / 26.4

    KB = 0.08 * KBTHETA ** (T_1B - 20)

    NH3_feedback_T_ISR = ((YBNSD1 * KB * 58679.42) / Volume_1B) * 24

    eff_fb_nh31 = Effluent_NH4_N1 + NH3_feedback_T_ISR 

    biomass_growth1 = 0.5 * (CBOD5 - Effluent_BOD5_1B )



    BOD_fb1 = 0.3 * Settled_solids1B

    biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

    Uptake_N12_isr = (14 / 115 ) * biomass_growth1

    NH4_N_feedback_ISR = (Uptake_N12_isr - eff_nh3_isr + Effluent_NH4_N1) / 1

    eff_nh31 = (Uptake_N12_isr - NH4_N_feedback_ISR) + eff_nh3_isr

    effluent_nh3_con = NH4_N_feedback_ISR + Effluent_NH4_N1

    final_nh3_fb_isr = (Settled_solids1B * 0.8 ) * (0.0593 /  np.where(T > 25, 1, 2))

    final_nh3_isr = final_nh3_fb_isr + Effluent_NH4_N1

    # Define the values
    pKw = 14.0  # Example value at 25°C
    pKb = 4.75  # Example value for ammonia at 25°C

    # Assume pH is an array of pH values
    pH1 = 8 # Example array of pH values

    # Calculate the exponent for each pH value
    exponent1 = pKw - pKb - pH1 

    # Calculate 10^(exponent) for each pH value
    fraction1 = 10 ** exponent1

    # Print the result


    fraction2_1 = 1 / (1 + fraction1)

    kN1_1 = 2.71828 ** (1.57 * (pH1 - 8.5))

    kN2_1 = 2.71828 ** (0.13 * (T - 20))

    kN_1 = kN1_1 * kN2_1

    A = V / 4

    AQ = A / Q

    effluentN1_1 = AQ * kN_1 * fraction2_1

    effluentN_12 = 1 / (1 + effluentN1_1)




    # Calculation of feedback1





    biomass_growth1 = 0.5 * ( CBOD5 - Effluent_BOD5_1B )

    biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

    Uptake_N12 = (14 / 115 ) * biomass_growth1


    # Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass

    #PO4

    Px_mg_isr = 0.015 * (Px_bio_isr * 1000)   #g P/ d

    Px_mg1_isr = Px_mg_isr / Q                  # g/m3

    Effluent_P_isr = EFF_PO4 - Px_mg1_isr       #g/m3

    PO4_fb_isr = (Settled_solids1B * 0.8 )  * (0.0049 / 1)

    EFF_PO4_isr = Effluent_P_isr +  PO4_fb_isr 

    P_Uptake_ratio = 2.2

    Growth_P_Uptake1 = (P_Uptake_ratio / 115) * biomass_growth1 # mgP/L

    PO4_P_fb1 = Growth_P_Uptake1 - EFF_PO4 + Effluent_P_isr # mgP/L





    TSS_BODisr = 1 - np.exp(-0.10 * Hydrau_Reten)



    TSS_BOD1isr = Eff_TSS_1B * 0.5 * TSS_BODisr


    Soluble_BOD1 =  Trial_new1 

    Ammonia_N1 = IN_NH4 * 0.9441

    TKN1 = (final_nh3_fb_isr / 1.215) + Organic_nitrogen1  # mg/L

    CBOD51 = Soluble_BOD1  + TSS_BOD1isr

    UOD_3 = ((cBOD5_Multiplier * CBOD51) + 4.57 * final_nh3_fb_isr) * Flow_rate_3 * 8.34

    return UOD_3



def run_combined_ponds(data,isPond1A=False):

    M = data['M']
    elevation_a  = data['elevation_a']
    R = data['R']
    d_e = data['d_e']
    Pa = data['Pa']
    C20 = data['C20']
    Foul_factor = data['Foul_factor']
    beta = data['beta']
    C12 = data['C12']
    S_NH3 = data['S_NH3']
    KBTHETA = data['KBTHETA']
    YBNSD = data['YBNSD']
    S_DO1 = data['S_DO1']

    Diffuser_depth = data['Diffuser_depth']

    t0 = data['t0']
    t = data['t']
    

    max_decay = data['max_decay']
    tot = data['tot']
    k_s = data['k_s']
    math_expo = data['math_expo']
    IN_NH4 = data['IN_NH4']
   
    IN_SBOD = data['IN_SBOD']
    split_COD = data['split_COD']
    IN_Q = data['IN_Q']
    solids_percent = data['solids_percent']
    specific_gravity = data['specific_gravity']
    TS_retention = data['TS_retention']
    IN_COD = data['IN_COD']
    final_solids_content = data['final_solids_content']
    Sludge_cake_solids = data['Sludge_cake_solids']
    liq_COD_IN = data['liq_COD_IN']
    dewatering_polymer_dose = data['dewatering_polymer_dose']
    Filtrate_flowrate = data['Filtrate_flowrate']
    Filtrate_Solids = data['Filtrate_Solids']
    SP_cake_COD = data['SP_cake_COD']
    theta = data['theta']
    K_20 = data['K_20']
    V = data['V']
    X_ot = data['X_ot']
    kd = data['kd']
    Y = data['Y']
    X = data['X']
    fd = data['fd']
    Oxygen_factor = data['Oxygen_factor']
    TSSe = data['TSSe']
    Growth = data['Growth']
    Power = data['Power_x']
    Volume = data['Volume']
    CM_Power = data['CM_Power']
    CM_Volume = data['CM_Volume']
    n = data['n']
    tn = data['tn']
    Initial_Concentration = data['Initial_Concentration']
    kL = data['kL']
    t_n = data['t_n']
    math_exp = data['math_exp']
    Input_SBOD = data['Input_SBOD']
    SBODe = data['SBODe']
    Hydrau_Reten = data['Hydrau_Reten']
    Ks = data['Ks']
    k = data['k']
    PO4_P = data['PO4_P']
    K20 = data['K20']
    X_O = data['X_O']
    kpo4 = data['kpo4']
    Coeff_30_to_20 = data['Coeff_30_to_20']
    Limi_PO4_P = data['Limi_PO4_P']
    Coeff_Power = data['Coeff_Power']
    Power_limi = data['Power_limi']
    NH4_Ni = data['NH4_Ni']
    NH4_Ne = data['NH4_Ne']
    PO4_Pe = data['PO4_Pe']
    cBOD5_Multiplier = data['cBOD5_Multiplier']
    f_c = data['f_c']

    # most recent additions
    Max_speci_growth = data['Max_speci_growth']
    Ko2 = data['Ko2']
    pH = data['pH']
    Y_o = data['Y_o']
    kd_o = data['kd_o']
    SF = data['SF']
    a = data['a']
    b = data['b']
    t_h = data['t_h']
   
    IN_NH3_z = data['IN_NH3_z']

    K = data['K']
    
    TSSi = data['TSSi']
    aeration = data['aeration']
    DO_supply_limi_x = data['DO_supply_limi_x']
    Influent_BOD = data['Influent_BOD']
    Effluent_BOD = data['Effluent_BOD']
    aeration_1B = data['aeration_1B']
    Effluent_BOD_1B  = data['Effluent_BOD_1B']
    Organic_nitrogen  = data['Organic_nitrogen']
    SBOD_1B = data['SBOD_1B']
    IN_NH41 = data['IN_NH41']
    Organic_nitrogen1  = data['Organic_nitrogen1']
    f = data['f']

    max_growth_coefficient = data['max_growth_coefficient']
    Volume_1B = data['Volume_1B']

    Sup_NH3 = data['Sup_NH3']

    Ti = data['Ti']
    air_temp_1b = data['air_temp_1b']

    b_20  = data['b_20']
    K_NH4 = data['K_NH4']
    S_DO = data['S_DO']
    K_AOB = data['K_AOB']
    bh = data['bh']
    YH = data['YH']
    fd_o = data['fd_o']

    IN_PO4_z = data['IN_PO4_z']
    Sup_PO4 = data['Sup_PO4']

    Area_1A = data['Area_1A']
    Area_1B = data['Area_1B']
    air_temp = data['air_temp']

    Ti_1B = data['Ti_1B']
    g = data['g']

    Q = IN_Q * 3785.4118

    Influent_BOD1 = Influent_BOD 

    Benchmark_efficiency = 10

    VSS = TSSi * 0.80

    IN_PO4 = IN_PO4_z + Sup_PO4

    IN_NH3 = IN_NH3_z + Sup_NH3

    Area_ft2_1A = Area_1A * 10.7639

    Area_ft2_1B = Area_1B * 10.7639

    #Temperature of Pond1A

    Ta = (Area_ft2_1A * 0.000012 * air_temp + IN_Q * Ti ) / ((Area_ft2_1A * 0.000012) + 1)

    T = ((Ta - 32) * 5) / 9

    #Temperature of Pond1B

    Tai = (Area_ft2_1B * 0.000012 * air_temp_1b + IN_Q * Ti_1B ) / ((Area_ft2_1B * 0.000012) + 1)

    T_1B = ((Tai - 32) * 5) / 9



    def get_theta(temp):
        return np.where((temp >= 5) & (temp <= 15), 1.109, 
                        np.where((temp > 15) & (temp <= 30), 1.042, 0.967))

    # Assuming T and T_1B are already defined NumPy arrays
    theta_T = get_theta(T)
    theta_T1_B = get_theta(T_1B)



        

    # STEP 1: TEMP SENSITIVITY- Calculate Arrhenius temperature sensitivity coefficient

    # STEP 2: k.VSS- Adjust oxidation rate to cell temperature


    K_t = K_20 * theta_T ** (T - 20)

    Influent_nbVSS = Q * X_ot / V  # g/m3.d

    Ks_SBODe = Ks + IN_SBOD  # mg/L

    rsu = - k * X * IN_SBOD / Ks_SBODe  # g/m3.d

    r_Xt_vss = -Y * (rsu) - kd * X + fd * kd * X + Influent_nbVSS  # g/m3.d

    VSS_per_day = r_Xt_vss * Q  # g/d

    oxidation_rate = r_Xt_vss * Oxygen_factor  # g/m3.d

    Adj_Oxidation_rate = oxidation_rate * K_t  # g/m3.d

    # STEP 3: Digest Factor-Temperature-adjust benthal feedback (not Arrhenius)

    # Aerobic oxidation yield of 0.5 mg biomass per mg BOD utilized

    aerobic_biomass_yield = 0.5

    Biomass_Growth = aerobic_biomass_yield * (Influent_BOD1 - Effluent_BOD)  # mg/L




    P_Uptake_ratio = 2.2

    Growth_P_Uptake = (P_Uptake_ratio / 115) * Biomass_Growth  # mgP/L

    PO4_P_fb = Growth_P_Uptake - IN_PO4 + PO4_Pe  # mgP/L

    TSS_solu_P = (PO4_P_fb * 115) / P_Uptake_ratio  # mg/L

    Settled_solids = TSSi + Growth - TSSe  # mg/L

    Digestion_eff_P = TSS_solu_P / Settled_solids * 100  # percentage

    # Stoichiometric ratio between the uptake of nitrogen (N) and the growth of microbial biomass

    N_Uptake_ratio = 14

    Growth_N_Uptake = (N_Uptake_ratio / 115) * Biomass_Growth  # mgN/L

    NH4_N_fb = Growth_N_Uptake - IN_NH3 + NH4_Ne  # mgN/L

    TSS_solu_N = (NH4_N_fb * 115) / N_Uptake_ratio  # mg/L

    Digestion_eff_N = (TSS_solu_N / Settled_solids) * 100  # percentage


    # Calculation of Power


    P_x = aeration / Volume  # W/m3

    Power = P_x * Volume  # W

    # STEP 4: MIXING- Calculate Mixing Intenstiy

    Mixing_Intensity = Power / Volume  # W/m3

    # STEP 5: SETTLING- Calculate percent of suspended solids that settle

    TSS_sett = Settled_solids / (TSSi + Growth) * 100  # percentage

    # STEP 6: PARTIAL MIX Calculate ration of cell mixing intensity to complete mix

    Complete_Mixing = CM_Power / CM_Volume  # W/m3

    Partial_Mix = Mixing_Intensity / Complete_Mixing  # fraction

    # STEP 7: K1.VSS- Adjust baseline oxidation rate to cell partial-mix level

    Adj_OR_Partial_mix = Partial_Mix * Adj_Oxidation_rate  # g/m3.d


    # STEP 8: Cells in series- Select number of complete-mix cells to represent hydraulics

    n = 1


    # STEP 9: Calculate denominator of first-order rate equation.

    #Rate = {1 + K_t * tn} ** n
    Rate = (1 + K_t * tn) ** n

    # STEP 10: Estimate benthal feedback of SBOD5

    Sett_TSS = TSSi + Growth - TSSe


    # Estimated soluble BOD feedback per mg of TSS settled

    SBOD_feedback_ratio = 0.3

    SBOD_fb = SBOD_feedback_ratio * Sett_TSS  # mg/L


    # Calculate the BOD5 removed

    # Calculate the BOD5 removed

    K = 2.5 * (IN_PO4 / (0.053 + IN_PO4)) * (theta_T ** (T - 20))

    Effluent_BOD5 = 1 / (1 + (K * Hydrau_Reten)) * Influent_BOD1

    BOD5_removal = Influent_BOD1 - Effluent_BOD5  # mg/L

    x_numerator = Y * (Influent_BOD1 - Effluent_BOD5)

    x_denomenator = 1 + (kd * Hydrau_Reten)

    x = x_numerator / x_denomenator  # mg/L

    Px = (x * Q) / 1000  # kg/d

    Px_O2 = 1.42 * Px 

    O2_requirement = Q * (BOD5_removal / (f * 1000))-(Px_O2) # kg/d

    oxygen_supply1 = aeration * Coeff_Power

    BOD_Effluent = ((oxygen_supply1 + Px_O2 ) / (Q * 1000)) 

    BOD5_Effluent = Influent_BOD - BOD_Effluent

    Power_req_bod = O2_requirement / (Coeff_Power)  # kW

    P_x = aeration / Volume  # W/m3

    Power = P_x * Volume

    Supply_O = ((aeration * 24) / 1000) * Coeff_Power #kg O2 / d 

    Eff_BOD_kg = (BOD5_removal / 1000000) * (Q * 1000)

    O2_required_per_BOD = Eff_BOD_kg / O2_requirement

    # Calculation of oxygen transfer rate under standard conditions

    pressure = -(g * M * (elevation_a - 0)) / (R * (273.15 + T))

    relative_pressure = math_exp ** pressure

    C_20 = 1 + d_e * (Diffuser_depth / Pa)

    Oxygen_Con_20 = C20 * C_20

    O2_requirement_h = Supply_O / 24

    SOTR1 = O2_requirement_h / (0.50 * Foul_factor)

    SOTR2 = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

    SOTR3 = Oxygen_Con_20 / SOTR2

    SOTR4 = SOTR3 * (1.024) ** (20 - T)

    SOTR5 = SOTR1 * SOTR4                       #kg O2/ h

    SOTR6 = SOTR5 * 26.4                        #kg O2/ d

    EFF_BOD_O2 = SOTR6 / O2_required_per_BOD

    EFF_BOD_O2_isr = (EFF_BOD_O2 * 1000000) / (Q * 1000)

    #For oxygen requirement

    O2_requirement_h_isr = O2_requirement / 24

    SOTR1_isr = O2_requirement_h_isr / (0.50 * Foul_factor)

    SOTR2_isr = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

    SOTR3_isr = Oxygen_Con_20 / SOTR2_isr

    SOTR4_isr = SOTR3_isr * (1.024) ** (20 - T)

    SOTR5_isr = SOTR1_isr * SOTR4_isr                       #kg O2/ h

    SOTR6_isr = SOTR5_isr * 26.4                        #kg O2/ d

    Effluent_BOD_trials1 = (K * Hydrau_Reten * (SOTR6 / SOTR6_isr))

    K_isr = (IN_PO4 / (0.053 + IN_PO4)) * (theta_T ** (T - 20))

    Effluent_BOD_trials2 = 1 + (K_isr * Hydrau_Reten) * (SOTR6  / SOTR6_isr ) 

    Trial_new = Influent_BOD / Effluent_BOD_trials2

    Effluent_BOD_trials3 = Effluent_BOD_trials1 / Effluent_BOD_trials2

    Effluent_BOD_trials4 = 1 - Effluent_BOD_trials3

    Effluent_BOD_trials = Influent_BOD * Effluent_BOD_trials4

    BOD_Effluent =  (((SOTR6 / SOTR6_isr) -  Px_O2) / Q ) * (f * 1000)

    BOD5_Effluent = Influent_BOD - Effluent_BOD_trials

    BOD5_Effluentnp = np.percentile(Effluent_BOD5 , 50) 


    bio_growth = aerobic_biomass_yield * (Influent_BOD1 - Effluent_BOD5)






    # Imapct of HRT and aeration on the Effluent BOD5

    K0 = 2.5 * (1.047 ** (T - 20))

    DO1 = Supply_O / SOTR6 

    DO2 = 10.2 * DO1

    keff = (K0 * DO2) / ((DO2/2) + DO2) 



    Effluent_BOD512 = 1 / (1 + (keff * Hydrau_Reten )) * Influent_BOD1
    

    # STEP 11:  LBOD to SBOD-Conversion of SBOD6-120 to SBOD5

    Exponent_cal = (math_exp) ** -kL * t_n

    LBOD_to_SBOD = Initial_Concentration * Exponent_cal  # mg/L

    # STEP 12: Input SBOD- Calculate total reactant SBOD5

    SBOD_input = IN_SBOD + SBOD_fb  # mg/L


    # STEP 13: ΔSBOD- Calculate cell effluent SBOD5


    Effluent_SBOD_denominator = Hydrau_Reten * (Y * k - kd) - 1  # mg/L.d


    Effluent_SBOD = Ks * (1 + Hydrau_Reten * kd) / Effluent_SBOD_denominator

    SBOD_removal = IN_SBOD - Effluent_SBOD  # mg/L or g/m3

    SBOD_rem_g_d = SBOD_removal * Q  # g/d

    SBOD_Yield = VSS_per_day / SBOD_rem_g_d  # g VSS / sBOD

    SBOD_oxygen_used = (IN_SBOD * Q) - (Effluent_SBOD * Q)  # g O2/ g sBOD

    Oxygen_per_unit_SBOD = SBOD_oxygen_used / SBOD_rem_g_d

    # STEP 14- P Supply-Calculate available phosphorous from all supply sources

    # OPO4

    # Aerobic oxidation yield of 0.5 mg biomass per mg BOD utilized

    aerobic_biomass_yield = 0.5

    Biomass_Growth_isr = aerobic_biomass_yield * (Influent_BOD1 - Effluent_BOD5)  # mg/L

    # STEP 15: PΔSBOD- Calculation of SBOD5 supported by phosphorous


    PO4_E = kpo4 + IN_PO4

    SBOD_O_P = (PO4_P / PO4_E)

    SBOD_P = K20 * X_O * SBOD_O_P * Coeff_30_to_20  # mg/L

    # STEP 16: P-limitΔSBOD- select higher effluent SBOD5, if phosphorous limited

    PO4_E_limi = kpo4 + Limi_PO4_P

    SBOD_O_limi_P = (Limi_PO4_P / PO4_E_limi)

    SBOD_limi_P = K20 * X_O * SBOD_O_limi_P * Coeff_30_to_20


    SBOD_removal_no_P = SBOD_removal - SBOD_P


    SBOD_removal_limi_P = SBOD_removal_no_P + SBOD_limi_P

    Effluent_SBOD_P_limi = Input_SBOD - SBOD_removal_limi_P  # mg/L

    # STEP 17: ΔSBOD no P- calculate SBOD5 removal after phosphorus exhausted


    SBOD_removal_no_P = SBOD_removal - SBOD_P  # mg/L

    # STEP 18- AEROBIC ΔSBOD- Calculate overall aerobic cell effluent SBOD5

    SBOD_P_1 = IN_SBOD - Effluent_BOD5 

    Aerobic_SBOD = SBOD_removal - SBOD_P_1  # mg/L


    # STEP 19- Ox supply- Calculate available oxygen supply

    Power_req_sbod = Power - Power_req_bod

    DO_supply = Power_req_sbod * (Coeff_Power)  # kg/d




    # STEP 20- DOΔSBOD- Calculate SBOD5 removal supported by oxygen supply


    DO_SBOD_a = 1.3 + DO_supply  # kg/d

    DO_SBOD_b = DO_supply / DO_SBOD_a

    DO_SBOD = Oxygen_per_unit_SBOD * X_O * DO_SBOD_b  # mg/L

    Combine_SBOD_removal = DO_SBOD + SBOD_limi_P  # mg/L

    Intr_SBOD_removal = IN_SBOD - Combine_SBOD_removal  # mg/L

    Intr_SBOD_removal_123 = IN_SBOD - Effluent_BOD5

    # Step 21: Aerobic ΔSBOD- Select higher effluent aerobic SBOD5, if oxygen limited

    DO_supply_limi = Power_limi * Coeff_Power  # kg/d

    DO_b = 1.3 + DO_supply_limi

    DO_c = DO_supply_limi / DO_b


    DO_SBOD_limi = 0.25 * X_O * DO_c  # mg/L


    SBOD_rem_no_DO = IN_SBOD - DO_SBOD

    SBOD_removal_limi_DO = SBOD_rem_no_DO + DO_SBOD_limi

    Effluent_SBOD_DO_limi = IN_SBOD - SBOD_removal_limi_DO  # mg/L

    # STEP 22: ΔSBOD no P0-Recalculate SBOD5 removal after phosphorus exhausted

    SBOD_rem_no_P = SBOD_removal_limi_DO - SBOD_P  # mg/L

    # Step 23: Aerobic ΔSBOD Recalculate overall aerobic cell effluent SBOD5

    Overall_Aerobic_SBOD = IN_SBOD - SBOD_rem_no_P  # mg/L

    # Step 24: Anoxic ΔSBOD Calculate SBOD5 removal after oxygen exhausted


    DO_x = 1.3 + DO_supply_limi_x

    DO_no_O = DO_supply_limi_x / DO_x


    SBOD_no_O = 0.082 * X_O * DO_no_O

    Anoxic_SBOD_eff = Aerobic_SBOD - SBOD_no_O  # mg/L


    # Step 25:Total ΔSBOD- Calculate aerobic-plus-anoxic cell effluent SBOD5

    Total_SBOD = Overall_Aerobic_SBOD + Anoxic_SBOD_eff  # mg/L

    # Sep 26: Aerobic growth-Calculate new aerobic biomass growth

    Aerobic_growth = IN_SBOD - Overall_Aerobic_SBOD

    Aerobic_Biomass_growth = aerobic_biomass_yield * Aerobic_growth  # mg/L

    # Step 27: : Anoxic growth-Calculate new anoxic biomass growth

    Anoxic_growth = IN_SBOD - Anoxic_SBOD_eff  # mg/L

    Anoxic_Biomass_growth = 0.3 * Anoxic_growth  # mg/L

    # Step 28: Uptake N-Calculate nitrogen uptake by biomass growth

    Overall_Biomass_growth = Aerobic_Biomass_growth + Anoxic_Biomass_growth

    Uptake_N = N_Uptake_ratio / 115 * Overall_Biomass_growth  # mg/L

    # Step 29: Uptake P-Calculate phosphorus uptake by biomass growth

    Uptake_P = P_Uptake_ratio / 115 * Overall_Biomass_growth  # mg/L

    b_t = b * t_h

    a_b = a + b_t

    TSS_removal_percent = t_h / a_b

    TSS_removal = TSS_removal_percent / 100 * TSSi

    Effluent_TSS = TSSi - TSS_removal


    #Nitrification 

    spec_growth_AOB = max_growth_coefficient * (1.072 ** (T - 20))

    spec_endo_decay = b_20 * (1.029 ** (T - 20))

    growth_NH4 = spec_growth_AOB * (S_NH3 / (S_NH3 + K_NH4))

    spec_growth_NH4 = growth_NH4 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay

    SRT = ( 1 / spec_growth_NH4) * 1.5

    EFF_NH = spec_growth_AOB * (S_DO / (S_DO + K_AOB))

    EFF_NH1 = 0.50 * (1 + spec_endo_decay * SRT) / (SRT * (EFF_NH - spec_endo_decay) - 1) 
        
    S_NH = K_NH4 * (1 + (spec_endo_decay * SRT))

    growth_AOB_DO = (spec_growth_AOB * S_DO) / (S_DO + K_AOB)

    Effluent_NH4_N_numenator = (SRT * (growth_AOB_DO - spec_endo_decay)) - 1.0

    Effluent_NH4_N = (S_NH / Effluent_NH4_N_numenator)

    r_NH = (spec_growth_AOB / 0.15 ) * (IN_NH3 / ( IN_NH3 + K_NH4))

    r_NH1 = r_NH * (S_DO / (S_DO + K_AOB)) * 20

    NOX = r_NH1 * 1.1 

    X_AOB = Q * 0.15 * NOX * 1.1 

    X_AOB1 = X_AOB / (V * (1 + spec_endo_decay * SRT))


    BOD5_removal1 = Influent_BOD1 - Effluent_BOD5

    bh_t = bh * (1.04 ** (T - 20))

    Px_bio1 = ( Q * YH * BOD5_removal1 ) / 1000

    Px_bio2 = [1 + bh_t * 1.1 ]

    Px_bio3 = Px_bio1 / Px_bio2

    Px_bio4 = (fd_o * bh_t * Q * YH * BOD5_removal1 * 1.1) / 1000

    Px_bio5 = 1 + bh_t * 1.1

    Px_bio6 = Px_bio4 / Px_bio5

    Px_bioi = Px_bio3  + Px_bio6

    Px_NH4 = (Q * 0.15 * (NOX)) / 1000

    Px_NH41 = 1 + (0.315) * 1.1 

    Px_NH42 = Px_NH4 / Px_NH41

    Px_bio = (Px_bio3  + Px_bio6) / 0.80                            #kg/d      

    TSS_pro1 = Px_bioi + 3421.845

    TSS_pro1A = TSS_pro1 

    TSS_pro2A = (TSS_pro1 * 1000000 )/ (Q * 1000 ) + TSSi 

    Px_bio_mg =  (Px_bio * 1000000 )/ (Q * 1000 ) #mg/L

    try12 = Px_bio_mg + TSSi

    Settled_solids =  ((Benchmark_efficiency  / 100) * try12)

    Settling_solids = (Settled_solids / 100) * TSS_pro2A

    Eff_TSS =  try12 - Settled_solids

    Total_VSS_1A = Eff_TSS * 0.8

    settled_solids_isr = TSSi + bio_growth - Px_bio_mg

    BOD_fb = 0.3 * Settled_solids

    #PO4

    Px_mg = 0.015 * (Px_bio * 1000)   #g P/ d

    Px_mg1 = Px_mg / Q                  # g/m3

    Effluent_P = IN_PO4 - Px_mg1       #g/m3


    # Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass

    P_Uptake_ratio = 2.2

    Growth_P_Uptake_isr = (P_Uptake_ratio / 115) * Biomass_Growth_isr  # mgP/L

    PO4_P_fb_isr = Growth_P_Uptake_isr - (IN_PO4 + Effluent_P ) # mgP/L

    PO4_fb = (Settled_solids * 0.8 ) * (0.0049 / 2)

    EFF_PO4 = Effluent_P +  PO4_fb 


    # Step 30: Settled TSS-Calculate new plus inlet suspended solids that settle

    Overall_Settled_TSS = TSSi + Overall_Biomass_growth - Effluent_TSS  # mg/L


    # Calculation of Effluent NH4

    DO_mg = DO_supply * 1000000

    Volume_liter = Volume * 1000


    DO = DO_mg / Volume_liter  # mg/L

    Temp_Corr_factor = 2.718 ** (0.098 * (T - 15))

    pH_Corr_factor = 1 - 0.833 * (7.2 - pH)


    Max_growth = Max_speci_growth * Temp_Corr_factor * \
        DO / Ko2 + DO * pH_Corr_factor  # d-1

    k_o = Max_growth / Y_o  # d-1

    Min_Resi_time = 1 / Y_o * k_o - kd_o  # d

    Design_Resi_time = SF * (Min_Resi_time)  # d

    Substrate_Utilization = (1 / Design_Resi_time + kd_o) * 1 / Y_o  # d-1

    Kn = - 10 ** (0.051 * T - 1.158)  # mg/L

    N = 1 - k_o / Substrate_Utilization

    Effluent_NH4 = Kn / N  # mg/L

    Ammonia_N = IN_NH4 * 0.9441


    # Step 31: Benthal N-Calculate nitrogen feedback from settled biomass solids

    NH4_N_feedback = Uptake_N - IN_NH4 + Effluent_NH4  # mg/L

    # Step 32: Benthal P-Calculate phosphorus feedback from settled biomass solids

    PO4_P_feedback = Uptake_P - IN_PO4 + PO4_Pe  # mg/L

    # Step 33:  Nitrogen fixation- Calculate nitrogen fixation required to meet nitrogen demand

    Nitrogen_Demand = IN_NH4 - Effluent_NH4

    Nitrogen_Fixation = Nitrogen_Demand - Uptake_N  # mg/L

    # Step 34:  Feedback check- Compare SBOD5 feedback with starting estimate- step 10

    Overall_SBOD_fb = SBOD_feedback_ratio * Overall_Settled_TSS  # mg/L




    # Calculation of effleunt NH3



    # Define the values
    pKw = 14.0  # Example value at 25°C
    pKb = 4.75  # Example value for ammonia at 25°C

    # Assume pH is an array of pH values
    pH = 8 # Example array of pH values

    # Calculate the exponent for each pH value
    exponent = pKw - pKb - pH

    # Calculate 10^(exponent) for each pH value
    fraction1 = 10 ** exponent

    # Print the result

    fraction2 = 1 / (1 + fraction1)

    kN1 = 2.71828 ** (1.57 * (pH - 8.5))

    kN2 = 2.71828 ** (0.13 * (T - 20))

    kN = kN1 * kN2

    A = V / 3

    AQ = A / Q

    effluentN = AQ * kN * fraction2

    effluentN1 = 1 / (1 + effluentN)

    NH3 = IN_NH3 

    effluentN2 = IN_NH3 * effluentN1


    # Calculation of feedback

    biomass_growth = 0.5 * (Influent_BOD1 - Effluent_BOD5)

    biomass_growth_array = np.full_like(IN_NH3, biomass_growth)

    Uptake_N1 = (14 / 115 ) * biomass_growth

    NH4_N_feedback_1 = (Uptake_N1 - IN_NH3 + Effluent_NH4_N) 

    eff_nh3 = (Uptake_N1 - NH4_N_feedback_1) + IN_NH3

    eff_nh3_isr = (Uptake_N1 -NH4_N_feedback_1) + IN_NH3

    Removal_N = Uptake_N1 - NH4_N_feedback_1

    effluentN3 = effluentN2 + Removal_N

    final_nh3_fb = (Settled_solids * 0.8 ) * (0.0593 / np.where(T > 25, 1, 2))

    final_nh3 = Uptake_N1 - final_nh3_fb

    KB = 0.08 * (KBTHETA ** (T - 20))

    NH3_feedback_T = ((YBNSD * KB * 69605) / V) * 24 

    eff_fb_nh3 = NH3_feedback_T + Effluent_NH4_N


    TSS_BOD = 1 - np.exp(-0.10 * Hydrau_Reten)


    TSS_BOD1 = Eff_TSS * 0.5 * TSS_BOD

    # Step 35: ULTIMATE OXYGEN DEMAND

    Effluent_sbod_cal = IN_SBOD - Intr_SBOD_removal_123

    Soluble_BOD = Trial_new 

    TKN = (effluentN2 / 1.215) + Organic_nitrogen  # mg/L

    # TKN removal 

    f_pH = 2.71828 ** (0.2 * (pH - 7))

    AQ_k_fpH = AQ * K_t * f_pH

    TKN_removal = TKN / (1 + AQ_k_fpH)

    NBOD = 4.6 * TKN_removal  # mg/L


    CBOD5 = Soluble_BOD + TSS_BOD1

    NH3 = Effluent_NH4 * 0.9441  # mg/L


    Flow_rate_2 = Q * 0.000409  # ft3/s

    Flow_rate_3 = Flow_rate_2 * 0.646317


    UOD_2 = ((cBOD5_Multiplier * CBOD5) + 4.57 * final_nh3_fb) * Flow_rate_3 * 8.34


    # Aerated Stabilization Basin 2


    Px_1B = aeration_1B / Volume_1B

    Power_1B = Px_1B * Volume_1B

    K1 = 2.5 * (EFF_PO4 / (0.053 + EFF_PO4)) * (theta_T1_B ** (T_1B - 20))

    Effluent_BOD51 = CBOD5

    Effluent_BOD5_1B = 1 / (1 + (K1 * Hydrau_Reten)) * Effluent_BOD51    


    BOD5_removal_1B = CBOD5 - Effluent_BOD5_1B    #mg/


    x_numerator_1B = Y * (CBOD5 - Effluent_BOD5_1B)

    x_denomenator_1B = 1 + (kd * Hydrau_Reten)

    x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

    Px_1B = (x_1B * Q) / 1000  # kg/d

    Px_O2_1B = 1.42 * Px_1B

    O2_requirement_1Bisr = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d

    oxygen_supply1B = ((aeration_1B * 24) / 1000) * Coeff_Power

    BOD_Effluent_1B = ((oxygen_supply1B -  Px_O2_1B) / Q ) * (f * 1000)

    BOD5_Effluent_1B = CBOD5 - Effluent_BOD5_1B

    BOD5_Effluentnp_1B = np.percentile(BOD5_Effluent_1B, 50)


    x_numerator_1B = Y * (BOD5_removal - Effluent_BOD5_1B)

    x_denomenator_1B = 1 + (kd * Hydrau_Reten)

    x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

    Px_1B = (x_1B * Q) / 1000  # kg/d

    Px_O2_1B = 1.42 * Px_1B

    O2_requirement_1B = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d


    Power_req_bod_1B = O2_requirement_1B / (Coeff_Power)  # kW


    # Calculation of oxygen transfer rate under standard conditions

    pressure = -(g * M * (elevation_a - 0)) / (R * (273.15 + 12))

    relative_pressure = math_exp ** pressure

    C_20 = 1 + d_e * (Diffuser_depth / Pa)

    Oxygen_Con_20 = C20 * C_20

    O2_requirement_h1 = oxygen_supply1B / 24

    SOTR1_isr = O2_requirement_h1 / (0.50 * Foul_factor)

    SOTR2_isr = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

    SOTR3_isr = Oxygen_Con_20 / SOTR2_isr

    SOTR4_isr = SOTR3_isr * (1.024) ** (20 - T_1B)

    SOTR5_isr = SOTR1_isr * SOTR4_isr                       #kg O2/ h

    SOTR6_isr0 = SOTR5_isr * 26.4                        #kg O2/ d

    #For oxygen requirement

    O2_requirement_h_isr1 = O2_requirement_1Bisr / 24

    SOTR1_isr1 = O2_requirement_h_isr1 / (0.50 * Foul_factor)

    SOTR2_isr1 = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

    SOTR3_isr1 = Oxygen_Con_20 / SOTR2_isr1

    SOTR4_isr1 = SOTR3_isr1 * (1.024) ** (20 - T_1B)

    SOTR5_isr1 = SOTR1_isr1 * SOTR4_isr1                       #kg O2/ h

    SOTR6_isr1 = SOTR5_isr1 * 26.4                        #kg O2/ d

    K_isr_0 = (EFF_PO4 / (0.053 + EFF_PO4)) * (theta_T1_B ** (T_1B - 20))

    Effluent_BOD_trials2_isr = 1 + (K_isr_0 * Hydrau_Reten) * (SOTR6_isr0 / SOTR6_isr1) 

    Trial_new1 = CBOD5 / Effluent_BOD_trials2_isr

    # STEP 19- Ox supply- Calculate available oxygen supply

    Power_req_sbod_1B = Power_1B - Power_req_bod_1B

    DO_supply_1B = Power_req_sbod_1B * (Coeff_Power)  # kg/d


    # STEP 13: ΔSBOD- Calculate cell effluent SBOD5

    SBOD_removal_1B = Effluent_SBOD - SBOD_1B  # mg/L or g/m3

    SBOD_rem_g_d_1B = SBOD_removal_1B * Q  # g/d


    SBOD_Yield_1B = VSS_per_day / SBOD_rem_g_d_1B  # g VSS / sBOD


    SBOD_oxygen_used_1B = (Effluent_SBOD * Q) - \
        (SBOD_removal_1B * Q)  # g O2/ g sBOD

    Oxygen_per_unit_SBOD_1B = SBOD_oxygen_used_1B / SBOD_rem_g_d_1B

    SBOD_P_1B = Effluent_sbod_cal - Effluent_BOD5_1B

    Aerobic_SBOD_1b = SBOD_removal_1B - SBOD_P_1B

    # STEP 20- DOΔSBOD- Calculate SBOD5 removal supported by oxygen supply




    # BOD concentration at the specified time


    aerobic_biomass_yield = 0.5




    bh_t = bh * (1.04 ** (T_1B - 20))

    Px_bio1B = ( Q * YH * (CBOD5 - Effluent_BOD5_1B )) / 1000

    Px_bio2B = (1 + bh_t * Hydrau_Reten)

    Px_bio3B = Px_bio1B / Px_bio2B

    Px_bio4B = (fd_o * bh_t * Q * YH * (CBOD5 - Effluent_BOD5_1B ) * Hydrau_Reten) / 1000

    Px_bio5B = 1 + bh_t * Hydrau_Reten 

    Px_bio6B = Px_bio4B / Px_bio5B

    Px_bioB = Px_bio3B + Px_bio6B



    #Nitrification 

    spec_growth_AOB1 = max_growth_coefficient * (1.072 ** (T_1B - 20))

    spec_endo_decay1 = b_20 * (1.029 ** (T_1B - 20))

    growth_NH41 = spec_growth_AOB1 * (S_NH3  / (S_NH3  + K_NH4))

    spec_growth_NH41 = growth_NH41 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay1

    SRT1 = ( 1 / spec_growth_NH41) * 1.5 

    EFF_NH_ISR = spec_growth_AOB1 * (S_DO1 / (S_DO1 + K_AOB))

    EFF_NH1_ISR = eff_nh3_isr * (1 + spec_endo_decay1 * SRT1) / (SRT1 * (EFF_NH_ISR - spec_endo_decay1) - 1) 

    S_NH1 = K_NH4 * (1 + (spec_endo_decay * SRT1))

    growth_AOB_DO1 = spec_growth_AOB1 * ((S_DO) / (S_DO + K_AOB))

    Effluent_NH4_N_numenator1 = (SRT1 * (growth_AOB_DO1 - spec_endo_decay1)) - 1.0

    Effluent_NH4_N1 = (S_NH1 / Effluent_NH4_N_numenator1 )

    r_NH = (spec_growth_AOB / 0.15 ) * (eff_nh3 / (eff_nh3 + K_NH4))

    r_NH12 = r_NH * (S_DO / (S_DO + K_AOB)) * 30

    NOX1 = r_NH12 * 1.1 

    X_AOB = Q * 0.15 * NOX * SRT 

    X_AOB1 = X_AOB / (V * (1 + spec_endo_decay * SRT))

    Px_NH4 = (Q * 0.15 * (NOX)) / 1000

    Px_NH41 = 1 + (0.315) * Hydrau_Reten 

    Px_NH42 = Px_NH4 / Px_NH41

    Px_bio_isr = (Px_bio3B  + Px_bio6B ) / 0.80

    Px_bio_mg1 =  (Px_bio_isr * 1000000 )/ (Q * 1000 )     #mg/L

    TSS_pro = Px_bio_isr + ((Q * (Eff_TSS - Total_VSS_1A)) / 1000)

    TSS_pro1isr = TSS_pro 

    TSS_pro2 = (TSS_pro1isr * 1000000 )/ (Q * 1000 ) + Eff_TSS

    Benchmark_efficiency1 = 84 - (10.6 * (aeration_1B / Volume_1B))

    TRY1 = Px_bio_mg1 + Total_VSS_1A

    Settled_solids1B =  (( TRY1 / 100) * Benchmark_efficiency1)

    Settling_solids_1B = (Settled_solids1B / 100) * TSS_pro2

    Eff_TSS_1B =  TRY1 - Settled_solids1B


    # Calculation of feedback1

    YBNSD1 = (0.0593 * (Settled_solids1B * 0.8 ) * 2.6 ) / 26.4

    KB = 0.08 * KBTHETA ** (T_1B - 20)

    NH3_feedback_T_ISR = ((YBNSD1 * KB * 58679.42) / Volume_1B) * 24

    eff_fb_nh31 = Effluent_NH4_N1 + NH3_feedback_T_ISR 

    biomass_growth1 = 0.5 * (CBOD5 - Effluent_BOD5_1B )

    Settled_solids1 = Eff_TSS + biomass_growth1 - (TSS_pro1 * 1000000 )/ (Q * 1000 )

    BOD_fb1 = 0.3 * Settled_solids1B

    biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

    Uptake_N12_isr = (14 / 115 ) * biomass_growth1

    NH4_N_feedback_ISR = (Uptake_N12_isr - eff_nh3 + Effluent_NH4_N1) / 1

    eff_nh31 = (Uptake_N12_isr - NH4_N_feedback_ISR) + eff_nh3

    effluent_nh3_con = NH4_N_feedback_ISR + Effluent_NH4_N1

    final_nh3_fb_isr = (Settled_solids1B * 0.8 ) * (0.0593 /  np.where(T > 25, 1, 2))

    final_nh3_isr = final_nh3_fb_isr + Effluent_NH4_N1

    # Define the values
    pKw = 14.0  # Example value at 25°C
    pKb = 4.75  # Example value for ammonia at 25°C

    # Assume pH is an array of pH values
    pH1 = 8 # Example array of pH values

    # Calculate the exponent for each pH value
    exponent1 = pKw - pKb - pH1 

    # Calculate 10^(exponent) for each pH value
    fraction1 = 10 ** exponent1

    # Print the result


    fraction2_1 = 1 / (1 + fraction1)

    kN1_1 = 2.71828 ** (1.57 * (pH1 - 8.5))

    kN2_1 = 2.71828 ** (0.13 * (T - 20))

    kN_1 = kN1_1 * kN2_1

    A = V / 4

    AQ = A / Q

    effluentN1_1 = AQ * kN_1 * fraction2_1

    effluentN_12 = 1 / (1 + effluentN1_1)

    NH3_i = effluentN2 + Removal_N

    effluentN2_1 =  NH3_i  * effluentN_12


    # Calculation of feedback1





    biomass_growth1 = 0.5 * ( CBOD5 - Effluent_BOD5_1B )

    biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

    Uptake_N12 = (14 / 115 ) * biomass_growth1

    NH4_N_feedback_12 = Uptake_N12 - (effluentN2 + effluentN2_1) 

    effluent_NH3 = effluentN2_1 + NH4_N_feedback_12 

    # Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass

    #PO4

    Px_mg_isr = 0.015 * (Px_bio_isr * 1000)   #g P/ d

    Px_mg1_isr = Px_mg_isr / Q                  # g/m3

    Effluent_P_isr = EFF_PO4 - Px_mg1_isr       #g/m3

    PO4_fb_isr = (Settled_solids1B * 0.8 )  * (0.0049 / 1)

    EFF_PO4_isr = Effluent_P_isr +  PO4_fb_isr 

    P_Uptake_ratio = 2.2

    Growth_P_Uptake1 = (P_Uptake_ratio / 115) * biomass_growth1 # mgP/L

    PO4_P_fb1 = Growth_P_Uptake - EFF_PO4 + Effluent_P_isr # mgP/L





    TSS_BODisr = 1 - np.exp(-0.10 * Hydrau_Reten)



    TSS_BOD1isr = Eff_TSS_1B * 0.5 * TSS_BODisr


    Soluble_BOD1 =  Trial_new1 

    Ammonia_N1 = IN_NH4 * 0.9441

    TKN1 = (effluentN2_1 / 1.215) + Organic_nitrogen1  # mg/L

    # TKN removal 

    f_pH1 = 2.71828 ** (0.2 * (pH1 - 7))

    AQ_k_fpH1 = AQ * K_t * f_pH1

    TKN_removal1 = TKN_removal / (1 + AQ_k_fpH1)

    NBOD1 = 4.6 * TKN_removal1



    CBOD51 = Soluble_BOD1  + TSS_BOD1isr

    UOD_3 = ((cBOD5_Multiplier * CBOD51) + 4.57 * final_nh3_fb_isr) * Flow_rate_3 * 8.34

    if isPond1A:

        return UOD_2
    
    else:
        return UOD_3


def run_model(data,special_simulation_values):

    t0 = data['t0']
    t = data['t']
    T = data['T']
    max_decay = data['max_decay']
    tot = data['tot']
    k_s = data['k_s']
    math_expo = data['math_expo']
    IN_NH4 = data['IN_NH4']
   
    IN_SBOD = data['IN_SBOD']
    split_COD = data['split_COD']
    IN_Q = data['IN_Q']
    solids_percent = data['solids_percent']
    specific_gravity = data['specific_gravity']
    TS_retention = data['TS_retention']
    IN_COD = data['IN_COD']
    final_solids_content = data['final_solids_content']
    Sludge_cake_solids = data['Sludge_cake_solids']
    liq_COD_IN = data['liq_COD_IN']
    dewatering_polymer_dose = data['dewatering_polymer_dose']
    Filtrate_flowrate = data['Filtrate_flowrate']
    Filtrate_Solids = data['Filtrate_Solids']
    SP_cake_COD = data['SP_cake_COD']
    theta = data['theta']
    K_20 = data['K_20']
    V = data['V']
    X_ot = data['X_ot']
    kd = data['kd']
    Y = data['Y']
    X = data['X']
    fd = data['fd']
    Oxygen_factor = data['Oxygen_factor']
    TSSe = data['TSSe']
    Growth = data['Growth']
    Power = data['Power_x']
    Volume = data['Volume']
    CM_Power = data['CM_Power']
    CM_Volume = data['CM_Volume']
    n = data['n']
    tn = data['tn']
    Initial_Concentration = data['Initial_Concentration']
    kL = data['kL']
    t_n = data['t_n']
    math_exp = data['math_exp']
    Input_SBOD = data['Input_SBOD']
    SBODe = data['SBODe']
    Hydrau_Reten = data['Hydrau_Reten']
    Ks = data['Ks']
    k = data['k']
    PO4_P = data['PO4_P']
    K20 = data['K20']
    X_O = data['X_O']
    kpo4 = data['kpo4']
    Coeff_30_to_20 = data['Coeff_30_to_20']
    Limi_PO4_P = data['Limi_PO4_P']
    Coeff_Power = data['Coeff_Power']
    Power_limi = data['Power_limi']
    NH4_Ni = data['NH4_Ni']
    NH4_Ne = data['NH4_Ne']
    PO4_Pe = data['PO4_Pe']
    cBOD5_Multiplier = data['cBOD5_Multiplier']
    f_c = data['f_c']

    # most recent additions
    Max_speci_growth = data['Max_speci_growth']
    Ko2 = data['Ko2']
    pH = data['pH']
    Y_o = data['Y_o']
    kd_o = data['kd_o']
    SF = data['SF']
    a = data['a']
    b = data['b']
    t_h = data['t_h']
   
    IN_NH3_z = data['IN_NH3_z']

    K = data['K']
    
    TSSi = data['TSSi']
    aeration = data['aeration']
    DO_supply_limi_x = data['DO_supply_limi_x']
    Influent_BOD = data['Influent_BOD']
    Effluent_BOD = data['Effluent_BOD']
    aeration_1B = data['aeration_1B']
    Effluent_BOD_1B  = data['Effluent_BOD_1B']
    Organic_nitrogen  = data['Organic_nitrogen']
    SBOD_1B = data['SBOD_1B']
    IN_NH41 = data['IN_NH41']
    Organic_nitrogen1  = data['Organic_nitrogen1']
    f = data['f']

    max_growth_coefficient = data['max_growth_coefficient']
    Volume_1B = data['Volume_1B']

    Sup_NH3 = data['Sup_NH3']

    b_20  = data['b_20']
    K_NH4 = data['K_NH4']
    S_DO = data['S_DO']
    K_AOB = data['K_AOB']
    bh = data['bh']
    YH = data['YH']
    fd_o = data['fd_o']

    IN_PO4_z = data['IN_PO4_z']
    Sup_PO4 = data['Sup_PO4']
    
    parameters_df = pd.read_excel('assumptions_AerationBasin.xlsx', sheet_name='Design')
        # %% Aerated Stabilization Basin

    # process models for ASB


    # Flow in Aerated Stablization Basin is set to the Flow volume from the sedimentation Tank


    Q = IN_Q * 3785.4118

    Influent_BOD1 = Influent_BOD


    Benchmark_efficiency = 15

    VSS = TSSi * 0.80

    IN_PO4 = IN_PO4_z + Sup_PO4

    IN_NH3 = IN_NH3_z + Sup_NH3

    # STEP 1: TEMP SENSITIVITY- Calculate Arrhenius temperature sensitivity coefficient

    # STEP 2: k.VSS- Adjust oxidation rate to cell temperature


    K_t = K_20 * theta ** (T - 20)

    Influent_nbVSS = Q * X_ot / V  # g/m3.d

    Ks_SBODe = Ks + IN_SBOD  # mg/L

    rsu = - k * X * IN_SBOD / Ks_SBODe  # g/m3.d

    r_Xt_vss = -Y * (rsu) - kd * X + fd * kd * X + Influent_nbVSS  # g/m3.d

    VSS_per_day = r_Xt_vss * Q  # g/d

    oxidation_rate = r_Xt_vss * Oxygen_factor  # g/m3.d

    Adj_Oxidation_rate = oxidation_rate * K_t  # g/m3.d

    # STEP 3: Digest Factor-Temperature-adjust benthal feedback (not Arrhenius)

    # Aerobic oxidation yield of 0.5 mg biomass per mg BOD utilized

    aerobic_biomass_yield = 0.5

    Biomass_Growth = aerobic_biomass_yield * (Influent_BOD1 - Effluent_BOD)  # mg/L




    P_Uptake_ratio = 2.2

    Growth_P_Uptake = (P_Uptake_ratio / 115) * Biomass_Growth  # mgP/L

    PO4_P_fb = Growth_P_Uptake - IN_PO4 + PO4_Pe  # mgP/L

    TSS_solu_P = (PO4_P_fb * 115) / P_Uptake_ratio  # mg/L

    Settled_solids = TSSi + Growth - TSSe  # mg/L

    Digestion_eff_P = TSS_solu_P / Settled_solids * 100  # percentage

    # Stoichiometric ratio between the uptake of nitrogen (N) and the growth of microbial biomass

    N_Uptake_ratio = 14

    Growth_N_Uptake = (N_Uptake_ratio / 115) * Biomass_Growth  # mgN/L

    NH4_N_fb = Growth_N_Uptake - IN_NH3 + NH4_Ne  # mgN/L

    TSS_solu_N = (NH4_N_fb * 115) / N_Uptake_ratio  # mg/L

    Digestion_eff_N = (TSS_solu_N / Settled_solids) * 100  # percentage


    # Calculation of Power


    P_x = aeration / Volume  # W/m3

    Power = P_x * Volume  # W

    # STEP 4: MIXING- Calculate Mixing Intenstiy

    Mixing_Intensity = Power / Volume  # W/m3

    # STEP 5: SETTLING- Calculate percent of suspended solids that settle

    TSS_sett = Settled_solids / (TSSi + Growth) * 100  # percentage

    # STEP 6: PARTIAL MIX Calculate ration of cell mixing intensity to complete mix

    Complete_Mixing = CM_Power / CM_Volume  # W/m3

    Partial_Mix = Mixing_Intensity / Complete_Mixing  # fraction

    # STEP 7: K1.VSS- Adjust baseline oxidation rate to cell partial-mix level

    Adj_OR_Partial_mix = Partial_Mix * Adj_Oxidation_rate  # g/m3.d


    # STEP 8: Cells in series- Select number of complete-mix cells to represent hydraulics

    n = 1


    # STEP 9: Calculate denominator of first-order rate equation.

    #Rate = {1 + K_t * tn} ** n
    Rate = (1 + K_t * tn) ** n

    # STEP 10: Estimate benthal feedback of SBOD5

    Sett_TSS = TSSi + Growth - TSSe


    # Estimated soluble BOD feedback per mg of TSS settled

    SBOD_feedback_ratio = 0.3

    SBOD_fb = SBOD_feedback_ratio * Sett_TSS  # mg/L


    # Calculate the BOD5 removed

    # Calculate the BOD5 removed

    K = 2.5 * (1.056 ** (T - 20))

    Effluent_BOD5 = 1 / (1 + (K * Hydrau_Reten)) * Influent_BOD1

    BOD5_removal = Influent_BOD1 - Effluent_BOD5  # mg/L

    x_numerator = Y * (Influent_BOD1 - Effluent_BOD5)

    x_denomenator = 1 + (kd * Hydrau_Reten)

    x = x_numerator / x_denomenator  # mg/L

    Px = (x * Q) / 1000  # kg/d

    Px_O2 = 1.42 * Px

    O2_requirement = Q * (BOD5_removal / (f * 1000))-(Px_O2)  # kg/d

    oxygen_supply1 = aeration * Coeff_Power

    BOD_Effluent = ((oxygen_supply1 -  Px_O2) / Q ) * (f * 1000)

    BOD5_Effluent = Influent_BOD - BOD_Effluent

    BOD5_Effluentnp = np.percentile(Effluent_BOD5, 50)

    Power_req_bod = O2_requirement / (Coeff_Power)  # kW


    # STEP 11:  LBOD to SBOD-Conversion of SBOD6-120 to SBOD5

    Exponent_cal = (math_exp) ** -kL * t_n

    LBOD_to_SBOD = Initial_Concentration * Exponent_cal  # mg/L

    # STEP 12: Input SBOD- Calculate total reactant SBOD5

    SBOD_input = IN_SBOD + SBOD_fb  # mg/L


    # STEP 13: ΔSBOD- Calculate cell effluent SBOD5


    Effluent_SBOD_denominator = Hydrau_Reten * (Y * k - kd) - 1  # mg/L.d


    Effluent_SBOD = Ks * (1 + Hydrau_Reten * kd) / Effluent_SBOD_denominator

    SBOD_removal = IN_SBOD - Effluent_SBOD  # mg/L or g/m3

    SBOD_rem_g_d = SBOD_removal * Q  # g/d


    SBOD_Yield = VSS_per_day / SBOD_rem_g_d  # g VSS / sBOD


    SBOD_oxygen_used = (IN_SBOD * Q) - (Effluent_SBOD * Q)  # g O2/ g sBOD

    Oxygen_per_unit_SBOD = SBOD_oxygen_used / SBOD_rem_g_d

    # STEP 14- P Supply-Calculate available phosphorous from all supply sources


    

    # Extract parameter values
    parameter_values = parameters_df.set_index(
        'Parameter').loc[:, 'expected'].to_dict()

    # Extract specific parameters needed for the simulation
    IN_SBOD = parameter_values['IN_SBOD']
    SBODe = parameter_values['SBODe']
    IN_PO4_z = parameter_values['IN_PO4_z']
    Sup_PO4 = parameter_values['Sup_PO4']
    PO4_Pe = parameter_values['PO4_Pe']
    TSSi = parameter_values['TSSi']
    TSSe = parameter_values['TSSe']
    Growth = parameter_values['Growth']
    T = parameter_values['T']

    # Define the function representing the differential equation


    def dSBOD5_dt(SBOD5, t, k20, OPO4_interp, Kopo4, T):
        OPO4 = OPO4_interp(t)
        return -k20 * (OPO4 / (Kopo4 + OPO4)) * (1.05 ** (T - 20)) * SBOD5


    IN_PO4 = IN_PO4_z + Sup_PO4

    Sett_TSS = TSSi + Growth - TSSe

    SBOD_feedback_ratio = 0.3

    SBOD_fb = SBOD_feedback_ratio * Sett_TSS  # mg/L

    SBOD_input = BOD5_Effluentnp + SBOD_fb


    # Set initial conditions and parameters
    initial_SBOD5 = BOD5_Effluentnp
    k20 = 0.02
    Kopo4 = 0.05

    # Function to calculate OPO4 at each time step


    def calculate_OPO4(t, IN_SBOD, SBODe, IN_PO4, PO4_Pe):
        aerobic_biomass_yield = 0.5
        Biomass_Growth = aerobic_biomass_yield * (IN_SBOD - SBODe)
        P_Uptake_ratio = 2.2
        Growth_P_Uptake = (P_Uptake_ratio / 115) * Biomass_Growth
        PO4_P_fb = Growth_P_Uptake - IN_PO4 + PO4_Pe
        return IN_PO4 + PO4_P_fb


    #  Time points for integration
    # Adjust the time range and number of points as needed
    time_points = np.linspace(0, 10, 100)

    # Evaluate calculate_OPO4 at each time point
    OPO4_values = [calculate_OPO4(t, IN_SBOD, SBODe, IN_PO4, PO4_Pe)
                for t in time_points]

    # Create an interpolation function for OPO4
    OPO4_interp = interp1d(time_points, OPO4_values,
                        kind='linear', fill_value='extrapolate')

    # Solve the differential equation using odeint
    result = odeint(dSBOD5_dt, initial_SBOD5, time_points,
                    args=(k20, OPO4_interp, Kopo4, T))


    # To get the value of SBOD5 at time t = 5 (for example)
    desired_time = 26.4

    # Find the index in the time_points array that is closest to the desired_time
    index_at_desired_time = np.abs(time_points - desired_time).argmin()

    # Get the corresponding value of SBOD5 from the result array
    sbod5_P_at_desired_time = result[index_at_desired_time]

    # Calculate OPO4 at the desired time
    opo4_at_desired_time = calculate_OPO4(desired_time, IN_SBOD, SBODe, IN_PO4, PO4_Pe)

    # Calculate the remaining amount of OPO4
    remaining_opo4 = IN_PO4 - opo4_at_desired_time   



    def calculate_OPO4_remaining(SBOD5_at_t, initial_SBOD5, IN_PO4, PO4_Pe):
        aerobic_biomass_yield = 0.5
        Biomass_Growth = aerobic_biomass_yield * (initial_SBOD5 - SBOD5_at_t)
        P_Uptake_ratio = 2.2
        Growth_P_Uptake = (P_Uptake_ratio / 115) * Biomass_Growth
        PO4_P_fb = Growth_P_Uptake - IN_PO4 + PO4_Pe
        return IN_PO4 - Growth_P_Uptake

    # Calculate SBOD5 and OPO4 at the desired time
    sbod5_at_desired_time = result[index_at_desired_time][0]
    opo4_remaining_at_desired_time = calculate_OPO4_remaining(sbod5_at_desired_time, initial_SBOD5, IN_PO4, PO4_Pe)




    # OPO4



    # Aerobic oxidation yield of 0.5 mg biomass per mg BOD utilized

    aerobic_biomass_yield = 0.5

    Biomass_Growth_isr = aerobic_biomass_yield * (Influent_BOD1 - Effluent_BOD5)  # mg/L

    # Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass

    P_Uptake_ratio = 2.2

    Growth_P_Uptake_isr = (P_Uptake_ratio / 115) * Biomass_Growth_isr  # mgP/L

    PO4_P_fb_isr = Growth_P_Uptake_isr - (IN_PO4 + remaining_opo4) # mgP/L



    # STEP 15: PΔSBOD- Calculation of SBOD5 supported by phosphorous


    PO4_E = kpo4 + IN_PO4

    SBOD_O_P = (PO4_P / PO4_E)

    SBOD_P = K20 * X_O * SBOD_O_P * Coeff_30_to_20  # mg/L

    # STEP 16: P-limitΔSBOD- select higher effluent SBOD5, if phosphorous limited

    PO4_E_limi = kpo4 + Limi_PO4_P

    SBOD_O_limi_P = (Limi_PO4_P / PO4_E_limi)

    SBOD_limi_P = K20 * X_O * SBOD_O_limi_P * Coeff_30_to_20


    SBOD_removal_no_P = SBOD_removal - SBOD_P


    SBOD_removal_limi_P = SBOD_removal_no_P + SBOD_limi_P

    Effluent_SBOD_P_limi = Input_SBOD - SBOD_removal_limi_P  # mg/L

    # STEP 17: ΔSBOD no P- calculate SBOD5 removal after phosphorus exhausted


    SBOD_removal_no_P = SBOD_removal - SBOD_P  # mg/L

    # STEP 18- AEROBIC ΔSBOD- Calculate overall aerobic cell effluent SBOD5

    SBOD_P_1 = IN_SBOD - sbod5_P_at_desired_time

    Aerobic_SBOD = SBOD_removal - SBOD_P_1  # mg/L


    # STEP 19- Ox supply- Calculate available oxygen supply

    Power_req_sbod = Power - Power_req_bod

    DO_supply = Power_req_sbod * (Coeff_Power)  # kg/d


    # Define the system of ODEs

    def model(y, t):
        S, X, DO = y
        dSdt = -(Us * S / (Ks + S)) * D0 / (Ko + DO)
        dXdt = a * dSdt - d * X
        dDOdt = -a_prime * dSdt - d_prime * X + KLa * (DOs - DO)
        return [dSdt, dXdt, dDOdt]


    # Set parameter values
    Us = 0.1
    Ks = 100
    D0 = 29310
    Ko = 0.1
    a = 0.70
    d = 0.002
    a_prime = 0.34
    d_prime = 0.0008
    KLa = 2
    DOs = 6

    

    # Extract parameter values
    parameter_values = parameters_df.set_index(
        'Parameter').loc[:, 'expected'].to_dict()

    # Extract specific parameters needed for the simulation
    Coeff_Power = parameter_values['Coeff_Power']
    aeration = parameter_values['aeration']
    Volume = parameter_values['Volume']
    Influent_BOD12 = parameter_values['Influent_BOD']
    Effluent_BOD = parameter_values['Effluent_BOD']
    Y = parameter_values['Y']
    kd = parameter_values['kd']
    Hydrau_Reten = parameter_values['Hydrau_Reten']
    f = parameter_values['f']


    P_x = aeration / Volume  # W/m3

    Power = P_x * Volume


    BOD5_removal = Influent_BOD12 - Effluent_BOD5 # mg/L

    x_numerator = Y * (Influent_BOD12 - Effluent_BOD5)

    x_denomenator = 1 + (kd * Hydrau_Reten)

    x = x_numerator / x_denomenator  # mg/L

    Px = (x * Q) / 1000  # kg/d

    Px_O2 = 1.42 * Px

    O2_requirement = Q * (BOD5_removal / (f * 1000))-(Px_O2)  # kg/d


    Power_req_bod = O2_requirement / (Coeff_Power)

    Power_req_sbod = Power - Power_req_bod

    DO_supply = Power_req_sbod * (Coeff_Power)

    DO_supply = DO_supply[0]

    # Initial conditions
    inital_SBOD = sbod5_P_at_desired_time

    initial_conditions = [sbod5_P_at_desired_time[0], 60, DO_supply[0]]

    # Time points for integration
    t = np.linspace(0, 20, 100)  # Replace with your time range

    # Solve the differential equations using odeint
    solution = odeint(model, initial_conditions, t)



    # Solve the differential equations using odeint
    solution = odeint(model, initial_conditions, t)

    # Extract BOD concentration at a specific time
    specific_time = 26.4  # Replace with the desired time
    # Find the index closest to the specified time
    index = np.abs(t - specific_time).argmin()

    # BOD concentration at the specified time
    sbod_concentration_at_specific_time = solution[index, 0]
    DO_concentration_at_specific_time = solution[index, 1]


    # STEP 20- DOΔSBOD- Calculate SBOD5 removal supported by oxygen supply


    DO_SBOD_a = 1.3 + DO_supply  # kg/d

    DO_SBOD_b = DO_supply / DO_SBOD_a

    DO_SBOD = Oxygen_per_unit_SBOD * X_O * DO_SBOD_b  # mg/L

    Combine_SBOD_removal = DO_SBOD + SBOD_limi_P  # mg/L

    Intr_SBOD_removal = IN_SBOD - Combine_SBOD_removal  # mg/L

    Intr_SBOD_removal_123 = IN_SBOD - sbod_concentration_at_specific_time

    # Step 21: Aerobic ΔSBOD- Select higher effluent aerobic SBOD5, if oxygen limited

    DO_supply_limi = Power_limi * Coeff_Power  # kg/d

    DO_b = 1.3 + DO_supply_limi

    DO_c = DO_supply_limi / DO_b


    DO_SBOD_limi = 0.25 * X_O * DO_c  # mg/L


    SBOD_rem_no_DO = IN_SBOD - DO_SBOD

    SBOD_removal_limi_DO = SBOD_rem_no_DO + DO_SBOD_limi

    Effluent_SBOD_DO_limi = IN_SBOD - SBOD_removal_limi_DO  # mg/L

    # STEP 22: ΔSBOD no P0-Recalculate SBOD5 removal after phosphorus exhausted

    SBOD_rem_no_P = SBOD_removal_limi_DO - SBOD_P  # mg/L

    # Step 23: Aerobic ΔSBOD Recalculate overall aerobic cell effluent SBOD5

    Overall_Aerobic_SBOD = IN_SBOD - SBOD_rem_no_P  # mg/L

    # Step 24: Anoxic ΔSBOD Calculate SBOD5 removal after oxygen exhausted


    DO_x = 1.3 + DO_supply_limi_x

    DO_no_O = DO_supply_limi_x / DO_x


    SBOD_no_O = 0.082 * X_O * DO_no_O

    Anoxic_SBOD_eff = Aerobic_SBOD - SBOD_no_O  # mg/L


    # Step 25:Total ΔSBOD- Calculate aerobic-plus-anoxic cell effluent SBOD5

    Total_SBOD = Overall_Aerobic_SBOD + Anoxic_SBOD_eff  # mg/L

    # Sep 26: Aerobic growth-Calculate new aerobic biomass growth

    Aerobic_growth = IN_SBOD - Overall_Aerobic_SBOD

    Aerobic_Biomass_growth = aerobic_biomass_yield * Aerobic_growth  # mg/L

    # Step 27: : Anoxic growth-Calculate new anoxic biomass growth

    Anoxic_growth = IN_SBOD - Anoxic_SBOD_eff  # mg/L

    Anoxic_Biomass_growth = 0.3 * Anoxic_growth  # mg/L

    # Step 28: Uptake N-Calculate nitrogen uptake by biomass growth

    Overall_Biomass_growth = Aerobic_Biomass_growth + Anoxic_Biomass_growth

    Uptake_N = N_Uptake_ratio / 115 * Overall_Biomass_growth  # mg/L

    # Step 29: Uptake P-Calculate phosphorus uptake by biomass growth

    Uptake_P = P_Uptake_ratio / 115 * Overall_Biomass_growth  # mg/L

    b_t = b * t_h

    a_b = a + b_t

    TSS_removal_percent = t_h / a_b

    TSS_removal = TSS_removal_percent / 100 * TSSi

    Effluent_TSS = TSSi - TSS_removal


    #Nitrification 

    spec_growth_AOB = max_growth_coefficient * (1.072 ** (T - 20))

    spec_endo_decay = b_20 * (1.029 ** (T - 20))

    growth_NH4 = spec_growth_AOB * (IN_NH3 / ( IN_NH3 + K_NH4))

    spec_growth_NH4 = growth_NH4 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay

    SRT = ( 1 / spec_growth_NH4) * 1.5
        
    S_NH = K_NH4 * (1 + (spec_endo_decay * SRT))

    growth_AOB_DO = (spec_growth_AOB * S_DO) / (S_DO + K_AOB)

    Effluent_NH4_N_numenator = (SRT * (growth_AOB_DO - spec_endo_decay)) - 1.0

    Effluent_NH4_N = (S_NH / Effluent_NH4_N_numenator) * (Hydrau_Reten/ SRT)

    r_NH = (spec_growth_AOB / 0.15 ) * (IN_NH3 / ( IN_NH3 + K_NH4))

    r_NH1 = r_NH * (S_DO / (S_DO + K_AOB)) * 20

    NOX = r_NH1 * 1.1 

    X_AOB = Q * 0.15 * NOX * 1.1 

    X_AOB1 = X_AOB / (V * (1 + spec_endo_decay * SRT))


    BOD5_removal1 = Influent_BOD1 - sbod5_P_at_desired_time

    bh_t = bh * (1.04 ** (T - 20))

    Px_bio1 = ( Q * YH * BOD5_removal1 ) / 1000

    Px_bio2 = [1 + bh_t * 1.1 ]

    Px_bio3 = Px_bio1 / Px_bio2

    Px_bio4 = (fd_o * bh_t * Q * YH * BOD5_removal1 * SRT) / 1000

    Px_bio5 = 1 + bh_t * 1.1

    Px_bio6 = Px_bio4 / Px_bio5

    Px_bioi = Px_bio3  + Px_bio6

    Px_NH4 = (Q * 0.15 * (NOX)) / 1000

    Px_NH41 = 1 + (0.315) * 1.1 

    Px_NH42 = Px_NH4 / Px_NH41

    Px_bio = Px_bio3  + Px_bio6                            #kg/d      

    TSS_pro1 = Px_bio + ((Q * (TSSi - VSS)) / 1000)

    TSS_pro1A = TSS_pro1 

    TSS_pro2A = (TSS_pro1 * 1000000 )/ (Q * 1000 ) + TSSi 



    Px_bio_mg =  (Px_bio * 1000000 )/ (Q * 1000 ) #mg/L


    Settled_solids =  ((TSS_pro2A / 100) * Benchmark_efficiency)

    Eff_TSS =  TSS_pro2A - Settled_solids

    Total_VSS_1A = Eff_TSS * 0.8


    # Step 30: Settled TSS-Calculate new plus inlet suspended solids that settle

    Overall_Settled_TSS = TSSi + Overall_Biomass_growth - Effluent_TSS  # mg/L


    # Calculation of Effluent NH4

    DO_mg = DO_supply * 1000000

    Volume_liter = Volume * 1000


    DO = DO_mg / Volume_liter  # mg/L

    Temp_Corr_factor = 2.718 ** (0.098 * (T - 15))

    pH_Corr_factor = 1 - 0.833 * (7.2 - pH)


    Max_growth = Max_speci_growth * Temp_Corr_factor * \
        DO / Ko2 + DO * pH_Corr_factor  # d-1

    k_o = Max_growth / Y_o  # d-1

    Min_Resi_time = 1 / Y_o * k_o - kd_o  # d

    Design_Resi_time = SF * (Min_Resi_time)  # d

    Substrate_Utilization = (1 / Design_Resi_time + kd_o) * 1 / Y_o  # d-1

    Kn = - 10 ** (0.051 * T - 1.158)  # mg/L

    N = 1 - k_o / Substrate_Utilization

    Effluent_NH4 = Kn / N  # mg/L

    Ammonia_N = IN_NH4 * 0.9441


    # Step 31: Benthal N-Calculate nitrogen feedback from settled biomass solids

    NH4_N_feedback = Uptake_N - IN_NH4 + Effluent_NH4  # mg/L

    # Step 32: Benthal P-Calculate phosphorus feedback from settled biomass solids

    PO4_P_feedback = Uptake_P - IN_PO4 + PO4_Pe  # mg/L

    # Step 33:  Nitrogen fixation- Calculate nitrogen fixation required to meet nitrogen demand

    Nitrogen_Demand = IN_NH4 - Effluent_NH4

    Nitrogen_Fixation = Nitrogen_Demand - Uptake_N  # mg/L

    # Step 34:  Feedback check- Compare SBOD5 feedback with starting estimate- step 10

    Overall_SBOD_fb = SBOD_feedback_ratio * Overall_Settled_TSS  # mg/L




    # Calculation of effleunt NH3



    # Define the values
    pKw = 14.0  # Example value at 25°C
    pKb = 4.75  # Example value for ammonia at 25°C

    # Assume pH is an array of pH values
    pH = 8 # Example array of pH values

    # Calculate the exponent for each pH value
    exponent = pKw - pKb - pH

    # Calculate 10^(exponent) for each pH value
    fraction1 = 10 ** exponent



    fraction2 = 1 / (1 + fraction1)

    kN1 = 2.71828 ** (1.57 * (pH - 8.5))

    kN2 = 2.71828 ** (0.13 * (T - 20))

    kN = kN1 * kN2

    A = V / 3

    AQ = A / Q

    effluentN = AQ * kN * fraction2

    effluentN1 = 1 / (1 + effluentN)

    NH3 = IN_NH3 

    effluentN2 = IN_NH3 * effluentN1


    # Calculation of feedback


    biomass_growth = 0.5 * (Influent_BOD1 - Effluent_BOD5)

    biomass_growth_array = np.full_like(IN_NH3, biomass_growth)

    Uptake_N1 = (14 / 115 ) * Px_bio_mg

    NH4_N_feedback_1 = Uptake_N1 - (IN_NH3 + Effluent_NH4_N)

    eff_nh3 = Uptake_N1 - (NH4_N_feedback_1) +  Effluent_NH4_N

    Removal_N = Uptake_N1 - NH4_N_feedback_1

    effluentN3 = effluentN2 + Removal_N


    # Step 35: ULTIMATE OXYGEN DEMAND

    Effluent_sbod_cal = IN_SBOD - Intr_SBOD_removal_123

    BOD = sbod5_P_at_desired_time

    TKN = (effluentN2 / 1.215) + Organic_nitrogen  # mg/L

    # TKN removal 

    f_pH = 2.71828 ** (0.2 * (pH - 7))

    AQ_k_fpH = AQ * K_t * f_pH

    TKN_removal = TKN / (1 + AQ_k_fpH)

    NBOD = 4.6 * TKN_removal  # mg/L


    CBOD5 = BOD 

    NH3 = Effluent_NH4 * 0.9441  # mg/L


    Flow_rate_2 = Q * 0.000409  # ft3/s

    Flow_rate_3 = Flow_rate_2 * 0.646317


    UOD_2 = ((cBOD5_Multiplier * CBOD5) + 4.57 * eff_nh3 ) * Flow_rate_3 * 8.34


    # Aerated Stabilization Basin 2



    # Define the function representing the differential equation

    def dSBOD5_dt1(SBOD51, t1, k201, OPO41, Kopo41, T):
        return -k201 * (OPO41 / (Kopo41 + OPO41)) * (1.05 ** (T - 10)) * SBOD51

   

    # Extract parameter values
    parameter_values = parameters_df.set_index(
        'Parameter').loc[:, 'expected'].to_dict()

    # Extract specific parameters needed for the simulation
    T = parameter_values['T']

    # Set initial conditions and parameters
    initial_SBOD51 = sbod_concentration_at_specific_time
    k201 = 0.02
    OPO41 = opo4_remaining_at_desired_time
    Kopo41 = 0.05


    # Set the time points for integration
    # Adjust the time range and number of points as needed
    time_points1 = np.linspace(0, 10, 100)

    # Solve the differential equation using odeint
    result1 = odeint(dSBOD5_dt1, initial_SBOD51, time_points1,
                    args=(k201, OPO41, Kopo41, T))


    # Define your differential equation function for OPO4
    def dOPO4_dt(OPO4, t, k201, OPO41, Kopo41, T):
        dOPO4_dt_value = k201 * OPO4 * (OPO41 / (OPO41 + Kopo41))**(T / 12)
        return dOPO4_dt_value


    result_OPO4 = odeint(dOPO4_dt, OPO41, time_points1, args=(k201, OPO41, Kopo41, T))
    # The result variable now contains the integrated values of SBOD5 over time

    # Suppose you want to get the value of SBOD5 at time t = 5 (for example)
    desired_time1 = 26.4

    # Find the index in the time_points array that is closest to the desired_time
    index_at_desired_time1 = np.abs(time_points1 - desired_time1).argmin()

    # Get the corresponding value of SBOD5 from the result array
    sbod5_P_at_desired_time1 = result1[index_at_desired_time1]

    opo4_at_desired_time1 = result_OPO4[index_at_desired_time1]

    # Calculate the remaining amount of OPO4 after consumption for SBOD removal
    opo4_left_after_consumption = opo4_at_desired_time1 - OPO41



    # Define the function representing the differential equation for SBOD5

    def dSBOD5_dt1(SBOD51, t1, k201, OPO41, Kopo41, T):
        return -k201 * (OPO41 / (Kopo41 + OPO41)) * (1.05 ** (T - 10)) * SBOD51

    # Define the function representing the differential equation for OPO4

    def dOPO4_dt(OPO4, t, SBOD51, k201, OPO41, Kopo41, T):
        # Assuming the OPO4 is consumed in proportion to the SBOD5 degradation
        return -k201 * OPO4 * (SBOD51 / (Kopo41 + SBOD51)) * (1.05 ** (T - 10))

    

    # Extract parameter values
    parameter_values = parameters_df.set_index('Parameter').loc[:, 'expected'].to_dict()

    # Extract specific parameters needed for the simulation
    T = parameter_values['T']

    # Set initial conditions and parameters
    initial_SBOD51 = sbod5_P_at_desired_time  # SBOD5 at the start of this calculation
    k201 = 0.02
    OPO41 = opo4_remaining_at_desired_time  # Initial OPO4 at the start of this calculation
    Kopo41 = 0.05

    # Set the time points for integration
    # Adjust the time range and number of points as needed
    time_points1 = np.linspace(0, 10, 100)

    # Solve the differential equation for SBOD5 using odeint
    result1 = odeint(dSBOD5_dt1, initial_SBOD51, time_points1, args=(k201, OPO41, Kopo41, T))

    # Solve the differential equation for OPO4 using odeint
    # Use the SBOD5 values from result1 in the OPO4 equation
    result_OPO4 = odeint(dOPO4_dt, OPO41, time_points1, args=(initial_SBOD51, k201, OPO41, Kopo41, T))

    # Suppose you want to get the value of SBOD5 and OPO4 at a specific time, say t = 26.4
    desired_time1 = 26.4

    # Find the index in the time_points array that is closest to the desired_time
    index_at_desired_time1 = np.abs(time_points1 - desired_time1).argmin()

    # Get the corresponding values of SBOD5 and OPO4 from the result arrays
    sbod5_P_at_desired_time1 = result1[index_at_desired_time1]
    opo4_at_desired_time1 = result_OPO4[index_at_desired_time1]

    # Calculate the remaining amount of OPO4 after consumption for SBOD removal
    # Here we are using the OPO4 value directly from the result_OPO4 array
    # There's no need to subtract OPO41 as it was already adjusted in the differential equation
    opo4_left_after_consumption1 = opo4_at_desired_time1




    Growth_P_Uptake = (P_Uptake_ratio / 115) * biomass_growth  # mgP/L

    PO4_P_fb1 = Growth_P_Uptake -  (opo4_remaining_at_desired_time +  opo4_left_after_consumption) # mgP/L



    Calculated_Effluent_PO4 = (opo4_left_after_consumption) + (PO4_P_fb1)


    Px_1B = aeration_1B / Volume_1B

    Power_1B = Px_1B * Volume_1B

    K1 =  2.5 * (1.06 ** (T - 20)) 

    Effluent_BOD51 = sbod5_P_at_desired_time1

    Effluent_BOD5_1B = 1 / (1 + (K1 *Hydrau_Reten)) * Effluent_BOD51    

    BOD5_removal_1B = BOD5_removal - Effluent_BOD5_1B    #mg/L


    x_numerator_1B = Y * (BOD5_removal - Effluent_BOD5_1B)

    x_denomenator_1B = 1 + (kd * Hydrau_Reten)

    x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

    Px_1B = (x_1B * Q) / 1000  # kg/d

    Px_O2_1B = 1.42 * Px_1B

    O2_requirement_1B = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d


    Power_req_bod_1B = O2_requirement_1B / (Coeff_Power)  # kW


    # STEP 19- Ox supply- Calculate available oxygen supply

    Power_req_sbod_1B = Power_1B - Power_req_bod_1B

    DO_supply_1B = Power_req_sbod_1B * (Coeff_Power)  # kg/d


    # STEP 13: ΔSBOD- Calculate cell effluent SBOD5

    SBOD_removal_1B = Effluent_SBOD - SBOD_1B  # mg/L or g/m3

    SBOD_rem_g_d_1B = SBOD_removal_1B * Q  # g/d


    SBOD_Yield_1B = VSS_per_day / SBOD_rem_g_d_1B  # g VSS / sBOD


    SBOD_oxygen_used_1B = (Effluent_SBOD * Q) - \
        (SBOD_removal_1B * Q)  # g O2/ g sBOD

    Oxygen_per_unit_SBOD_1B = SBOD_oxygen_used_1B / SBOD_rem_g_d_1B

    SBOD_P_1B = Effluent_sbod_cal - sbod5_P_at_desired_time1

    Aerobic_SBOD_1b = SBOD_removal_1B - SBOD_P_1B

    # STEP 20- DOΔSBOD- Calculate SBOD5 removal supported by oxygen supply


    # Define the system of ODEs

    def model1(y1, t1):
        S1, X1, DO1 = y1
        dSdt1 = -(Us1 * S1 / (Ks1 + S1)) * D01 / (Ko1 + DO1)
        dXdt1 = a1 * dSdt1 - d1 * X1
        dDOdt1 = -a_prime1 * dSdt1 - d_prime1 * X1 + KLa1 * (DOs1 - DO1)
        return [dSdt1, dXdt1, dDOdt1]


    # Set parameter values
    Us1 = 0.1  # maximum rate of sbod removal
    Ks1 = 100  # saturation coefficient for sBOD
    D01 = 12634  # concentration of dissolved oxygen
    Ko1 = 0.1  # saturation coefficient for DO
    a1 = 0.70   # yield coefficient
    d1 = 0.002   # yield cofficient
    a_prime1 = 0.34  # oxygen-use coefficient
    d_prime1 = 0.0008  # oxygen-use coefficient
    KLa1 = 2  # overall-oxygen transfer rate
    DOs1 = 6  # saturation concentration of dissolved oxygen

    
    # Extract parameter values
    parameter_values = parameters_df.set_index(
        'Parameter').loc[:, 'expected'].to_dict()

    # Extract specific parameters needed for the simulation
    Coeff_Power = parameter_values['Coeff_Power']
    aeration_1B = parameter_values['aeration_1B']
    Volume_1B = parameter_values['Volume_1B']
    Effluent_BOD_1B = parameter_values['Effluent_BOD_1B']
    Y = parameter_values['Y']
    kd = parameter_values['kd']
    Hydrau_Reten = parameter_values['Hydrau_Reten']
    f = parameter_values['f']


    K1 = 2.5 * (1.06 ** (T - 20))   

    Effluent_BOD51 = sbod5_P_at_desired_time1

    Effluent_BOD5_1B = 1 / (1 + (K1 *Hydrau_Reten)) * Effluent_BOD51

    BOD5_removal_1B = BOD5_removal - Effluent_BOD5_1B    #mg/L


    x_numerator_1B = Y * (BOD5_removal - Effluent_BOD5_1B)

    x_denomenator_1B = 1 + (kd * Hydrau_Reten)

    x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

    Px_1B = (x_1B * Q) / 1000  # kg/d

    Px_O2_1B = 1.42 * Px_1B

    O2_requirement_1B = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d


    Power_req_bod_1B = O2_requirement_1B / (Coeff_Power)  # kW


    Px_1B = aeration_1B / Volume_1B

    Power_1B = Px_1B * Volume_1B

    Power_req_sbod_1B = Power_1B - Power_req_bod_1B

    DO_supply_1B = Power_req_sbod_1B * (Coeff_Power)

    # Initial conditions
    initial_SBOD1 = sbod5_P_at_desired_time1
    # Replace with your initial values
    initial_conditions1 = [initial_SBOD1[0], 20, 12643]

    # Time points for integration
    t1 = np.linspace(0, 20, 100)  # Replace with your time range

    # Solve the differential equations using odeint
    solution1 = odeint(model1, initial_conditions1, t1)


    # Solve the differential equations using odeint
    solution1 = odeint(model1, initial_conditions1, t1)

    # Extract BOD concentration at a specific time
    specific_time1 = 26.4  # Replace with the desired time
    # Find the index closest to the specified time
    index1 = np.abs(t1 - specific_time1).argmin()

    # BOD concentration at the specified time
    sbod_concentration_at_specific_time1 = solution1[index1, 0]

    aerobic_biomass_yield = 0.5

    Biomass_Growth1 = aerobic_biomass_yield * (Effluent_BOD - Effluent_BOD_1B)  # mg/L


    bh_t = bh * (1.04 ** (T - 20))

    Px_bio1B = ( Q * YH * BOD5_removal_1B ) / 1000

    Px_bio2B = [1 + bh_t * Hydrau_Reten ]

    Px_bio3B = Px_bio1B / Px_bio2B

    Px_bio4B = (fd_o * bh_t * Q * YH * BOD5_removal_1B * Hydrau_Reten) / 1000

    Px_bio5B = 1 + bh_t * Hydrau_Reten 

    Px_bio6B = Px_bio4B / Px_bio5B

    Px_bioB = Px_bio3B  + Px_bio6B



    #Nitrification 

    spec_growth_AOB1 = max_growth_coefficient * (1.072 ** (T - 20))

    spec_endo_decay1 = b_20 * (1.029 ** (T - 20))

    growth_NH41 = spec_growth_AOB1 * (eff_nh3  / (eff_nh3  + K_NH4))

    spec_growth_NH41 = growth_NH41 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay1

    SRT1 = (1 / spec_growth_NH41) * 1.5

    S_NH1 = K_NH4 * (1 + (spec_endo_decay * SRT1))

    growth_AOB_DO1 = spec_growth_AOB1 * ((S_DO) / (S_DO + K_AOB))

    Effluent_NH4_N_numenator1 = (SRT1 * (growth_AOB_DO1 - spec_endo_decay1)) - 1.0

    Effluent_NH4_N1 = (S_NH1 / Effluent_NH4_N_numenator1 ) * (Hydrau_Reten / SRT1)

    r_NH = (spec_growth_AOB / 0.15 ) * (eff_nh3 / (eff_nh3 + K_NH4))

    r_NH12 = r_NH * (S_DO / (S_DO + K_AOB)) * 30

    NOX1 = r_NH12 * 1.1 

    X_AOB = Q * 0.15 * NOX * SRT 

    X_AOB1 = X_AOB / (V * (1 + spec_endo_decay * SRT))

    Px_NH4 = (Q * 0.15 * (NOX)) / 1000

    Px_NH41 = 1 + (0.315) * Hydrau_Reten 

    Px_NH42 = Px_NH4 / Px_NH41

    Px_bio_isr = Px_bio3B  + Px_bio6B 

    Px_bio_mg1 =  (Px_bio_isr * 1000000 )/ (Q * 1000 )     #mg/L

    TSS_pro = Px_bio_isr + ((Q * (Eff_TSS - Total_VSS_1A)) / 1000)

    TSS_pro1 = TSS_pro 

    TSS_pro2 = (TSS_pro1 * 1000000 )/ (Q * 1000 ) + TSS_pro2A

    Benchmark_efficiency1 = 84 - (10.6 * (aeration_1B / Volume_1B))


    Settled_solids1B =  ((TSS_pro2 / 100) * Benchmark_efficiency1)

    Eff_TSS_1B =  TSS_pro2 - Settled_solids1B


    # Calculation of feedback1
    biomass_growth1 = 0.5 * ( Effluent_BOD51 - Effluent_BOD5_1B )

    Settled_solids1 = TSSi + biomass_growth1 - Effluent_TSS 

    BOD_fb1 = 0.3 * biomass_growth1

    biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

    Uptake_N12 = (14 / 115 ) * Px_bio_mg1

    NH4_N_feedback_12 = Uptake_N1 - (eff_nh3 + Effluent_NH4_N1) 

    eff_nh31 = Uptake_N12 - NH4_N_feedback_12 + Effluent_NH4_N1

    # Define the values
    pKw = 14.0  # Example value at 25°C
    pKb = 4.75  # Example value for ammonia at 25°C

    # Assume pH is an array of pH values
    pH1 = 8 # Example array of pH values

    # Calculate the exponent for each pH value
    exponent1 = pKw - pKb - pH1

    # Calculate 10^(exponent) for each pH value
    fraction1 = 10 ** exponent1



    fraction2_1 = 1 / (1 + fraction1)

    kN1_1 = 2.71828 ** (1.57 * (pH1 - 8.5))

    kN2_1 = 2.71828 ** (0.13 * (T - 20))

    kN_1 = kN1_1 * kN2_1

    A = V / 4

    AQ = A / Q

    effluentN1_1 = AQ * kN_1 * fraction2_1

    effluentN_12 = 1 / (1 + effluentN1_1)

    NH3_i = effluentN2 + Removal_N

    effluentN2_1 =  NH3_i  * effluentN_12


    # Calculation of feedback1


    biomass_growth1 = 0.5 * ( Effluent_BOD5 - Effluent_BOD5_1B )

    biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

    Uptake_N12 = (14 / 115 ) * biomass_growth1

    NH4_N_feedback_12 = Uptake_N12 - (effluentN2 + effluentN2_1) 

    effluent_NH3 = effluentN2_1 + NH4_N_feedback_12 

    # Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass


    P_Uptake_ratio = 2.2

    Growth_P_Uptake1 = (P_Uptake_ratio / 115) * Biomass_Growth1  # mgP/L

    PO4_P_fb1 = Growth_P_Uptake - opo4_remaining_at_desired_time + opo4_left_after_consumption1  # mgP/L

    Effluent_PO4 = opo4_left_after_consumption1 + PO4_P_fb1

    Effluent_sbod_cal1 = Effluent_sbod_cal - sbod_concentration_at_specific_time1

    BOD1 =  Effluent_BOD5_1B + BOD_fb1

    Ammonia_N1 = IN_NH4 * 0.9441

    TKN1 = (effluentN2_1 / 1.215) + Organic_nitrogen1  # mg/L

    # TKN removal 

    f_pH1 = 2.71828 ** (0.2 * (pH1 - 7))

    AQ_k_fpH1 = AQ * K_t * f_pH1

    TKN_removal1 = TKN_removal / (1 + AQ_k_fpH1)

    NBOD1 = 4.6 * TKN_removal1

    CBOD51 = BOD1 

    UOD_3 = ((cBOD5_Multiplier * CBOD51) + 4.57 * eff_nh31) * Flow_rate_3 * 8.34

    
    return UOD_3

   




def calculate_uncertainty(assumption_data,low_COD,expected_COD,high_COD,
                          correlation_distributions,correlation_parameters,n_samples):
    
    dis = {'correlation':'yes','distribution':'uniform','low':low_COD,'expected':expected_COD,'high':high_COD}

    dis_series= pd.Series(dis)

    result = setup_data(assumption_data,correlation_distributions,correlation_parameters,n_samples,check_variable="COD_influent",replace_series=dis_series)

    zero = result['zero'] 
    flowrate_influent = result['flowrate_influent']
    steel_cost = result['steel_cost']
    electricity_cost = result['electricity_cost']
    stainless_steel_cost = result['stainless_steel_cost']
    COD_influent = result['COD_influent']

    

    # inputs
    # wastewater (WW)
    output_perc_mid = 50
    output_perc_low = 5
    output_perc_high = 95
    WW_influent_COD = COD_influent * flowrate_influent / 1000 # kg COD / d
    total_cost = WW_influent_COD * (steel_cost + electricity_cost + stainless_steel_cost)

    #%% outputs
    total_cost_mid = np.percentile(total_cost, output_perc_mid)

    all_inputs_name =[]
    all_inputs = []

    for item in result.keys():
        if isinstance(result[item], np.ndarray)==True:
            all_inputs_name.append(item)
            all_inputs.append(result[item])
        
    output_name = ('Total', 'dummy')
    output_data = [total_cost, zero]
    dfinputs = pd.DataFrame({k:v.flatten() for k,v in zip(all_inputs_name, all_inputs)})
    #dfcost = pd.DataFrame(total_cost)
    dfcost = pd.DataFrame({k:v.flatten() for k,v in zip(output_name, output_data)})

    sensitivity = (dfinputs.corrwith(dfcost.Total, method='spearman'))

    # taking absolute values of all the day and then sorting by the values in descending order
    sensitivity = sensitivity.abs()
    sensitivity = sensitivity.sort_values(ascending=False)

    COD_influent = COD_influent.flatten()
    total_cost = total_cost.flatten()

    return COD_influent, total_cost, sensitivity, total_cost_mid



def calculate_cost(result,COD_influent):

    zero = result['zero'] 
    flowrate_influent = result['flowrate_influent']
    steel_cost = result['steel_cost']
    electricity_cost = result['electricity_cost']
    stainless_steel_cost = result['stainless_steel_cost']
    
    # inputs
    # wastewater (WW)
    output_perc_mid = 50
    output_perc_low = 5
    output_perc_high = 95
    WW_influent_COD = COD_influent * flowrate_influent / 1000 # kg COD / d
    total_cost = WW_influent_COD * (steel_cost + electricity_cost + stainless_steel_cost)

    #%% outputs
    total_cost_mid = np.percentile(total_cost, output_perc_mid)

    all_inputs_name =[]
    all_inputs = []

    for item in result.keys():
        if isinstance(result[item], np.ndarray)==True:
            all_inputs_name.append(item)
            all_inputs.append(result[item])
        
    output_name = ('Total', 'dummy')
    output_data = [total_cost, zero]
    dfinputs = pd.DataFrame({k:v.flatten() for k,v in zip(all_inputs_name, all_inputs)})
    #dfcost = pd.DataFrame(total_cost)
    dfcost = pd.DataFrame({k:v.flatten() for k,v in zip(output_name, output_data)})

    sensitivity = (dfinputs.corrwith(dfcost.Total, method='spearman'))

    # taking absolute values of all the day and then sorting by the values in descending order
    sensitivity = sensitivity.abs()
    sensitivity = sensitivity.sort_values(ascending=False)


    return COD_influent, total_cost_mid, sensitivity

