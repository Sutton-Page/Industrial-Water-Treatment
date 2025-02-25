import warnings
# Remove anoying warning
warnings.filterwarnings("ignore")

from dash import Dash, html,ctx, dcc,Input,Output,State
from dash.exceptions import PreventUpdate
from funcs import setup_data,run_model,group_data,extract_inital_data,run_combined_ponds,run_pond1B
from tabs import *
import plotly.express as px
import pandas as pd
import numpy as np



combined_general_assumptions = pd.read_excel('both-ponds.xlsx', sheet_name = 'General', index_col = 'Parameter')
combined_cost_assumptions = pd.read_excel('both-ponds.xlsx', sheet_name = 'Cost', index_col='Parameter')
combined_LCA_assumptions = pd.read_excel('both-ponds.xlsx', sheet_name = 'LCA', index_col='Parameter')
combined_design_assumptions = pd.read_excel('both-ponds.xlsx', sheet_name = 'Design', index_col='Parameter')

pond1B_general_assumptions = pd.read_excel('pond1B.xlsx', sheet_name = 'General', index_col = 'Parameter')
pond1B_cost_assumptions = pd.read_excel('pond1B.xlsx', sheet_name = 'Cost', index_col='Parameter')
pond1B_LCA_assumptions = pd.read_excel('pond1B.xlsx', sheet_name = 'LCA', index_col='Parameter')
pond1B_design_assumptions = pd.read_excel('pond1B.xlsx', sheet_name = 'Design', index_col='Parameter')


#global variable for downloading graph data as excel file
ultimate_oxygen_combined_data = pd.DataFrame()


#Replacing 
#default-IN_COD , IN_COD -> Organic_nitrogen
#default-solids_percent , solids percent -> T (temperature)

# Replacing
#default-Power -> default-aeration
# Power -> aeration

t1 = [
    'default', 'default-Influent_BOD', 
    'default-Hydrau_Reten', 'default-TSSi','default-air_temp','default-Ti',
    'default-aeration','default-IN_Q', 'default-IN_NH3_z','default-Sup_NH3',
    'default-IN_PO4_z','default-Sup_PO4']



t2 = ['default', 'CBOD5','Hydrau_Reten','Eff_TSS',
      'air_temp_1b','Ti_1B','aeration_1B','Flow_rate_3','eff_nh3_isr','EFF_PO4'
      ]

t3 = ['default','com-Influent_BOD','com-IN_NH3_z',
      'com-Sup_NH3','com-IN_PO4_z','com-Sup_PO4','com-IN_Q',
      'com-aeration','com-aeration_1B','com-Ti',
      'com-air_temp','com-Ti_1B','com-air_temp_1b','com-TSSi','com-Hydrau_Reten']



#advanced_layout = experiment([t1,t2],pd.DataFrame({}))
main_layout = generate_basis_input_layout()
advanced_layout = experiment([t1,t2,t3],pd.DataFrame({}))
# creating default violin


app = Dash(__name__)
app.title = "Effluent Treatment System Digital Twin"
server = app.server


app.layout = html.Div(className="container", children=[

    
    html.Div(className="content", children=[
    
        html.Img(id="top-image",src=""),
        html.H2("Effluent Treatment System Digital Twin"),

        dcc.Tabs(
            id="tab_control",
            value="tab_basic",
            children=[

                dcc.Tab(
                    label="Basic Inputs",
                    value="tab_basic",
                    className="custom-tab",
                    selected_className="custom-tab-selected"
                ),
                dcc.Tab(
                    label="Advanced Inputs",
                    value="tab_main",
                    className="custom-tab",
                    selected_className="custom-tab-selected"
                ),

                
                dcc.Tab(
                    label="Background Info",
                    value="tab_info",
                    className="custom-tab",
                    selected_className="custom-tab-selected"
                ),
                dcc.Tab(
                    label="Demonstration Video",
                    value="tab_example",
                    className="custom-tab",
                    selected_className="custom-tab-selected"
                )
            ],
            className="custom-tabs"
        ),
        html.Div(id="main_content",children=[main_layout])
    ])
])


@app.callback(
        
        Output("download-excel","data"),
        Input("excel-button","n_clicks")

)

def download_oxygen_data(n_clicks):

    if n_clicks == None:

        raise PreventUpdate
    
    else:

        if len(ultimate_oxygen_combined_data.index)!=0:

            return dcc.send_data_frame(ultimate_oxygen_combined_data.to_excel,
                                   "Ultimate_Oxygen_data.xlsx", sheet_name="Data")


@app.callback(
        
        
        Output('CBOD5-low','value'),
        Output('CBOD5-expected','value'),
        Output('CBOD5-high','value'),
        
        Output('Hydrau_Reten-low','value'),
        Output('Hydrau_Reten-expected','value'),
        Output('Hydrau_Reten-high','value'),

        Output('Eff_TSS-low','value'),
        Output('Eff_TSS-expected','value'),
        Output('Eff_TSS-high','value'),

        Output('air_temp_1b-low','value'),
        Output('air_temp_1b-expected','value'),
        Output('air_temp_1b-high','value'),


        Output('Ti_1B-low','value'),
        Output('Ti_1B-expected','value'),
        Output('Ti_1B-high','value'),

        Output('aeration_1B-low','value'),
        Output('aeration_1B-expected','value'),
        Output('aeration_1B-high','value'),

        Output('Flow_rate_3-low','value'),
        Output('Flow_rate_3-expected','value'),
        Output('Flow_rate_3-high','value'),

        Output('eff_nh3_isr-low','value'),
        Output('eff_nh3_isr-expected','value'),
        Output('eff_nh3_isr-high','value'),

        Output('EFF_PO4-low','value'),
        Output('EFF_PO4-expected','value'),
        Output('EFF_PO4-high','value'),

        Input("load-default-B","n_clicks"),
        Input("clear-B","n_clicks")

)

def clear_senarioB(n_clicks,n_clicks2):

    if n_clicks == None and n_clicks2 == None:

        raise PreventUpdate
    
    button_clicked = ctx.triggered_id


    if button_clicked == "load-default-B":

        t2 = [
            'default', 'CBOD5','Hydrau_Reten','Eff_TSS',
            'air_temp_1b','Ti_1B','aeration_1B','Flow_rate_3','eff_nh3_isr','EFF_PO4'
            ]
        
        
        inital_values = extract_inital_data(pond1B_design_assumptions,t2)

        
        return inital_values['CBOD5']['low'],inital_values['CBOD5']['expected'],inital_values['CBOD5']['high'],inital_values['Hydrau_Reten']['low'],inital_values['Hydrau_Reten']['expected'],inital_values['Hydrau_Reten']['high'],inital_values['Eff_TSS']['low'],inital_values['Eff_TSS']['expected'],inital_values['Eff_TSS']['high'],inital_values['air_temp_1b']['low'],inital_values['air_temp_1b']['expected'],inital_values['air_temp_1b']['high'],inital_values['Ti_1B']['low'],inital_values['Ti_1B']['expected'],inital_values['Ti_1B']['high'],inital_values['aeration_1B']['low'],inital_values['aeration_1B']['expected'],inital_values['aeration_1B']['high'],inital_values['Flow_rate_3']['low'],inital_values['Flow_rate_3']['expected'],inital_values['Flow_rate_3']['high'],inital_values['eff_nh3_isr']['low'],inital_values['eff_nh3_isr']['expected'],inital_values['eff_nh3_isr']['high'],inital_values['EFF_PO4']['low'],inital_values['EFF_PO4']['expected'],inital_values['EFF_PO4']['high']
    
    elif button_clicked == "clear-B":

        return None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None


@app.callback(
        
        
    
        Output('com-aeration_1B-low','value'),
        Output('com-aeration_1B-expected','value'),
        Output('com-aeration_1B-high','value'),

        Output('com-Ti_1B-low','value'),
        Output('com-Ti_1B-expected','value'),
        Output('com-Ti_1B-high','value'),

        Output('com-air_temp_1b-low','value'),
        Output('com-air_temp_1b-expected','value'),
        Output('com-air_temp_1b-high','value'),

        Output('com-Influent_BOD-low','value'),
        Output('com-Influent_BOD-expected','value'),
        Output('com-Influent_BOD-high','value'),
    
        Output('com-Hydrau_Reten-low','value'),
        Output('com-Hydrau_Reten-expected','value'),
        Output('com-Hydrau_Reten-high','value'),

        Output('com-TSSi-low','value'),
        Output('com-TSSi-expected','value'),
        Output('com-TSSi-high','value'),

        Output('com-air_temp-low','value'),
        Output('com-air_temp-expected','value'),
        Output('com-air_temp-high','value'),

        Output('com-Ti-low','value'),
        Output('com-Ti-expected','value'),
        Output('com-Ti-high','value'),

        Output('com-aeration-low','value'),
        Output('com-aeration-expected','value'),
        Output('com-aeration-high','value'),

        Output('com-IN_Q-low','value'),
        Output('com-IN_Q-expected','value'),
        Output('com-IN_Q-high','value'),
       
        Output('com-IN_NH3_z-low','value'),
        Output('com-IN_NH3_z-expected','value'),
        Output('com-IN_NH3_z-high','value'),

        Output('com-Sup_NH3-low','value'),
        Output('com-Sup_NH3-expected','value'),
        Output('com-Sup_NH3-high','value'),

    
        Output('com-IN_PO4_z-low','value'),
        Output('com-IN_PO4_z-expected','value'),
        Output('com-IN_PO4_z-high','value'),

        Output('com-Sup_PO4-low','value'),
        Output('com-Sup_PO4-expected','value'),
        Output('com-Sup_PO4-high','value'),

        Input("com-load","n_clicks"),
        Input("com-clear","n_clicks"),
        
)

def com_load_clear_default_values(n_clicks,n_clicks2):

    if n_clicks == None and n_clicks2 == None:

        raise PreventUpdate
    
    button_clicked = ctx.triggered_id

    if button_clicked == "com-load":
         
         t2 =  [
    'default', 'Influent_BOD', 
    'Hydrau_Reten', 'TSSi','air_temp','Ti',
    'aeration','IN_Q', 'IN_NH3_z','Sup_NH3',
    'IN_PO4_z','Sup_PO4','aeration_1B','Ti_1B','air_temp_1b']
         
         inital_values = extract_inital_data(combined_design_assumptions,t2)
        
       
         return inital_values['aeration_1B']['low'],inital_values['aeration_1B']['expected'],inital_values['aeration_1B']['high'],inital_values['Ti_1B']['low'],inital_values['Ti_1B']['expected'],inital_values['Ti_1B']['high'],inital_values['air_temp_1b']['low'],inital_values['air_temp_1b']['expected'],inital_values['air_temp_1b']['high'],inital_values['Influent_BOD']['low'],inital_values['Influent_BOD']['expected'],inital_values['Influent_BOD']['high'],inital_values['Hydrau_Reten']['low'],inital_values['Hydrau_Reten']['expected'],inital_values['Hydrau_Reten']['high'],inital_values['TSSi']['low'],inital_values['TSSi']['expected'],inital_values['TSSi']['high'], inital_values['air_temp']['low'],inital_values['air_temp']['expected'],inital_values['air_temp']['high'], inital_values['Ti']['low'],inital_values['Ti']['expected'],inital_values['Ti']['high'],inital_values['aeration']['low'],inital_values['aeration']['expected'],inital_values['aeration']['high'],inital_values['IN_Q']['low'],inital_values['IN_Q']['expected'],inital_values['IN_Q']['high'],inital_values['IN_NH3_z']['low'],inital_values['IN_NH3_z']['expected'],inital_values['IN_NH3_z']['high'],inital_values['Sup_NH3']['low'],inital_values['Sup_NH3']['expected'],inital_values['Sup_NH3']['high'],inital_values['IN_PO4_z']['low'],inital_values['IN_PO4_z']['expected'],inital_values['IN_PO4_z']['high'],inital_values['Sup_PO4']['low'],inital_values['Sup_PO4']['expected'],inital_values['Sup_PO4']['high']
    
    elif button_clicked == "com-clear":

        return None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None



@app.callback(
        
        
        Output('default-Influent_BOD-low','value'),
        Output('default-Influent_BOD-expected','value'),
        Output('default-Influent_BOD-high','value'),
    
        
        Output('default-Hydrau_Reten-low','value'),
        Output('default-Hydrau_Reten-expected','value'),
        Output('default-Hydrau_Reten-high','value'),

        Output('default-TSSi-low','value'),
        Output('default-TSSi-expected','value'),
        Output('default-TSSi-high','value'),

        Output('default-air_temp-low','value'),
        Output('default-air_temp-expected','value'),
        Output('default-air_temp-high','value'),

        Output('default-Ti-low','value'),
        Output('default-Ti-expected','value'),
        Output('default-Ti-high','value'),

        Output('default-aeration-low','value'),
        Output('default-aeration-expected','value'),
        Output('default-aeration-high','value'),

        Output('default-IN_Q-low','value'),
        Output('default-IN_Q-expected','value'),
        Output('default-IN_Q-high','value'),
       
        Output('default-IN_NH3_z-low','value'),
        Output('default-IN_NH3_z-expected','value'),
        Output('default-IN_NH3_z-high','value'),

        Output('default-Sup_NH3-low','value'),
        Output('default-Sup_NH3-expected','value'),
        Output('default-Sup_NH3-high','value'),

        

        Output('default-IN_PO4_z-low','value'),
        Output('default-IN_PO4_z-expected','value'),
        Output('default-IN_PO4_z-high','value'),

        Output('default-Sup_PO4-low','value'),
        Output('default-Sup_PO4-expected','value'),
        Output('default-Sup_PO4-high','value'),

        Input("load-default","n_clicks"),
        Input("clear-default","n_clicks"),
        
       
)

def load_default_values(n_clicks,n_clicks2):

    if n_clicks == None and n_clicks2 == None:

        raise PreventUpdate
    
    button_clicked = ctx.triggered_id

    if button_clicked == "load-default":

        t2 =  [
    'default', 'Influent_BOD', 
    'Hydrau_Reten', 'TSSi','air_temp','Ti',
    'aeration','IN_Q', 'IN_NH3_z','Sup_NH3',
    'IN_PO4_z','Sup_PO4']
        

        inital_values = extract_inital_data(pond1B_design_assumptions,t2)
        

        return inital_values['Influent_BOD']['low'],inital_values['Influent_BOD']['expected'],inital_values['Influent_BOD']['high'],inital_values['Hydrau_Reten']['low'],inital_values['Hydrau_Reten']['expected'],inital_values['Hydrau_Reten']['high'],inital_values['TSSi']['low'],inital_values['TSSi']['expected'],inital_values['TSSi']['high'], inital_values['air_temp']['low'],inital_values['air_temp']['expected'],inital_values['air_temp']['high'], inital_values['Ti']['low'],inital_values['Ti']['expected'],inital_values['Ti']['high'],inital_values['aeration']['low'],inital_values['aeration']['expected'],inital_values['aeration']['high'],inital_values['IN_Q']['low'],inital_values['IN_Q']['expected'],inital_values['IN_Q']['high'],inital_values['IN_NH3_z']['low'],inital_values['IN_NH3_z']['expected'],inital_values['IN_NH3_z']['high'],inital_values['Sup_NH3']['low'],inital_values['Sup_NH3']['expected'],inital_values['Sup_NH3']['high'],inital_values['IN_PO4_z']['low'],inital_values['IN_PO4_z']['expected'],inital_values['IN_PO4_z']['high'],inital_values['Sup_PO4']['low'],inital_values['Sup_PO4']['expected'],inital_values['Sup_PO4']['high']
    
    elif button_clicked == "clear-default":

        return None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None



@app.callback(
        
        Output("basic-graph","figure"),
        Output("display-median-basic",'children'),

        Input('basic-submit','n_clicks'),

        State('basic-PO4','value'),
        State('basic-Sup_PO4','value'),

        
        State("basic-IN_NH3_z",'value'),
        State("basic-Sup_NH3",'value'),

        
        State('basic-BOD','value'),

        State('basic-TSSi','value'),

        State('basic-aeration','value'),

        State('basic-aeration_1B','value'),
        State('basic-Hydrau_Reten','value'),
        State('basic-IN_Q','value'),
        State('basic-Ti','value'),
        State('basic-air_temp','value'),
        State('basic-Ti_1B','value'),

        State('basic-air_temp_1b','value'),



        
)

def basic_input_gen_graph(n_clicks,PO4,Sup_PO4,IN_NH3_z,Sup_NH3,BOD,TSSi,aeration,aeration_1b,
                          hydrau_reten,IN_Q,Ti,air_temp,Ti_1b,air_temp_1b):

    if n_clicks == None:

        raise PreventUpdate
    
    else:

        t2 = ['IN_COD','IN_PO4_z','aeration','Influent_BOD',
        'solids_percent','TSSi','Sup_PO4','IN_NH3_z','Sup_NH3','aeration',
        'aeration_1B','Hydrau_Reten','IN_Q','Ti','air_temp','Ti_1B','air_temp_1b']

        
            
        inital_values = extract_inital_data(combined_design_assumptions,t2)
        update_values = {'IN_PO4_z': PO4, 'IN_NH3_z':IN_NH3_z,
                         'Sup_NH3':Sup_NH3,
                         'Influent_BOD':BOD,'TSSi':TSSi,'Sup_PO4':Sup_PO4,
                         'aeration':aeration,
                         'aeration_1B': aeration_1b,
                         'Hydrau_Reten': hydrau_reten,'IN_Q': IN_Q,'Ti':Ti,
                         'air_temp': air_temp,'Ti_1B': Ti_1b,'air_temp_1b': air_temp_1b
                         }

        special_values = {}
    
        for item in update_values.keys():

            value = update_values[item]

            data  = {}
            percent = value * 0.1
            data['low'] = value - percent
            data['high'] = value + percent
            data['expected'] = value 

            special_values[item] = value

            inital_values[item] = data

        assumptions = [combined_general_assumptions,combined_design_assumptions,combined_cost_assumptions,
                        combined_LCA_assumptions]
        
        n_samples = int(combined_general_assumptions.loc['n_samples','expected'])
        correlation_distributions = np.full((n_samples, n_samples), np.nan)
        correlation_parameters = np.full((n_samples, 1), np.nan)
        correlation_parameters = correlation_parameters.tolist()

        # Replacing default value with real values
        data = setup_data(assumptions,correlation_distributions,correlation_parameters,
                                n_samples,check_variable=inital_values)
        

        print("GOT here")

        graph_y_axis_value = "UOD (lb/day)"

        #result = run_model(data,special_values)

        result = run_combined_ponds(data)

        default = result.flatten()

        median_value = np.median(default)
        median_value = "{:,.5f}".format(median_value)
        h4_text = f'Scenario median: {median_value} {graph_y_axis_value}'
        display_median = html.H4(h4_text)
            

        UOD= pd.DataFrame({
                graph_y_axis_value: default
            })

        return px.violin(UOD
                        ,y=graph_y_axis_value,width=750,height=750,box=True), display_median



@app.callback(
        
    Output("violin-graph","figure"),
    Output("display-median",'children'),

    dict(

        button=Input("submit-params",'n_clicks'),

        com_Influent_BOD_low=State('com-Influent_BOD-low','value'),
        com_Influent_BOD_expected=State('com-Influent_BOD-expected','value'),
        com_Influent_BOD_high=State('com-Influent_BOD-high','value'),

        com_IN_NH3_z_low = State('com-IN_NH3_z-low','value'),
        com_IN_NH3_z_expected = State('com-IN_NH3_z-expected','value'),
        com_IN_NH3_z_high = State('com-IN_NH3_z-high','value'),

        com_Sup_NH3_low = State('com-Sup_NH3-low','value'),
        com_Sup_NH3_expected = State('com-Sup_NH3-expected','value'),
        com_Sup_NH3_high = State('com-Sup_NH3-high','value'),

        com_IN_PO4_z_low=State('com-IN_PO4_z-low','value'),
        com_IN_PO4_z_expected=State('com-IN_PO4_z-expected','value'),
        com_IN_PO4_z_high=State('com-IN_PO4_z-high','value'),

        com_Sup_PO4_low = State('com-Sup_PO4-low','value'),
        com_Sup_PO4_expected = State('com-Sup_PO4-expected','value'),
        com_Sup_PO4_high = State('com-Sup_PO4-high','value'),

        com_IN_Q_low = State('com-IN_Q-low','value'),
        com_IN_Q_expected = State('com-IN_Q-expected','value'),
        com_IN_Q_high= State('com-IN_Q-high','value'),

        com_aeration_low = State('com-aeration-low','value'),
        com_aeration_expected = State('com-aeration-expected','value'),
        com_aeration_high = State('com-aeration-high','value'),

        com_aeration_1B_low = State('com-aeration_1B-low','value'),
        com_aeration_1B_expected = State('com-aeration_1B-expected','value'),
        com_aeration_1B_high = State('com-aeration_1B-high','value'),

        com_Ti_low = State("com-Ti-low",'value'),
        com_Ti_expected = State("com-Ti-expected",'value'),
        com_Ti_high = State("com-Ti-high",'value'),

        com_air_temp_low = State('com-air_temp-low','value'),
        com_air_temp_expected = State('com-air_temp-expected','value'),
        com_air_temp_high = State('com-air_temp-high','value'),

        com_Ti_1B_low = State('com-Ti_1B-low','value'),
        com_Ti_1B_expected = State('com-Ti_1B-expected','value'),
        com_Ti_1B_high = State('com-Ti_1B-high','value'),

        com_air_temp_1b_low = State('com-air_temp_1b-low','value'),
        com_air_temp_1b_expected = State('com-air_temp_1b-expected','value'),
        com_air_temp_1b_high = State('com-air_temp_1b-high','value'),

        com_TSSi_low = State('com-TSSi-low','value'),
        com_TSSi_expected = State('com-TSSi-expected','value'),
        com_TSSi_high = State('com-TSSi-high','value'),

        com_Hydrau_Reten_low = State('com-Hydrau_Reten-low','value'),
        com_Hydrau_Reten_expected = State('com-Hydrau_Reten-expected','value'),
        com_Hydrau_Reten_high = State('com-Hydrau_Reten-high','value'),

        default_Influent_BOD_low=State('default-Influent_BOD-low','value'),
        default_Influent_BOD_expected=State('default-Influent_BOD-expected','value'),
        default_Influent_BOD_high=State('default-Influent_BOD-high','value'),

        default_Hydrau_Reten_low=State('default-Hydrau_Reten-low','value'),
        default_Hydrau_Reten_expected=State('default-Hydrau_Reten-expected','value'),
        default_Hydrau_Reten_high=State('default-Hydrau_Reten-high','value'),

        default_TSSi_low=State('default-TSSi-low','value'),
        default_TSSi_expected=State('default-TSSi-expected','value'),
        default_TSSi_high=State('default-TSSi-high','value'),

        default_air_temp_low=State('default-air_temp-low','value'),
        default_air_temp_expected=State('default-air_temp-expected','value'),
        default_air_temp_high=State('default-air_temp-high','value'),

        default_Ti_low=State('default-Ti-low','value'),
        default_Ti_expected=State('default-Ti-expected','value'),
        default_Ti_high=State('default-Ti-high','value'),

        default_aeration_low=State('default-aeration-low','value'),
        default_aeration_expected=State('default-aeration-expected','value'),
        default_aeration_high=State('default-aeration-high','value'),

        default_IN_Q_low=State('default-IN_Q-low','value'),
        default_IN_Q_expected=State('default-IN_Q-expected','value'),
        default_IN_Q_high=State('default-IN_Q-high','value'),

        default_IN_NH3_z_low=State('default-IN_NH3_z-low','value'),
        default_IN_NH3_z_expected=State('default-IN_NH3_z-expected','value'),
        default_IN_NH3_z_high=State('default-IN_NH3_z-high','value'),

        default_Sup_NH3_low=State('default-Sup_NH3-low','value'),
        default_Sup_NH3_expected=State('default-Sup_NH3-expected','value'),
        default_Sup_NH3_high=State('default-Sup_NH3-high','value'),

        default_IN_PO4_z_low=State('default-IN_PO4_z-low','value'),
        default_IN_PO4_z_expected=State('default-IN_PO4_z-expected','value'),
        default_IN_PO4_z_high=State('default-IN_PO4_z-high','value'),

        default_Sup_PO4_low=State('default-Sup_PO4-low','value'),
        default_Sup_PO4_expected=State('default-Sup_PO4-expected','value'),
        default_Sup_PO4_high=State('default-Sup_PO4-high','value'),


        CBOD5_low=State('CBOD5-low','value'),
        CBOD5_expected=State('CBOD5-expected','value'),
        CBOD5_high=State('CBOD5-high','value'),

        Hydrau_Reten_low=State('Hydrau_Reten-low','value'),
        Hydrau_Reten_expected=State('Hydrau_Reten-expected','value'),
        Hydrau_Reten_high=State('Hydrau_Reten-high','value'),

        Eff_TSS_low=State('Eff_TSS-low','value'),
        Eff_TSS_expected=State('Eff_TSS-expected','value'),
        Eff_TSS_high=State('Eff_TSS-high','value'),

        air_temp_1b_low=State('air_temp_1b-low','value'),
        air_temp_1b_expected=State('air_temp_1b-expected','value'),
        air_temp_1b_high=State('air_temp_1b-high','value'),

        Ti_1B_low=State('Ti_1B-low','value'),
        Ti_1B_expected=State('Ti_1B-expected','value'),
        Ti_1B_high=State('Ti_1B-high','value'),

        aeration_1B_low=State('aeration_1B-low','state'),
        aeration_1B_expected=State('aeration_1B-expected','state'),
        aeration_1B_high=State('aeration_1B-high','state'),

        Flow_rate_3_low=State('Flow_rate_3-low','value'),
        Flow_rate_3_expected=State('Flow_rate_3-expected','value'),
        Flow_rate_3_high=State('Flow_rate_3-high','value'),

        eff_nh3_isr_low=State('eff_nh3_isr-low','value'),
        eff_nh3_isr_expected=State('eff_nh3_isr-expected','value'),
        eff_nh3_isr_high=State('eff_nh3_isr-high','value'),

    )
)

def run_multiple_types(**kwargs):

    global ultimate_oxygen_combined_data

    if kwargs['button'] == None:

        raise PreventUpdate
    
    else:

        pond1A_params = [ 'default_Influent_BOD', 
            'default_Hydrau_Reten', 'default_TSSi','default_air_temp','default_Ti',
            'default_aeration','default_IN_Q', 'default_IN_NH3_z','default_Sup_NH3',
            'default_IN_PO4_z','default_Sup_PO4']
        
        pond1B_params = ['CBOD5','Hydrau_Reten','Eff_TSS',
                'air_temp_1b','Ti_1B','aeration_1B','Flow_rate_3','eff_nh3_isr'
                ]
        
        combined_pond_params = ['com_Influent_BOD','com_IN_NH3_z',
            'com_Sup_NH3','com_IN_PO4_z','com_Sup_PO4','com_IN_Q',
            'com_aeration','com_aeration_1B','com_Ti',
            'com_air_temp','com_Ti_1B','com_air_temp_1b','com_TSSi','com_Hydrau_Reten']
        

        # grabbing the input data for pond1A and B from input 
        pond1A = group_data(pond1A_params,kwargs)
        pond1B = group_data(pond1B_params,kwargs)
        pond1A_and_B = group_data(combined_pond_params,kwargs)

        # Assumptions for pond1A and pond1B
        pond1A_assumptions = [pond1B_general_assumptions,pond1B_design_assumptions,pond1B_cost_assumptions,
                       pond1B_LCA_assumptions]
        
        pond1B_assumptions = [pond1B_general_assumptions,pond1B_design_assumptions,pond1B_cost_assumptions,
                       pond1B_LCA_assumptions]
        
        pond1A_and_B_assumptions = [combined_general_assumptions,combined_design_assumptions,           combined_cost_assumptions,combined_LCA_assumptions]
        

        pond1A_data = None
        pond1B_data = None
        combined_pond_data = None

        if pond1A_and_B != {}:

            n_samples = int(combined_general_assumptions.loc['n_samples','expected'])
            correlation_distributions = np.full((n_samples, n_samples), np.nan)
            correlation_parameters = np.full((n_samples, 1), np.nan)
            correlation_parameters = correlation_parameters.tolist()

            combined_pond_data = setup_data(pond1A_and_B_assumptions,correlation_distributions,correlation_parameters,
                                n_samples,check_variable=pond1A_and_B)

        if pond1A != {}:

            n_samples = int(pond1B_general_assumptions.loc['n_samples','expected'])
            correlation_distributions = np.full((n_samples, n_samples), np.nan)
            correlation_parameters = np.full((n_samples, 1), np.nan)
            correlation_parameters = correlation_parameters.tolist()

            pond1A_data = setup_data(pond1A_assumptions,correlation_distributions,correlation_parameters,
                                n_samples,check_variable=pond1A)
            

        if pond1B != {}:

            n_samples = int(pond1B_general_assumptions.loc['n_samples','expected'])
            correlation_distributions = np.full((n_samples, n_samples), np.nan)
            correlation_parameters = np.full((n_samples, 1), np.nan)
            correlation_parameters = correlation_parameters.tolist()

            pond1B_data = setup_data(pond1B_assumptions,correlation_distributions,correlation_parameters,
                                n_samples,check_variable=pond1B)
            

        
        calculated_pond_data = {'pond1A': None, 'pond1B': None,'Pond 1A, B': None}

        if combined_pond_data != None:

            final_combined_pond_data = run_combined_ponds(combined_pond_data)
            calculated_pond_data['Pond 1A, B'] = final_combined_pond_data

        if pond1A_data != None:

            final_pond1A_data = run_combined_ponds(pond1A_data,True)
            calculated_pond_data['pond1A'] = final_pond1A_data
        
        if pond1B_data != None:

            final_pond1B_data = run_pond1B(pond1B_data)
            calculated_pond_data['pond1B'] = final_pond1B_data
        

        
        
        html_display = []

        graph_y_axis_value = "UOD (lb/day)"
        processed_data = []
        
        key_index = 0
        key_values = ['1A (UOD 2)', '1B (UOD 3)','1A and B (UOD 3)']
        for key in list(calculated_pond_data.keys()):

            if calculated_pond_data[key] is not None :

                flattend_data = calculated_pond_data[key].flatten()

                data_median = np.median(flattend_data)
                data_text = f' {key} median: {data_median} {graph_y_axis_value}'
                html_display.append(html.H4(data_text))


                filler = []
                for i in range(len(flattend_data)):

                    filler.append(key_values[key_index])
                
                pd_data = pd.DataFrame({

                    graph_y_axis_value: flattend_data,
                    "Pond": filler
                    
                })

                pd_data.set_index(graph_y_axis_value,inplace=True)

                processed_data.append(pd_data)
            

            key_index+=1
        

        ultimate_oxygen_combined_data= pd.concat(processed_data)
        ultimate_oxygen_combined_data.reset_index(inplace=True)

        return px.violin(ultimate_oxygen_combined_data
                             ,x="Pond",y=graph_y_axis_value,color="Pond",
                             width=750,height=750,box=True), html_display


        
@app.callback(

    Output("main_content","children"),
    Output("top-image","src"),
    Input("tab_control","value")

)

def update_tab(tab_value):

    
    if tab_value == "tab_main":

        image_path = "assets/GPI-1-reduced.jpg"

        return advanced_layout,image_path
    
    elif tab_value == "tab_basic":

        image_path = "assets/GPI-2-reduced.jpg"
    
        return main_layout, image_path

    elif tab_value == "tab_info":

        image_path = "assets/GPI-1-reduced.jpg"
        
        data = {"Presentation":"#abs_pres","Assumptions":"#abs_assumptions"}
        pdf_data = ['abs_presentation.pdf',"abs_assumptions.pdf"]

        # text for the presentation and assumption file on info page
        presentation_text = "This presentation gives a general overview of the models and algorithms developed for the digital twin."
        assertions_text = "This file lists all the assumptions related to the aerated stabilization basin"

        return html.Div(children=generate_info_layout(pdf_data,[presentation_text,assertions_text]) ), image_path #,generate_sidebar_links(data)
    
    elif tab_value == "tab_example":

        image_path = "assets/GPI-2-reduced.jpg"
        video_path = "assets/demo_video.mp4"

        data = {"fake1":"#fake1","fake2":"#fake2","fake3":"#fake3"}
        return html.Div(children=[generate_demo_layout(video_path)]), image_path #, generate_sidebar_links(data)

    





if __name__ == "__main__":
    app.run(debug=False)
