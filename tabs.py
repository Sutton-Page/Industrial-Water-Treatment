from dash import Dash, html, dcc,Input,Output,State


def generate_content_row(p):

        replace = {
                'Influent_BOD': 'Biochemical oxygen demand (mg/L)',
                'com-Influent_BOD':'Biochemical oxygen demand (mg/L)',
                'com-IN_NH3_z': 'Ammonia (mg/L)',
                'com-Sup_NH3':'Ammonia Supplement (mg/l)',
                'com-IN_PO4_z':'Phosphate (mg/L)',
                'com-Sup_PO4':'Phosphate Supplement (mg/l)',
                'com-IN_Q':'Flowrate (MGD)',
                'com-aeration':'Power supplied to the ASB 1A (Watts)',
                'com-aeration_1B':'Power supplied to the ASB 1B (Watts)',
                'com-Ti':'Wastewater Temperature 1A (°F)',
                'com-air_temp':'Air Temperature 1A (°F)',
                'com-Ti_1B':'Wastewater Temperature 1B (°F)',
                'com-air_temp_1b':'Air Temperature 1B (°F)',
                'com-TSSi':'Total suspended solids (mg/L)',
                'com-Hydrau_Reten':'Hydraulic retention time (days)',
                'CBOD5':'Biochemical oxygen demand (mg/L)',
                'Hydrau_Reten': 'Hydraulic retention time (days)',
                'TSSi':'Total suspended solids (mg/L)',
                'Eff_TSS':'Total suspended solids (mg/L)',
                'air_temp':'Air Temperature (°F)',
                'air_temp_1b': 'Air Temperature (°F)',
                'Ti':'Wastewater Temperature (°F)',
                'Ti_1B':"Wastewater Temperature (°F)",
                'aeration':'Power supplied to the ASB 1A (Watts)',
                'aeration_1B':'Power supplied to the ASB 1B (Watts)',
                'IN_Q':'Flowrate (MGD)',
                'Flow_rate_3':'Flowrate (MGD)',
                'IN_NH3_z':'Ammonia (mg/L)',
                'eff_nh3_isr':'Ammonia (mg/L)',
                'Sup_NH3':'Ammonia Supplement (mg/l)',
                'IN_PO4_z':'Phosphate (mg/L)',
                'Sup_PO4':'Phosphate Supplement (mg/l)',
                'EFF_PO4':'Phosphate (mg/L)'
                }

        if p == "default":
         
              return html.Tr(children=[

                        html.Th(children=[
                                html.H4("Parameter")
                        ]),

                        html.Th(children=[

                                html.H4("Low")
                        ]),

                        html.Th(children=[

                                html.H4("Expected")
                        ]),

                        html.Th(children=[

                                html.H4("High")
                        ]),

                ])  
        
        else:
         
                return  html.Tr(children=[

                        
                        html.Td(children=[

                                        html.H4(replace[p.replace("default-",'')])
                                        ]),

                                html.Td(children=[
                                        
                                        dcc.Input(id=p+"-low")
                                ]),

                                html.Td(children=[
                                        
                                        dcc.Input(id=p+"-expected")
                                ]),

                                html.Td(children=[
                                        
                                        dcc.Input(id=p+"-high")
                                ]),

        ])
                         
        
        

def generate_tables(data,idx):
        letters = ["A","B","C","D"]

        return html.Div(children=[

                html.H4("Pond 1" + letters[idx]),
                html.Table(children=[generate_content_row(p) for p in data])
        ])
           
        #return html.Table(children=[generate_content_row(p) for p in data])


          
def experiment(para_items,data):

        f= open("main-page-instructions.txt")
        main_page_instructions = f.read()
        f.close()

        graph_message = "The results for UOD in the effluent of the ASB are shown below."
        graph_message2 = "Note these results can also be downloaded as an Excel file"
        graph_warning = "Note: If the graph doesn't load within 30 seconds then click submit again to retry"

        return html.Div(className="Z0",children=[

                html.Div(className="special",children=[

                        html.Div(className="main-paragraph",children=[

                                html.P(main_page_instructions)
                        ]),
                        html.H4("Parameters to calculate Ultimate Oxygen Demand"),

                        html.Div(className="button-spacer",children=[

                                html.Button("Load Defaults: Pond 1A",id="load-default"),
                                html.Button("Clear Defaults: Pond 1A",id="clear-default"),
                                html.Button("Load Defaults: Pond 1B","load-default-B"),
                                html.Button("Clear Defaults: Pond 1B",id="clear-B")

                        ]),

                      
                ]),

                

                html.Div(className="Z0",children=[generate_tables(t,idx) for idx,t in enumerate(para_items[0:2])]),

                html.Div(className="special",children=[

                        

                        

                        html.Div(children=[

                        html.Div(children=[

                        html.H4("Combined Ponds 1A and B"),
                        html.Div(className="button-spacer",children=[

                                html.Button("Load Defaults: Pond 1A and B",id="com-load"),
                                html.Button("Clear Defaults: Pond 1A and B",id="com-clear"),
                                

                        ]),
                        html.Table(children=[generate_content_row(p) for p in para_items[-1]])
                        ])
                ]),

                      
                ]),
                

                
                html.Div(className="special-button",children=[

                        html.Button("Submit",id="submit-params"),

                            
                ]),




                html.Div(className="compact",children=[

                        html.H2("Plot of Ultimate Oxygen Demand"),
                        html.H4(graph_message),
                        html.H4(graph_message2),
                        html.H4(graph_warning,className="compact-warn"),
                        html.Button("Download data as Excel file",id="excel-button"),
                        dcc.Download(id="download-excel"),
                     

                ]),

               
                html.Div(children=[

                        html.Div(id="display-median"),

                        html.Div(className="center-graph",children=[

                                dcc.Loading(children=[
                                        
                                        dcc.Graph("violin-graph")
                                ])
                         ]),
                ])
        ])


def generate_basis_input_layout():

        graph_message = "The results for UOD in the effluent of the ASB are shown below."
        graph_message2 = "Note these results can also be downloaded as an Excel file"
        graph_warning = "Note: If the graph doesn't load within 30 seconds then click submit again to retry"
        instructions = open("./basic-page-instructions.txt",'r').read()

        return html.Div(children=[

                html.Div(className="basic-paragraph",children=[

                                html.P(instructions)
                        ]),

                html.Div(className="basic-content", children=[

                        html.Table(children=[
                                html.Tr(children=[

                                        html.Td(html.H4("Biochemical oxygen demand (mg/L)")),
                                        html.Td(dcc.Input(id="basic-BOD",type='number')),

                                        html.Td(html.H4("Phosphate (mg/L)"),),
                                        html.Td(dcc.Input(id="basic-PO4",type='number')),

                                        html.Td(html.H4("Phosphate Supplement (mg/l)"),),
                                        html.Td(dcc.Input(id="basic-Sup_PO4",type='number')),

                                        html.Td(html.H4("Ammonia (mg/l)"),),
                                        html.Td(dcc.Input(id="basic-IN_NH3_z",type='number')),

                                        



                                ]),

                                html.Tr(children=[

                                
                                        html.Td(html.H4("Ammonia Supplement (mg/l)"),),
                                        html.Td(dcc.Input(id="basic-Sup_NH3",type='number')),


                                        html.Td(html.H4("Total suspended solids (mg/L) "),),
                                        html.Td(dcc.Input(id="basic-TSSi",type='number')),

                                        html.Td(html.H4("Power supplied to ASB 1A (Watts)"),),
                                        html.Td(dcc.Input(id="basic-aeration",type='number')),

                                        html.Td(html.H4("Power supplied to ASB 1B (Watts)"),),
                                        html.Td(dcc.Input(id="basic-aeration_1B",type='number')),

                                ]),

                                html.Tr(children=[

                                       
                                        html.Td(html.H4("Hydraulic retention time (days)"),),
                                        html.Td(dcc.Input(id="basic-Hydrau_Reten",type='number')),

                                        html.Td(html.H4("Flow rate (MGD)."),),
                                        html.Td(dcc.Input(id="basic-IN_Q",type='number')),

                                        html.Td(html.H4("Wastewater Temperature 1A"),),
                                        html.Td(dcc.Input(id="basic-Ti",type='number')),

                                        html.Td(html.H4("Air Temperature 1A"),),
                                        html.Td(dcc.Input(id="basic-air_temp",type='number')),

                                ]),

                                html.Tr(children=[
                                        

                                        html.Td(html.H4("Wastewater Temperature 1B"),),
                                        html.Td(dcc.Input(id="basic-Ti_1B",type='number')),

                                         html.Td(html.H4("Air Temperature 1B"),),
                                        html.Td(dcc.Input(id="basic-air_temp_1b",type='number')),

                                ])

                        
                        ]),

                        

                ]),

                html.Div(className="basic-button-content", children=[

                        html.Button("Submit",id="basic-submit")
                ]),

                

                html.Div(className="basic-message",children=[

                        html.H2("Plot of Ultimate Oxygen Demand"),
                        html.H4(graph_message),
                        html.H4(graph_warning,className="compact-warn"),
                

                ]),

                html.Div(className="basic-graph-content",children=[

                        html.Div(id="display-median-basic"),

                        dcc.Loading(children=[
                                
                                 dcc.Graph(id="basic-graph")
                        ])
                ]),

                



                

        ])

def generate_main_layout_v2():
        
        return html.Div(className="Z0",children=[
                
                
                html.Div(className="Z1",children=[
                        
                        html.Div(className="Z2",children=[
                                html.H4("NH4_Ni")
                        ]),

                        html.Div(className="Z2",children=[
                            
                            html.H4("Low"),
                            html.H4("Expected"),
                            html.H4("High"),
                           
                                
                        ]),

                        html.Div(className="Z2-special",children=[
                                dcc.Input(),
                                dcc.Input(),
                                dcc.Input()
                        ])
                        
                ]),

                html.Div(className="Z1",children=[
                        
                        html.Div(className="Z2",children=[
                                html.H4("Q")
                        ]),

                        html.Div(className="Z2",children=[
                            
                            html.H4("Low"),
                            html.H4("Expected"),
                            html.H4("High"),
                           
                                
                        ]),

                        html.Div(className="Z2-special",children=[
                                dcc.Input(),
                                dcc.Input(),
                                dcc.Input()
                        ])
                        
                ]),

                html.Div(className="Z1",children=[
                        
                        html.Div(className="Z2",children=[
                                html.H4("Y")
                        ]),

                        html.Div(className="Z2",children=[
                            
                            html.H4("Low"),
                            html.H4("Expected"),
                            html.H4("High"),
                           
                                
                        ]),

                        html.Div(className="Z2-special",children=[
                                dcc.Input(),
                                dcc.Input(),
                                dcc.Input()
                        ])
                        
                ]),

                html.Div(className="Z1",children=[
                        
                        html.Div(className="Z2",children=[
                                html.H4("K")
                        ]),

                        html.Div(className="Z2",children=[
                            
                            html.H4("Low"),
                            html.H4("Expected"),
                            html.H4("High"),
                           
                                
                        ]),

                        html.Div(className="Z2-special",children=[
                                dcc.Input(),
                                dcc.Input(),
                                dcc.Input()
                        ])
                        
                ])



                
        
        
    ]) 
        


def generate_demo_layout(video_link):

        return html.Div(className="demo_container",children=[

                html.Video(src=video_link,controls=True)

        ])

def generate_info_layout(pdf_links,text_content):

        return html.Div(children=[

                html.Div(className="info-page",children=[

                html.H2("Presentation",id="abs_pres"),
                html.H3(text_content[0]),

                ]),

                html.Iframe(src="assets/"+pdf_links[0]),

                html.Div(className="info-page",children=[

                html.H2("Assumptions",id="abs_assumptions"),
                html.H3(text_content[1]),
                

                ]),

                html.Iframe(src="assets/"+pdf_links[1])

        ])
        
        ''''
        return [
                html.H2("Presentation",id="abs_pres"),
                html.Iframe(src="assets/"+pdf_links[0]),
                html.H2("Assumptions",id="abs_assumptions"),
                html.Iframe(src="assets/"+pdf_links[1])
                
        ]'''

        
        
        
        

def generate_layout_main(low_COD_influent,default_COD_influent,high_COD_influent):


        return html.Div(children=[
                
            html.H2("Influent_COD",id="influent_L", className="centered-container"),

            html.Div(className='centered-container', children=[
            html.Div(children=[
                html.H2("Low"),
                dcc.Input(id="influent_low_state", value=low_COD_influent),
            ]),
            
            html.Div(children=[
                html.H2("Expected"),
                dcc.Input(id="influent_state", value=default_COD_influent),
            ]),
            
            html.Div(children=[
                html.H2("High"),
                dcc.Input(id="influent_high_state", value=high_COD_influent),
                html.Button("Submit", id="influent_submit", n_clicks=0),
            ]),
        
        
            
        ]),

        # add infleunt_fig  back latter
        dcc.Loading(children=[

            dcc.Graph("influent_graph"),

        ]),
        
        html.H2("Uncertainty graph",id="uncertain_L",className="lazy"),

        html.Div(className="centered-container",children=[

            dcc.Loading(
                id="uncertain_loading",
                children=[
                    # add uncertain_fig later
                    dcc.Graph(id="uncertain_graph")
                ]
            )

        ]),
        
        
        html.Div(className='centered-container', children=[

        
            html.H2("Sensitivity",id="sensitive_L",),
        ]),

        dcc.Loading(children=[
            # add sensitivity_fig back later
            dcc.Graph("sensitivity_graph")

        ])
        
        
    ]) 

    



