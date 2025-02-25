import warnings
# Remove anoying warning
warnings.filterwarnings("ignore")

from pandas import isna
import lhs



def setup_data(assumption_data,correlation_distributions,correlation_parameters,
               n_samples,check_variable=None,replace_series=None):

    data_holder = {}

    for assumption in assumption_data:

        for data in assumption.index:

            if isna(data) == False:

                row = assumption.loc[data]
              
                # cleaning the variable names to ensure they are valid python variable names
                data = data.replace(' ','_')
                data = data.replace('.','_')

                if row.__contains__('distribution'):

                    if isna(row['distribution']) != True:

                        if check_variable !=None:
                            
                            '''
                            if data == check_variable:

                                row = replace_series'''
                            if check_variable.keys().__contains__(data):


                                root = check_variable[data]

                                row = row.replace(to_replace=row['low'],value=root['low'])
                                row = row.replace(to_replace=row['expected'],value=root['expected'])
                                row = row.replace(to_replace=row['high'],value=root['high'])

                                
                                '''
                                row.loc['low'] = check_variable[data]['low']
                                row.loc['expected'] = check_variable[data]['expected']
                                row.loc['high'] = check_variable[data]['high']
                                '''
                                

                        data_holder[data], correlation_distributions, correlation_parameters = lhs.lhs_distribution(row,correlation_distributions,correlation_parameters, n_samples)

                    else:

                        data_holder[data] = row
            
                else:

                    data_holder[data] = row
                        
    
    return data_holder



    

    
