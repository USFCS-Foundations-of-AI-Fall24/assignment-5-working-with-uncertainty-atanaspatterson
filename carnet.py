from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination




car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent", "Starts")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_starts = TabularCPD(variable='Starts', variable_card=2, 
                        values=[[0.99, 0.9, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0], 
                                [0.01, 0.1, 0.1, 1.0, 0.1, 1.0, 1.0, 1.0]], 
                        evidence=['Ignition', 'Gas', 'KeyPresent'], 
                        evidence_card=[2, 2, 2], state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent": ["yes", "no"]}) 




cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

# **(5 points)**  3. Last, we will add an additional node to the network, called KeyPresent, that indicates whether or not we have the key for the car.
#   This is a Categorical variable with two state values, yes and no. The prior for 'yes' is 0.7.
  
#   KeyPresent should only affect Starts. Add an edge to starts and update the CPD to indicate that:
# <pre>
# P(starts | gas, ignition, keyPresent) = 0.99
# P(starts | gas, !ignition, keyPresent) = 0.01
# P(starts | !gas, ignition, keyPresent) = 0.01
# P(starts | gas, ignition, !keyPresent) = 0.01
# P(starts | !gas, !ignition, keyPresent) = 0.01
# P(starts | !gas, ignition, !keyPresent) = 0.01
# P(starts | gas, !ignition, !keyPresent) = 0.01 
# P(starts | !gas, !ignition, !keyPresent) = 0.01
# </pre>

cpd_keypresent = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]], 
    state_names={"KeyPresent" :["yes", "no"]}
)

car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(car_model)

def main():

    print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))
    print("Given that the car will not move, what is the probability that the battery is not working?")
    print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))
    print("Given that the radio is not working, what is the probability that the car will not start?")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))
    print("Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it? (fitst without gas observation, then with gas)")
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works"}))
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}))
    print("Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it? (first without gas observation, then with gas)")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no"}))
    print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}))
    print("What is the probability that the car starts if the radio works and it has gas in it?")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))
    print("Add a query showing the probability that the key is not present given that the car does not move")
    print(car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"}))





if __name__ == "__main__":
    main()

